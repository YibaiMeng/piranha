/*
 * DeviceData.h
 * ----
 * 
 * Top-level class for managing/manipulating GPU data on-device.
 */

/*
DeviceData<uint64_t> actual_data(10); // allocate device buffer of size 10, actual_data.begin() is 1st element, .end() is last element
DeviceData<uint64_t> actual_data(10, 5); // allocate device buffer of size 10 precision 5, actual_data.begin() is 1st element, .end() is last element

// actual_data[0] TODO
// copy operation -> std::vector<T> on host
// print for DeviceData is different from print for e.g. RSS

DeviceData<uint64_t> actual_data; // TODO 0 sized

// I and SRIterator comes from somewhere
StridedRange<I> odds(actual_data.begin() + 1, actual_data.end(), 2); // iterator over idx 1, 3, 5, ...
StridedRange<I> superodds(odds.begin(), odds.end(), 2); // iterator over idx 1, 5, 9

DeviceData<T, SRIterator> getEvens(DeviceData<T, Iterator> &actual_data) {
    StridedRange<I> evens(actual_data.begin(), actual_data.end(), 2); // iterator over idx 0, 2, 4, ...
    DeviceData<uint64_t, SRIterator> evenView(evens.begin(), evens.end()); // NO device buffer allocation, .begin() is idx 0 of actual_data and .end() is idx 8 of actual_data
    return evenView;
}

auto evenView = getEvens(actual_data);

*/

#pragma once

#include <thread>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include "util/connect.h"
#include "util/Profiler.h"
#include "gpu/functors.cuh"
#include "gpu/gpu.h"

#include <loguru.hpp>

extern BmrNet **communicationSenders;
extern BmrNet **communicationReceivers;
extern CommunicationObject commObject;

extern Profiler comm_profiler;
extern Profiler memory_profiler;

// TODO: put DeviceDataBase into .c file?
extern std::vector<void*> send_buffer_per_pipeline_group, recv_buffer_per_pipeline_group;

/// @brief Convert bytes to MiB.
template<typename T>
static double b_to_mib(T b) {
    return (double) b / 1024.0 / 1024.0;
}

template<typename T>
void initializeDeviceDataCache() {
    if(interparty_comm_buffer_mode == INTERPARTY_COMM_BUFFER_MODE_CACHED_PIN) {
        send_buffer_per_pipeline_group.resize(PIPELINE_GROUPS);
        recv_buffer_per_pipeline_group.resize(PIPELINE_GROUPS);
        for(int idx = 0; idx < PIPELINE_GROUPS; idx++) {
            void *send_cache_ptr, *recv_cache_ptr;
            CUDA_CHECK(cudaMallocHost((void**)&recv_cache_ptr, DEVICE_DATA_RECV_CACHE_SIZE / PIPELINE_GROUPS));
            CUDA_CHECK(cudaMallocHost((void**)&send_cache_ptr, DEVICE_DATA_SEND_CACHE_SIZE / PIPELINE_GROUPS));
            send_buffer_per_pipeline_group[idx] = send_cache_ptr;
            recv_buffer_per_pipeline_group[idx] = recv_cache_ptr;
        }
        LOG_S(INFO) << b_to_mib((uint64_t)DEVICE_DATA_RECV_CACHE_SIZE +  (uint64_t)DEVICE_DATA_SEND_CACHE_SIZE) << " MiB of pinned memory initialized.";
    }
}

template<typename T>
void freeDeviceDataCache() {
    if(interparty_comm_buffer_mode == INTERPARTY_COMM_BUFFER_MODE_CACHED_PIN) {
        for(void* ptr : send_buffer_per_pipeline_group) {
            CUDA_CHECK(cudaFreeHost(ptr));
        }
        for(void* ptr : recv_buffer_per_pipeline_group) {
            CUDA_CHECK(cudaFreeHost(ptr));
        }
        LOG_S(INFO) << b_to_mib((uint64_t)DEVICE_DATA_RECV_CACHE_SIZE +  (uint64_t)DEVICE_DATA_SEND_CACHE_SIZE) << " MiB of pinned memory freed.";
    }
}

template<typename T>
using BufferIterator = thrust::detail::normal_iterator<thrust::device_ptr<T> >;

template<typename T, typename Iterator>
class DeviceDataBase {

    protected:

        DeviceDataBase() : transmitting(false) {
            CUDA_CHECK(cudaGetDevice(&(this->cuda_device_id)));
            pinnedHostBufferSize = 0;
            
        }

        DeviceDataBase(Iterator _first, Iterator _last) :
            first(_first), last(_last), transmitting(false) {
            CUDA_CHECK(cudaGetDevice(&(this->cuda_device_id)));
            pinnedHostBufferSize = 0;

        }

        ~DeviceDataBase() {
            if(pinnedHostBufferSize > 0) CUDA_CHECK(cudaFreeHost(pinnedHostBuffer));
        }
    public:

        Iterator begin() const {
            return first;
        }

        Iterator end() const {
            return last;
        }

        size_t size() const {
            return end() - begin();
        }

        int cudaDeviceID() const {
            return this->cuda_device_id;
        }

        void set(Iterator _first, Iterator _last) {
            first = _first; last = _last;
        }

        void zero() {
            CUDA_CHECK(cudaSetDevice(this->cuda_device_id));
            THRUST_CHECK(thrust::fill(begin(), end(), static_cast<T>(0)));
        }

        void fill(T val) {
            CUDA_CHECK(cudaSetDevice(this->cuda_device_id));
            THRUST_CHECK(thrust::fill(begin(), end(), val));
        }

        void transmit(size_t party, bool use_pinned=false) {
            CUDA_CHECK(cudaSetDevice(this->cuda_device_id));
            if (rtxThread.joinable()) {
                throw std::runtime_error("DeviceBuffer tx failed: already transmitting or receiving");
            }

            comm_profiler.add_comm_bytes(size() * sizeof(T), true);
            if(!use_pinned) {
                hostBuffer.resize(size());
                thrust::copy(begin(), end(), hostBuffer.begin());
            } else {
                // This is very inefficient
                if(pinnedHostBufferSize < size()) {
                    if(pinnedHostBufferSize > 0) CUDA_CHECK(cudaFreeHost(pinnedHostBuffer));
                    CUDA_CHECK(cudaMallocHost((void**)&pinnedHostBuffer, size() * sizeof(T)));
                    pinnedHostBufferSize = size();
                }
            }
            // transmit
            transmitting = true;
            if(!use_pinned) {
                rtxThread = std::thread(sendVector<T>, party, hostBuffer.data(), size(), pipeline_id);
            } else {
                rtxThread = std::thread(sendVector<T>, party, pinnedHostBuffer, size(), pipeline_id);
            }
        }

        // Alternative version without allocating paged host memory.
        void transmit_no_malloc(size_t party) {
            CUDA_CHECK(cudaSetDevice(this->cuda_device_id));
            if (rtxThread.joinable()) {
                throw std::runtime_error("DeviceBuffer tx failed: already transmitting or receiving");
            }
            comm_profiler.add_comm_bytes(size() * sizeof(T), true);
            transmitting = true;
            rtxThread = std::thread([&]() {
                LOG_SCOPE_F(1, "Sending %.4f MiB using pre-allocated pinned memory", b_to_mib(size() * sizeof(T)));
                std::stringstream ss;
                ss << "Send " << party << " @ " << pipeline_id;
                loguru::set_thread_name(ss.str().c_str());
                // Important: all sizes are in BYTES, as there's no promise a socket read/write would align to size(T).
                // Sending message through this FD
                int fd = communicationSenders[party]->socketFd[pipeline_id];
                DLOG_S(2) << "Socket " << fd;
                int64_t cache_size = (int64_t)DEVICE_DATA_SEND_CACHE_SIZE / (int64_t)PIPELINE_GROUPS;
                DLOG_S(2) << "Cache size for transmission is " << b_to_mib(cache_size) << " MiB";
                T* cache_ptr = (T*)send_buffer_per_pipeline_group.at(pipeline_id);
                T* src_ptr = thrust::raw_pointer_cast(&(begin()[0]));
                int64_t left = size() * sizeof(T);
                cudaEvent_t memcpy_event;
                while(left > 0) {
                    DLOG_S(2) << b_to_mib(left) << "MiB left to transmit";
                    CUDA_CHECK(cudaEventCreateWithFlags(&memcpy_event, cudaEventBlockingSync));
                    int trans_size = std::min((int64_t)cache_size, (int64_t)left);
                    LOG_S(2) << "Starting " << b_to_mib(trans_size) << " MiB DtoH (pinned) transfer"; 
                    CUDA_CHECK(cudaMemcpyAsync((void*)cache_ptr, (void*)src_ptr, trans_size, cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaEventRecord(memcpy_event));
                    CUDA_CHECK(cudaEventSynchronize(memcpy_event));
                    LOG_S(2) << "Finishing " << b_to_mib(trans_size) << " MiB DtoH (pinned) transfer"; 
                    left -= trans_size;
                    int64_t socket_transmitted = 0;
                    ssize_t write_status = -1;
                    while(socket_transmitted < trans_size) {
                        // We attempt to transmit trans_size everytime, but `write` may not be able to do that.
                        // So we might need to try multiple times.
                        DLOG_S(2) << "Trying to send " << b_to_mib(trans_size - socket_transmitted) << " MiB over socket"; 
                        write_status = write(fd, (void*)((char*)(cache_ptr) + socket_transmitted), trans_size - socket_transmitted);
                        if(write_status < 0) {
                            LOG_F(ERROR, "Cannot write to TCP socket, errno %i", write_status);
                            return;
                        }
                        DLOG_S(2) << "Actually send " << b_to_mib(write_status) << " MiB over socket"; 
                        CHECK_F(write_status <= trans_size - socket_transmitted, "Transmitted more data than requested");
                        socket_transmitted += write_status;
                    }
                    DLOG_S(2) << "Still " << b_to_mib(left) << " MiB left to send";
                }
                DLOG_S(2) << "All finished"; 
                commObject.incrementSent(size());
            });
        }

        void receive(size_t party, bool use_pinned=false) {
            CUDA_CHECK(cudaSetDevice(this->cuda_device_id));
            if (rtxThread.joinable()) {
                throw std::runtime_error("DeviceBuffer rx failed: already transmitting or receiving");
            }

            comm_profiler.add_comm_bytes(size() * sizeof(T), false);

            if(!use_pinned) {
                hostBuffer.resize(size());
            } else {
                if(pinnedHostBufferSize < size()) {
                    if(pinnedHostBufferSize > 0) CUDA_CHECK(cudaFreeHost(pinnedHostBuffer));
                    CUDA_CHECK(cudaMallocHost((void**)&pinnedHostBuffer, size() * sizeof(T)));
                    pinnedHostBufferSize = size();
                }
            }

            transmitting = false;
            //receiveVector<T>(party, hostBuffer);
            if(!use_pinned) {
                rtxThread = std::thread(receiveVector<T>, party, hostBuffer.data(), size(), pipeline_id);
            } else {
                rtxThread = std::thread(receiveVector<T>, party, pinnedHostBuffer, size(), pipeline_id);
            }
        }


        // Alternative version without allocating paged host memory.
        void receive_no_malloc(size_t party) {
            CUDA_CHECK(cudaSetDevice(this->cuda_device_id));
            if (rtxThread.joinable()) {
                throw std::runtime_error("DeviceBuffer rx failed: already transmitting or receiving");
            }
            comm_profiler.add_comm_bytes(size() * sizeof(T), false);
            transmitting = false;
            rtxThread = std::thread([&](){
                
                std::stringstream ss;
                ss << "Recv " << party << " @ " << pipeline_id;
                loguru::set_thread_name(ss.str().c_str());
                LOG_SCOPE_F(1, "Receiving %.4f MiB using pre-allocated pinned memory", b_to_mib(size() * sizeof(T)));
                // Important: all sizes are in BYTES, as there's no promise a socket read/write would align to size(T).
                // Receiving message through this FD
                int fd = communicationReceivers[party]->socketFd[pipeline_id];
                DLOG_S(2) << "Socket " << fd;
                T* cache_ptr = (T*)recv_buffer_per_pipeline_group.at(pipeline_id);
                int64_t cache_size = (int64_t)(((double)DEVICE_DATA_RECV_CACHE_SIZE / (double)PIPELINE_GROUPS));
                DLOG_S(2) << "Cache size for transmission is " << b_to_mib(cache_size) << " MiB";
                // The number of bytes in the cache that is currently valid. 
                int64_t cache_filled = 0;         
                T* dst_ptr = thrust::raw_pointer_cast(&(begin()[0]));
                // How much of the device destination memory have been filled so far.
                int64_t dst_filled = 0;
                int64_t socket_recv = 0;
                cudaEvent_t memcpy_event;            
                while(socket_recv < size() * sizeof(T)) {
                    DLOG_S(2) << b_to_mib(socket_recv) << " MiB recv";
                    int trans_size = std::min(cache_size - cache_filled, (int64_t)(size() * sizeof(T)) - socket_recv);
                    // Amount to receive this iteration.                
                    LOG_S(2) << "Trying to recv " << b_to_mib(trans_size) << " MiB from socket"; 
                    ssize_t read_status = -1;
                    read_status = read(fd, (void*)((char*)(cache_ptr) + cache_filled), trans_size);
                    if(read_status < 0) {
                        LOG_F(ERROR, "Cannot read from TCP socket, errno %i", read_status);
                        return;
                    }
                    LOG_S(2) << "Actually recv " << b_to_mib(read_status) << " MiB from socket"; 
                    CHECK_F(read_status <= trans_size, "Received more data that asked for.");

                    cache_filled += read_status;
                    socket_recv += read_status;
                    // Now either cache is filled or everything needed has arrived
                    // We send it to the device.
                    if(socket_recv == size() * sizeof(T)  || cache_filled == cache_size) {
                        LOG_S(2) << "Starting " << b_to_mib(trans_size) << " MiB HtoD (pinned) transfer"; 
                        CUDA_CHECK(cudaEventCreateWithFlags(&memcpy_event, cudaEventBlockingSync));
                        CUDA_CHECK(cudaMemcpyAsync((void*)((char*)(dst_ptr) + dst_filled), (void*)cache_ptr, cache_filled, cudaMemcpyHostToDevice));
                        CUDA_CHECK(cudaEventRecord(memcpy_event));
                        CUDA_CHECK(cudaEventSynchronize(memcpy_event));
                        LOG_S(2) << "Finishing " << b_to_mib(trans_size) << " MiB HtoD (pinned) transfer"; 
                        dst_filled += cache_filled;
                        cache_filled = 0;
                    }
                    LOG_S(2) << "Still " << b_to_mib(size() * sizeof(T) - socket_recv) << " MiB left to recv";
                }
                LOG_S(2) << "Finished"; 
                commObject.incrementRecv(size());
            });
        }

        void join(bool use_pinned=false) {
            CUDA_CHECK(cudaSetDevice(this->cuda_device_id));
            if (!rtxThread.joinable()) return;
            
            rtxThread.join();
            if (!transmitting) {
                if(!use_pinned) {
                    THRUST_CHECK(thrust::copy(hostBuffer.begin(), hostBuffer.end(), begin()));
                    std::vector<T>().swap(hostBuffer); // clear buffer
                } else {
                    THRUST_CHECK(thrust::copy(pinnedHostBuffer, pinnedHostBuffer + size(), begin()));
                }
            }
        }

        void join_no_malloc() {
            CUDA_CHECK(cudaSetDevice(this->cuda_device_id));
            if (!rtxThread.joinable()) return;
            rtxThread.join();
        }
        
        // scalar overloads
        DeviceDataBase<T, Iterator> &operator+=(const T rhs) {
            CUDA_CHECK(cudaSetDevice(this->cuda_device_id));
            THRUST_CHECK(thrust::transform(begin(), end(), begin(), scalar_plus_functor<T>(rhs)));
            return *this;
        }

        DeviceDataBase<T, Iterator> &operator-=(const T rhs) {
            CUDA_CHECK(cudaSetDevice(this->cuda_device_id));
            THRUST_CHECK(thrust::transform(begin(), end(), begin(), scalar_minus_functor<T>(rhs)));
            return *this;
        }

        DeviceDataBase<T, Iterator> &operator*=(const T rhs) {
            CUDA_CHECK(cudaSetDevice(this->cuda_device_id));
            THRUST_CHECK(thrust::transform(begin(), end(), begin(), scalar_mult_functor<T>(rhs)));
            return *this;
        }
        
        DeviceDataBase<T, Iterator> &operator/=(const T rhs) {
            CUDA_CHECK(cudaSetDevice(this->cuda_device_id));
            THRUST_CHECK(thrust::transform(begin(), end(), begin(), scalar_divide_functor<T>(rhs)));
            return *this;
        }

        DeviceDataBase<T, Iterator> &operator>>=(const T rhs) {
            CUDA_CHECK(cudaSetDevice(this->cuda_device_id));
            THRUST_CHECK(thrust::transform(begin(), end(), begin(), scalar_arith_rshift_functor<T>(rhs)));
            return *this;
        }

        DeviceDataBase<T, Iterator> &operator<<=(const T rhs) {
            CUDA_CHECK(cudaSetDevice(this->cuda_device_id));
            THRUST_CHECK(thrust::transform(begin(), end(), begin(), scalar_lshift_functor<T>(rhs)));
            return *this;
        }

        // vector overloads
        template<typename I2>
        DeviceDataBase<T, Iterator> &operator+=(const DeviceDataBase<T, I2> &rhs) {
            CUDA_CHECK(cudaSetDevice(this->cuda_device_id));
            THRUST_CHECK(thrust::transform(this->begin(), this->end(), rhs.begin(), this->begin(), thrust::plus<T>()));
            return *this;
        }

        template<typename I2>
        DeviceDataBase<T, Iterator> &operator-=(const DeviceDataBase<T, I2> &rhs) {
            CUDA_CHECK(cudaSetDevice(this->cuda_device_id));
            THRUST_CHECK(thrust::transform(this->begin(), this->end(), rhs.begin(), this->begin(), thrust::minus<T>()));
            return *this;
        }

        template<typename I2>
        DeviceDataBase<T, Iterator> &operator*=(const DeviceDataBase<T, I2> &rhs) {
            CUDA_CHECK(cudaSetDevice(this->cuda_device_id));
            THRUST_CHECK(thrust::transform(this->begin(), this->end(), rhs.begin(), this->begin(), thrust::multiplies<T>()));
            return *this;
        }

        template<typename I2>
        DeviceDataBase<T, Iterator> &operator/=(const DeviceDataBase<T, I2> &rhs) {
            CUDA_CHECK(cudaSetDevice(this->cuda_device_id));
            THRUST_CHECK(thrust::transform(this->begin(), this->end(), rhs.begin(), this->begin(), signed_divide_functor<T>()));
            return *this;
        }

        template<typename I2>
        DeviceDataBase<T, Iterator> &operator^=(const DeviceDataBase<T, I2> &rhs) {
            CUDA_CHECK(cudaSetDevice(this->cuda_device_id));
            THRUST_CHECK(thrust::transform(this->begin(), this->end(), rhs.begin(), this->begin(), thrust::bit_xor<T>()));
            return *this;
        }

        template<typename I2>
        DeviceDataBase<T, Iterator> &operator&=(const DeviceDataBase<T, I2> &rhs) {
            CUDA_CHECK(cudaSetDevice(this->cuda_device_id));
            THRUST_CHECK(thrust::transform(this->begin(), this->end(), rhs.begin(), this->begin(), thrust::bit_and<T>()));
            return *this;
        }

        template<typename I2>
        DeviceDataBase<T, Iterator> &operator>>=(const DeviceDataBase<T, I2> &rhs) {
            CUDA_CHECK(cudaSetDevice(this->cuda_device_id));
            THRUST_CHECK(thrust::transform(this->begin(), this->end(), rhs.begin(), this->begin(), arith_rshift_functor<T>()));
            return *this;
        }

        template<typename I2>
        DeviceDataBase<T, Iterator> &operator<<=(const DeviceDataBase<T, I2> &rhs) {
            CUDA_CHECK(cudaSetDevice(this->cuda_device_id));
            THRUST_CHECK(thrust::transform(this->begin(), this->end(), rhs.begin(), this->begin(), lshift_functor<T>()));
            return *this;
        }

    protected:

        Iterator first;
        Iterator last;
        int cuda_device_id;

    private:

        bool transmitting;
        T* pinnedHostBuffer;
        size_t pinnedHostBufferSize;
        // Note: the contents of the host buffer MAY NOT always be sychronized to the device buffer.
        std::vector<T> hostBuffer;
        std::thread rtxThread;
};

template<typename T, typename I = BufferIterator<T> >
class DeviceData : public DeviceDataBase<T, I> {

    public:

        DeviceData(I _first, I _last) : DeviceDataBase<T, I>(_first, _last) {}
};

template<typename T>
class DeviceData<T, BufferIterator<T> > : public DeviceDataBase<T, BufferIterator<T> > {

    public:

        DeviceData() : data(0) {
            LOG_S(2) << "Empty DeviceData initialized on GPU "  << this->cudaDeviceID();
            // set iterators after data is initialized
            this->set(data.begin(), data.end());
        }

        // A DeviceData<T, BufferIterator<T>> constructed this way does not store any data. 
        // It only stores the Iterators that points to the thrust device vector.
        // Therefore, if *_first got deconstructed, there will be problems.
        DeviceData(BufferIterator<T> _first, BufferIterator<T> _last) :
                data(0),
                DeviceDataBase<T, BufferIterator<T> >(_first, _last) {}

        ~DeviceData() {
            LOG_S(2) << "Device Data of size " << this->size() * sizeof(T) << " freed on GPU "  << this->cudaDeviceID();
            memory_profiler.track_free(data.size() * sizeof(T), this->cuda_device_id);
        }

        DeviceData(int n) : data(n) {
            LOG_S(2) << "Device Data of size " << n * sizeof(T) << " initialized on GPU "  << this->cudaDeviceID();
            this->set(data.begin(), data.end());
            memory_profiler.track_alloc(n * sizeof(T), this->cuda_device_id);
        }

        DeviceData(std::initializer_list<T> il) : data(il.size()) {
            CUDA_CHECK(cudaSetDevice(this->cuda_device_id));
            THRUST_CHECK(thrust::copy(il.begin(), il.end(), data.begin()));
            this->set(data.begin(), data.end());

            memory_profiler.track_alloc(il.size() * sizeof(T), this->cuda_device_id);
        }

        void resize(size_t n) {
            CUDA_CHECK(cudaSetDevice(this->cuda_device_id));
            memory_profiler.track_free(data.size() * sizeof(T), this->cuda_device_id);
            data.resize(n);
            memory_profiler.track_alloc(n * sizeof(T), this->cuda_device_id);
            this->set(data.begin(), data.end());
        }

        thrust::device_vector<T> &raw() {
            return data;
        }

        // Transfer the contents of this device buffer to another device buffer on
        // Note: does not allow casting yet.
        int copyToDevice(DeviceData<T, BufferIterator<T> >& dst_buffer) {
            CUDA_CHECK(cudaSetDevice(this->cuda_device_id));
            if(dst_buffer.size() != this->size()) {
                LOG_S(FATAL) << "Size of destination DeviceData different from source DeviceData.";
            }
            LOG_S(1) << "Copying " << dst_buffer.size() * sizeof(T) << " bytes of data from " << this->cudaDeviceID() << " to " << dst_buffer.cudaDeviceID();

            // first, assert the dest_buffer have enough space.
            T* dst_ptr = thrust::raw_pointer_cast(dst_buffer.data.data());
            T* src_ptr = thrust::raw_pointer_cast(data.data());
            comm_profiler.add_intergpu_comm_bytes(sizeof(T) * this->size(), this->cudaDeviceID(), dst_buffer.cudaDeviceID());
            CUDA_CHECK(cudaMemcpyPeer((void*)dst_ptr, dst_buffer.cudaDeviceID(), (void*)src_ptr, this->cudaDeviceID(), sizeof(T) * this->size()));
            return -1;
        }        

        void copyAsync(DeviceData& dst_buffer, cudaStream_t stream) {
             CUDA_CHECK(cudaSetDevice(this->cuda_device_id));
             if(dst_buffer.size() != this->size()) {
                LOG_S(FATAL) << "Size of destination DeviceData different from source DeviceData.";
             }
             LOG_S(1) << "Initiate asynchronous copying of " << dst_buffer.size() * sizeof(T) << " bytes of data from " << this->cudaDeviceID() << " to " << dst_buffer.cudaDeviceID();
             // We must cast the iterator, instead of casting data.data(), as data may be empty.
             T* dst_ptr = thrust::raw_pointer_cast(&dst_buffer.begin()[0]);
             T* src_ptr = thrust::raw_pointer_cast(&(this->begin()[0]));
             comm_profiler.add_intergpu_comm_bytes(sizeof(T) * this->size(), this->cudaDeviceID(), dst_buffer.cudaDeviceID());
             CUDA_CHECK(cudaMemcpyPeerAsync((void*)dst_ptr, dst_buffer.cudaDeviceID(), (void*)src_ptr, this->cudaDeviceID(), sizeof(T) * this->size(), stream));
        }

    private:
        // stores which device data is on
        thrust::device_vector<T> data;
};

