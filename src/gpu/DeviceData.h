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

extern Profiler comm_profiler;
extern Profiler memory_profiler;

template<typename T>
using BufferIterator = thrust::detail::normal_iterator<thrust::device_ptr<T> >;

template<typename T, typename Iterator>
class DeviceDataBase {

    protected:

        DeviceDataBase() : transmitting(false), hostBuffer(0) {
            CUDA_CHECK(cudaGetDevice(&(this->cuda_device_id)));
        }

        DeviceDataBase(Iterator _first, Iterator _last) :
            first(_first), last(_last), transmitting(false), hostBuffer(0) {
            CUDA_CHECK(cudaGetDevice(&(this->cuda_device_id)));
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

        void transmit(size_t party) {
            CUDA_CHECK(cudaSetDevice(this->cuda_device_id));
            if (rtxThread.joinable()) {
                throw std::runtime_error("DeviceBuffer tx failed: already transmitting or receiving");
            }

            comm_profiler.add_comm_bytes(size() * sizeof(T), true);

            // copy to host
            //printf("copying %d bytes from device to host (end - begin is %d)\n", size(), end() - begin());
            hostBuffer.resize(size());
            thrust::copy(begin(), end(), hostBuffer.begin());

            // transmit
            transmitting = true;
            rtxThread = std::thread(sendVector<T>, party, std::ref(hostBuffer));
        }

        void receive(size_t party) {
            CUDA_CHECK(cudaSetDevice(this->cuda_device_id));
            if (rtxThread.joinable()) {
                throw std::runtime_error("DeviceBuffer rx failed: already transmitting or receiving");
            }

            comm_profiler.add_comm_bytes(size() * sizeof(T), false);

            hostBuffer.resize(size());

            transmitting = false;
            //receiveVector<T>(party, hostBuffer);
            rtxThread = std::thread(receiveVector<T>, party, std::ref(hostBuffer));
        }

        void join() {
            CUDA_CHECK(cudaSetDevice(this->cuda_device_id));
            if (!rtxThread.joinable()) return;
            
            rtxThread.join();
            if (!transmitting) {
                thrust::copy(hostBuffer.begin(), hostBuffer.end(), begin());
            }
            std::vector<T>().swap(hostBuffer); // clear buffer
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

    private:
        // stores which device data is on
        thrust::device_vector<T> data;
};

