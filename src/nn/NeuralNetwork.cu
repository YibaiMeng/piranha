
#pragma once

#include "../util/util.cuh"
#include "FCLayer.h"
#include "CNNLayer.h"
#include "MaxpoolLayer.h"
#include "AveragepoolLayer.h"
#include "ReLULayer.h"
#include "ResLayer.h"
#include "LNLayer.h"
#include "NeuralNetwork.h"
#include "../mpc/RSS.h"
#include "../mpc/TPC.h"
#include "../mpc/FPC.h"
#include "../util/functors.h"
#include <math.h>       /* log2 */
#include <sys/types.h>
#include <sys/stat.h>
#include <strstream>
#include <loguru.hpp>

extern size_t INPUT_SIZE;
extern size_t NUM_CLASSES;
extern bool WITH_NORMALIZATION;
extern bool LARGE_NETWORK;
extern size_t db_layer_max_bytes;

extern nlohmann::json piranha_config;

// input batch, labels
// get output of last layer, normalize, then subtract from labels for derivates
template<typename T, template<typename, typename...> typename Share>
NeuralNetwork<T, Share>::NeuralNetwork(NeuralNetConfig* config, int seed) : input(MINI_BATCH_SIZE * INPUT_SIZE), _pipeline_group_streams(PIPELINE_GROUPS) {
    LOG_S(INFO) << "Loading NeuralNetConfig";
    int cuda_device_count = -1;
    CUDA_CHECK(cudaGetDeviceCount(&cuda_device_count));
    if(cuda_device_count < 0) {
        LOG_S(FATAL) << "No CUDA Device available. Unable to initialize NeuralNetwork";
        exit(-1);
    }
    int prev_device_id = input.cudaDeviceID();
    int prev_start_idx = 0;
	for (int i = 0; i < config->layerConf.size(); i++) {
        int layer_device_id = config->layerCUDADevice[i];
        LOG_S(1) << "Layer " << config->layerConf[i]->type << " (" << i << ") is assigned to GPU " << layer_device_id;         
        if(layer_device_id >= cuda_device_count) {
            LOG_S(FATAL) << "CUDA Device " << layer_device_id << " assigned to layer " << i << " does not exist.";
        }
        CUDA_CHECK(cudaSetDevice(layer_device_id));        
		if (config->layerConf[i]->type.compare("FC") == 0) {
			layers.push_back(new FCLayer<T, Share>((FCConfig *) config->layerConf[i], i, seed+i));
        } else if (config->layerConf[i]->type.compare("CNN") == 0) {
			layers.push_back(new CNNLayer<T, Share>((CNNConfig *) config->layerConf[i], i, seed+i));
        } else if (config->layerConf[i]->type.compare("ReLU") == 0) {
			layers.push_back(new ReLULayer<T, Share>((ReLUConfig *) config->layerConf[i], i, seed+i));
        } else if (config->layerConf[i]->type.compare("Maxpool") == 0) {
			layers.push_back(new MaxpoolLayer<T, Share>((MaxpoolConfig *) config->layerConf[i], i, seed+i));
        } else if (config->layerConf[i]->type.compare("LN") == 0) {
		    layers.push_back(new LNLayer<T, Share>((LNConfig *) config->layerConf[i], i, seed+i));
        } else if (config->layerConf[i]->type.compare("Averagepool") == 0) {
		    layers.push_back(new AveragepoolLayer<T, Share>((AveragepoolConfig *) config->layerConf[i], i, seed+i));
        } else if (config->layerConf[i]->type.compare("Res") == 0) {
            layers.push_back(new ResLayer<T, Share>((ResLayerConfig *) config->layerConf[i], i, seed+i));
        } else {
            LOG_S(FATAL) << "Only FC, CNN, ReLU, Maxpool, Averagepool, ResLayer, and LN layer types currently supported.";
        }
        layers.back()->layerCUDADeviceId = layer_device_id;
        if(prev_device_id != layer_device_id) {
            _pipeline_groups.push_back({prev_start_idx, i - 1, prev_device_id});
            prev_start_idx = i;
            LOG_S(2) << "Layer " << config->layerConf[i]->type << " (" << i << ") 's device " << layer_device_id << " is different from the previous layer";
            size_t activation_size, delta_size;
            if(i >= 1) activation_size = layers[i-1]->getActivation()->size();
            else activation_size = input.size();
            LOG_S(2) << "Scratch space of activation output of layer " << i-1 << " is " << activation_size;
            _activation_cache.emplace(std::make_pair(i-1, layer_device_id), activation_size);
            //_activation_before_layer.emplace(i, activation_size);
            CUDA_CHECK(cudaSetDevice(prev_device_id));
            delta_size = layers[i]->getDelta()->size();
            _delta_cache.emplace(std::make_pair(i, prev_device_id), delta_size);
            prev_device_id = layer_device_id;
        }
	}
    std::stringstream activation_cache_status, delta_cache_status;
    int activation_cache_space = 0, delta_cache_space = 0;
    for(auto& p : _activation_cache) {
            activation_cache_status << "Layer " << p.first.first << ", GPU " << p.first.second << " ";
            activation_cache_space += p.second.size();
    }
    for(auto& p : _delta_cache) {
            delta_cache_status << "Layer " << p.first.first << ", GPU " << p.first.second << " ";
            delta_cache_space += p.second.size();
    }

    if(!activation_cache_status.str().empty()) {
        LOG_S(INFO) << "Cache for Activation: " <<  activation_cache_status.str() << "; Size: " << activation_cache_space;
    } else {
        LOG_S(INFO) << "No cache for Activation";
    }
    if(!delta_cache_status.str().empty()) {
        LOG_S(INFO) << "Cache for Delta: " << delta_cache_status.str() << "; Size: " << delta_cache_space; 
    } else {
        LOG_S(INFO) << "No cache for Delta";
    }

    _pipeline_groups.push_back({prev_start_idx, config->layerConf.size() - 1, prev_device_id});
    // TODO: redudant information specifying the size of the pipeline group.
    if(piranha_config["pipeline_parallel"]) {
        CHECK_F(_pipeline_groups.size() == PIPELINE_GROUPS);
        for(int idx = 0; idx < _pipeline_groups.size(); idx++) {
            CUDA_CHECK(cudaSetDevice(std::get<2>(_pipeline_groups.at(idx))));
            CUDA_CHECK(cudaStreamCreateWithFlags(&(_pipeline_group_streams.at(idx)), cudaStreamNonBlocking));
        }
    }
    for(auto&p : _pipeline_groups) {
        int device = std::get<2>(p);
        // Get the amount of memory available on the device
        size_t free_memory, total_memory;
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
        // Print the amount of memory available on the device
        LOG_S(INFO) << "Memory available: " << free_memory << " bytes out of " << total_memory << " on Device " << device;
    }
    LOG_S(INFO) << "NeuralNetConfig loading finished";
}

template<typename T, template<typename, typename...> typename Share>
NeuralNetwork<T, Share>::~NeuralNetwork()
{
	for (auto it = layers.begin(); it != layers.end(); ++it) {
		delete (*it);
    }
	layers.clear();
    // No iterating is needed as they are maps of Share<T>, the deconstructor would be called.
    _activation_cache.clear();
    _delta_cache.clear();
    for(int idx = 0; idx < _pipeline_group_streams.size(); idx++) {
        CUDA_CHECK(cudaStreamDestroy(_pipeline_group_streams.at(idx)));
    }
}


template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::printNetwork() {
    for (auto it = layers.begin(); it != layers.end(); it++) {
        (*it)->printLayer();
    }
}

template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::printPipelineGroup() {
   for(int i = 0; i < _pipeline_groups.size(); i++) {
         LOG_F(0, "Group %i on GPU %i: between layers %i and layers %i\n", i, std::get<2>(_pipeline_groups[i]), std::get<0>(_pipeline_groups[i]), std::get<1>(_pipeline_groups[i]));
   }
}


template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::loadSnapshot(std::string path) {

    for (auto it = layers.begin(); it != layers.end(); it++) {
        (*it)->loadSnapshot(path);
    }

    //std::string input_file = path + "/input";
    //loadShareFromFile(input_file, input);
}

template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::saveSnapshot(std::string path) {

    int status = mkdir(path.c_str(), S_IRWXU | S_IRWXG); 
    /*
    //printf("directory create status: %d\n", status);
    if (errno == EEXIST && partyNum != 0) {
        return;
    }

    if (errno != 0 && errno != EEXIST) {
        printf("directory create failed with status %d\n", errno);
        exit(errno);
    }
    // TODO make sure this works on localhost
    */

    for (auto it = layers.begin(); it != layers.end(); it++) {
        (*it)->saveSnapshot(path);
    }

    printf("snapshot saved to: %s\n", path.c_str());
}

template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::_forward_pipeline_group(int group_index) { 
    std::string thread_name = "Fp " + std::to_string(group_index);
    loguru::set_thread_name(thread_name.c_str());
    LOG_F(1, "Starting forward pipeline group, layers %i to %i on GPU %i", std::get<0>(_pipeline_groups.at(group_index)), std::get<1>(_pipeline_groups.at(group_index)), std::get<2>(_pipeline_groups.at(group_index)));
    CHECK_F(MINI_BATCH_SIZE % MICRO_BATCH_SIZE == 0, "MICRO_BATCH_SIZE must be divisible by MINI_BATCH_SIZE.");
    int iter_count = MINI_BATCH_SIZE / MICRO_BATCH_SIZE;
    int start = std::get<0>(_pipeline_groups.at(group_index));
    int end = std::get<1>(_pipeline_groups.at(group_index));
    int gpu_id = std::get<2>(_pipeline_groups.at(group_index));
    for(int iter = 0; iter < iter_count; iter++) {
        CUDA_CHECK(cudaSetDevice(gpu_id));
        for(int l = start; l <= end; l++) {
            LOG_F(1, "Layer %i", l);
            Share<T>* previous_activation;
            if(l > start) {
                CHECK_F(layers[l-1]->getActivation()->size() % MINI_BATCH_SIZE == 0, "The size of Share<T> layers[l-1]->getActivation() must be divisible by MINI_BATCH_SIZE");
                int size_per_microbatch = layers[l-1]->getActivation()->size() / MINI_BATCH_SIZE * MICRO_BATCH_SIZE;
                LOG_F(1, "Input activation to layer %i size is %i", l, size_per_microbatch);
                previous_activation = new Share<T>(*(layers[l-1]->getActivation()), iter * size_per_microbatch, (iter + 1) * size_per_microbatch);
            } else if(l == 0) {
                CHECK_F(input.size() % MINI_BATCH_SIZE == 0, "The size of Share<T> input must be divisible by MINI_BATCH_SIZE");
                int size_per_microbatch = input.size() / MINI_BATCH_SIZE * MICRO_BATCH_SIZE;
                LOG_F(1, "Input activation to layer %i size is %i", l, size_per_microbatch);
                previous_activation = new Share<T>(input, iter * size_per_microbatch, (iter + 1) * size_per_microbatch);
            } else if(l == start) {
                // Wait for the output of the previous microbatch to finish their transfer.
                // No need to wait for input layer, as it's input will be on the same device as the first group, 
                // and it will be ready before executing the first group. 
                CHECK_F(group_index > 0, "The first pipeline group must start at the first layer");
                LOG_S(1) << "The previous pipeline group is located on GPU " << layers[l-1]->cudaDeviceID();
                Share<T>& previous_activation_ref = _activation_cache.at(std::make_pair(l-1, layers[l]->cudaDeviceID()));
                LOG_S(1) << "Waiting for the transfer of the activation output from the previous pipeline group. ";
                CUDA_CHECK(cudaSetDevice(layers[l-1]->cudaDeviceID()));
                // Wait for the transfer from the previous group to complete.
                CUDA_CHECK(cudaStreamSynchronize(_pipeline_group_streams.at(group_index - 1)));
                CUDA_CHECK(cudaSetDevice(layers[l]->cudaDeviceID()));
                LOG_S(1) << "Transfer of the activation cache is completed.";
                int size_per_microbatch = layers[l-1]->getActivation()->size() / MINI_BATCH_SIZE * MICRO_BATCH_SIZE;
                LOG_F(1, "Input activation to layer %i size is %i", l, size_per_microbatch);
                previous_activation = new Share<T>(previous_activation_ref, iter * size_per_microbatch, (iter + 1) * size_per_microbatch);                    
            }
            CUDA_CHECK(cudaSetDevice(gpu_id));
            // Note we need to specify the microbatch index in our commands.
            layers[l]->forward(*previous_activation, iter);
            // Wait for the legacy stream to finish execution.
            CUDA_CHECK(cudaStreamSynchronize(0));
        }
        LOG_F(0, "Finished the %i/%i th microbatch.", iter, iter_count);
        // Now forward is complete, we need to transfer the activation layer to the GPU on the next device.
        // This is done asychronously.
        // TODO: need to call destructor on previous_activation and curr_activation, otherwise would result in memory leak.
        if(end + 1 < layers.size()) {
            CHECK_F(gpu_id != layers[end + 1]->cudaDeviceID());
            CUDA_CHECK(cudaSetDevice(gpu_id));
            int size_per_microbatch = layers[end]->getActivation()->size() / MINI_BATCH_SIZE * MICRO_BATCH_SIZE;
            Share<T>& curr_activation_ref = *layers[end]->getActivation();
            Share<T>* curr_activation = new Share<T>(curr_activation_ref, size_per_microbatch * iter, size_per_microbatch * (iter + 1));
            CUDA_CHECK(cudaSetDevice(layers[end+1]->cudaDeviceID()));
            Share<T>& next_activation_ref = _activation_cache.at(std::make_pair(end, layers[end+1]->cudaDeviceID()));
            Share<T>* next_activation = new Share<T>(next_activation_ref, size_per_microbatch * iter, size_per_microbatch * (iter + 1));
            CUDA_CHECK(cudaSetDevice(gpu_id));
            // So far we only allow one transfer to be in-flight at anytime.
            // TODO: allow multiple transfers to be inflight.
            LOG_F(0, "Waiting for the previous copy, if any, to finished.");
            CUDA_CHECK(cudaStreamSynchronize(_pipeline_group_streams.at(group_index)));
            LOG_F(0, "Previous copy finished, start copying the activation at layer %i to layer %i", end, end+1);
            curr_activation->copyAsync(*next_activation, _pipeline_group_streams.at(group_index));
        }
    }  
}

template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::forward_pipeline(std::vector<double> &data) {
    LOG_S(1) << "Finish executing pipelined NN.forward on input of size " << data.size() << " with " << _pipeline_groups.size() << " GPUs";
    input.zero();
    input.setPublic(data);
    std::vector<std::thread> _pipeline_threads;
    for(int group_idx = 0; group_idx < _pipeline_groups.size(); group_idx++) {
        _pipeline_threads.emplace_back([this](int group_idx) {this->_forward_pipeline_group(group_idx);}, group_idx);
    }
    LOG_F(2, "All %i pipeline threads started", _pipeline_threads.size()); 

    for(int group_idx = 0; group_idx < _pipeline_groups.size(); group_idx++) {
        _pipeline_threads.at(group_idx).join();
    }
    LOG_S(1) << "Finish executing NN.forward with pipelines";
}


template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::forward(std::vector<double> &data) {
    LOG_S(1) << "Executing NN.forward on input of size " << data.size();
    input.zero();
    input.setPublic(data);
    db_layer_max_bytes = 0;
	layers[0]->forward(input);
    CUDA_CHECK(cudaStreamSynchronize(0));
    printMemUsage();
	for (size_t i = 1; i < layers.size(); ++i) {
        db_layer_max_bytes = 0;
	    Share<T>* previous_activation = layers[i-1]->getActivation();
        if(previous_activation->cudaDeviceID() != layers[i]->cudaDeviceID()) {
            LOG_S(1) << "The activations of layer " << i - 1 << " is located on GPU " << layers[i-1]->cudaDeviceID() << ", while the activations of layer " << i 
            << " is located on GPU " << layers[i]->cudaDeviceID();
            LOG_S(1) << "Using the activation copy stored in the cache.";
            previous_activation = &(_activation_cache.at({i-1, layers[i]->cudaDeviceID()}));            
        }
        CUDA_CHECK(cudaSetDevice(layers[i]->cudaDeviceID()));
        layers[i]->forward(*previous_activation); 
        CUDA_CHECK(cudaStreamSynchronize(0));
        printMemUsage();
        // Copy activation to next layer, if needed.
        if(i + 1 < layers.size() and layers[i+1]->cudaDeviceID() != layers[i]->cudaDeviceID()) {
            LOG_S(1) << "The ID of the next layer is located on GPU " << layers[i+1]->cudaDeviceID() << ", copying to cache.";
            Share<T>& next_activation = _activation_cache.at(std::make_pair(i, layers[i+1]->cudaDeviceID()));
            memory_profiler.add_intergpu_comm_bytes(next_activation.size() * sizeof(T), layers[i]->cudaDeviceID(),  layers[i+1]->cudaDeviceID()); 
            CUDA_CHECK(cudaSetDevice(layers[i]->cudaDeviceID()));
            layers[i]->getActivation()->copySync(next_activation);
        }
	}

    if (piranha_config["print_activations"]) {
        printShareFinite(*(layers[layers.size()-1]->getActivation()), "output activation", 10);
    }

    LOG_S(1) << "Finish executing NN.forward";
}

template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::backward(std::vector<double> &labels) {
	LOG_S(1) << "Executing backward on labels of " << labels.size();
    // Label share same as the share on the last device.
    CUDA_CHECK(cudaSetDevice(layers[layers.size() - 1]->cudaDeviceID()));

    Share<T> labelShare(labels.size());
    labelShare.setPublic(labels);

    Share<T> deltas(labels.size());
    _backward_delta(labelShare, deltas);

    if (piranha_config["print_deltas"]) {
        printShareFinite(deltas, "input delta to bw pass", 10);
    }
    if(piranha_config["pipeline_parallel"]) {
        _backward_pass_pipeline(deltas);
    } else {
        _backward_pass(deltas);
    }
}

template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::_relu_grad(Share<T> &labels, Share<T> &deltas) {

    deltas += *(layers[layers.size() - 1]->getActivation());
    deltas -= labels;
    if (LOG_MINI_BATCH > 1) {
        dividePublic(deltas, (T)1 << LOG_MINI_BATCH);
    }

    //printShareTensor(deltas, "deltas (non-normalized)", 128, 1, 1, 10);
    //exit(1);
}

template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::_relu_norm_grad(Share<T> &labels, Share<T> &deltas) {

    int nClasses = labels.size() / MINI_BATCH_SIZE;

    Share<T> mu(MINI_BATCH_SIZE);
    for(int share = 0; share < Share<T>::numShares(); share++) {
        gpu::reduceSum(
            layers[layers.size() - 1]->getActivation()->getShare(share),
            mu.getShare(share), 
            false, MINI_BATCH_SIZE, nClasses
        );
    }

    //printShare(mu, "mu");

    Share<T> inversedMu(MINI_BATCH_SIZE);
    inverse(mu, inversedMu);

    //printShare(inversedMu, "inverse mu");

    for(int share = 0; share < Share<T>::numShares(); share++) {
        gpu::vectorExpand(inversedMu.getShare(share), deltas.getShare(share), nClasses);
    }

    deltas *= *(layers[layers.size() - 1]->getActivation());
    dividePublic(deltas, (T)1 << FLOAT_PRECISION);

    //printShareTensor(deltas, "after divide", 128, 1, 1, 10);

    deltas -= labels;

    //printShareTensor(deltas, "minus labels", 128, 1, 1, 10);
    if (LOG_MINI_BATCH > 1) {
        dividePublic(deltas, (T)1 << LOG_MINI_BATCH);
    }

    /*
    printShareTensor(deltas, "deltas (normalized)", 1, 1, 128, 10);
    exit(1);
    */
}

template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::_reveal_softmax_grad(Share<T> &labels, Share<T> &deltas) {
    int last_layer_device_id = layers[layers.size() - 1]->cudaDeviceID();
    if(labels.cudaDeviceID() != last_layer_device_id or deltas.cudaDeviceID() != last_layer_device_id) {
        LOG_S(FATAL) << "labels and deltas should be on the same device as the activations of thefinal layer";
    }
    CUDA_CHECK(cudaSetDevice(last_layer_device_id));
    Share<T> x(deltas.size());
    x += *layers[layers.size() - 1]->getActivation();

    //printShareFinite(*layers[layers.size() - 1]->getActivation(), "activations", 10);

    //printShareTensor(x, "logits", 1, 1, 1, 10);

    DeviceData<T> revealedX(x.size());
    reconstruct(x, revealedX);

    thrust::device_vector<double> floatActivations(revealedX.size());
    thrust::transform(revealedX.begin(), revealedX.end(),
            floatActivations.begin(), to_double_functor<T>());

    /*
    printf("input to softmax:\n");
    for (int i = 0; i < floatActivations.size(); i++) {
        double act = floatActivations[i];
        printf("%f ", act);
    }
    printf("\n");
    */

    int nClasses = labels.size() / MINI_BATCH_SIZE;
    thrust::device_vector<double> sums(floatActivations.size() / nClasses, 0);
    for (int i = 0; i < floatActivations.size(); i++) {
        floatActivations[i] = exp(floatActivations[i]); 
        //floatActivations[i] *= floatActivations[i]; 

        sums[i / nClasses] += floatActivations[i];
    }

    for (int i = 0; i < floatActivations.size(); i++) {
        floatActivations[i] /= sums[i / nClasses];
    }

    /*
    printf("after softmax:\n");
    for (int i = 0; i < floatActivations.size(); i++) {
        double act = floatActivations[i];
        printf("%f ", act);
    }
    printf("\n");
    */

    DeviceData<T> softmax_values(floatActivations.size());
    thrust::transform(floatActivations.begin(), floatActivations.end(), softmax_values.begin(),
        to_fixed_functor<T>());

    deltas.zero();
    deltas += softmax_values;
    deltas -= labels;
    if (LOG_MINI_BATCH > 1) {
        dividePublic(deltas, (T)1 << LOG_MINI_BATCH);
    }

    //printShareFinite(deltas, "softmax delta", 10);
}

template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::_softmax_grad(Share<T> &labels, Share<T> &deltas) {

    Share<T> x(deltas.size());
    x += *layers[layers.size() - 1]->getActivation();


    size_t n = 3;
    dividePublic(x, (T)1 << n);

    x += 1 << FLOAT_PRECISION;
    
    for (int i = 0; i < n - 1; i++) {
        x *= x;
        dividePublic(x, (T)1 << FLOAT_PRECISION);
    } 

    int nClasses = labels.size() / MINI_BATCH_SIZE;

    Share<T> sums(MINI_BATCH_SIZE);
    for(int share = 0; share < Share<T>::numShares(); share++) {
        gpu::reduceSum(
            x.getShare(share),
            sums.getShare(share), 
            false, MINI_BATCH_SIZE, nClasses
        );
    }

    Share<T> inversedSums(MINI_BATCH_SIZE);

#if 1
    DeviceData<T> revealedSums(sums.size());
    reconstruct(sums, revealedSums);

    DeviceData<T> invertedRevealedSums(sums.size());

    thrust::transform(revealedSums.begin(), revealedSums.end(), invertedRevealedSums.begin(),
        inv_fixed_point_functor<T>());

    inversedSums += invertedRevealedSums;
#else
    inverse(sums, inversedSums);
#endif

    for(int share = 0; share < Share<T>::numShares(); share++) {
        gpu::vectorExpand(inversedSums.getShare(share), deltas.getShare(share), nClasses);
    }

    deltas *= x;
    dividePublic(deltas, (T)1 << FLOAT_PRECISION);

    deltas -= labels;

    if (LOG_MINI_BATCH > 1) {
        dividePublic(deltas, (T)1 << LOG_MINI_BATCH);
    }
}

template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::_adhoc_softmax_grad(Share<T> &labels, Share<T> &deltas) {

    Share<T> maxVals(MINI_BATCH_SIZE);

    int numClasses = deltas.size() / MINI_BATCH_SIZE;
    int paddedSize = pow(ceil(log2(numClasses)), 2);

    Share<T> pools(MINI_BATCH_SIZE * paddedSize);
    for(int i = 0; i < Share<T>::numShares(); i++) {

        // TODO fix 4PC
        T pad_value = (T)(-10 * (1 << FLOAT_PRECISION));

        if(Share<T>::numShares() == 3) {
            switch(partyNum) {
                case 0:
                    pad_value = 0;
                    break;
                case 1:
                    if(i != 2) pad_value = 0;
                    break;
                case 2:
                    if(i != 1) pad_value = 0; 
                    break;
                case 3:
                    if(i != 0) pad_value = 0;
                    break;
            }
        }


        gpu::stride_pad(
            layers[layers.size() - 1]->getActivation()->getShare(i),
            pools.getShare(i),
            numClasses,
            paddedSize - numClasses,
            pad_value
        );
    }

    Share<uint8_t> expandedPrime(pools.size());
    maxpool(pools, maxVals, expandedPrime, paddedSize);

    //printShareFinite(maxVals, "max val", 1);

    Share<T> expandedMaxVals(deltas.size());
    for(int i = 0; i < Share<T>::numShares(); i++) {
        gpu::vectorExpand(maxVals.getShare(i), expandedMaxVals.getShare(i), numClasses);
    }

    //printShareFinite(expandedMaxVals, "expanded max val", 10);

    Share<T> diff(deltas.size());
    diff += *layers[layers.size() - 1]->getActivation();
    diff -= expandedMaxVals;
    //printShareFinite(diff, "diff", 10);

    diff += (1 << (FLOAT_PRECISION + 1));
    //printShareFinite(diff, "diff + 2", 10);

    Share<T> zeros(deltas.size());
    Share<uint8_t> b(deltas.size());
    ReLU(diff, zeros, b);
    zeros.zero();
    zeros += (T)(0.001 * (1 << FLOAT_PRECISION));

    dividePublic(diff, (T)2);

    Share<T> exponentialApprox(deltas.size());
    selectShare(zeros, diff, b, exponentialApprox);

    Share<T> sums(MINI_BATCH_SIZE);
    for(int share = 0; share < Share<T>::numShares(); share++) {
        gpu::reduceSum(
            exponentialApprox.getShare(share),
            sums.getShare(share),
            false, MINI_BATCH_SIZE, numClasses
        );
    }

    DeviceData<T> revealedSums(sums.size());
    reconstruct(sums, revealedSums);

    thrust::transform(revealedSums.begin(), revealedSums.end(), revealedSums.begin(), inv_fixed_point_functor<T>());

    Share<T> inversedSums(MINI_BATCH_SIZE);
    inversedSums += revealedSums;
    //inverse(sums, inversedSums);

    for(int share = 0; share < Share<T>::numShares(); share++) {
        gpu::vectorExpand(inversedSums.getShare(share), deltas.getShare(share), numClasses);
    }

    deltas *= exponentialApprox;
    dividePublic(deltas, (T)1 << FLOAT_PRECISION);

    deltas -= labels;

    if (LOG_MINI_BATCH > 1) {
        dividePublic(deltas, (T)1 << LOG_MINI_BATCH);
    }

    //printShareFinite(deltas, "approx deltas", 10);
    //printf("\n");
}

template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::_backward_delta(Share<T> &labels, Share<T> &deltas) {

    //_relu_grad(labels, deltas);
    //_relu_norm_grad(labels, deltas);
    //_softmax_grad(labels, deltas);
    _reveal_softmax_grad(labels, deltas);
    //deltas.zero();
    //_adhoc_softmax_grad(labels, deltas);
}

template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::_backward_pass(Share<T> &deltas) {

    // backwards pass
    CUDA_CHECK(cudaSetDevice(layers[layers.size() - 1]->cudaDeviceID()));
    int prev_layer_cuda_id = layers[layers.size() - 2]->cudaDeviceID();
    
    if(deltas.cudaDeviceID() != layers[layers.size() - 1]->cudaDeviceID()) {
        LOG_S(FATAL) << "The first delta must be on the same device as the last layer";
    }
    Share<T>* first_activation_input = layers[layers.size() - 2]->getActivation();
    if(first_activation_input->cudaDeviceID() != layers[layers.size() - 1]->cudaDeviceID()) {
        first_activation_input = &(_activation_cache.at({layers.size() - 2, layers[layers.size() - 1]->cudaDeviceID()}));
    }
    layers[layers.size() - 1]->backward(
        deltas,
        *first_activation_input
    );

	for (size_t i = layers.size() - 2; i > 0; i--) {
        Share<T>* delta_input = layers[i+1]->getDelta();
        Share<T>* activation_input = layers[i-1]->getActivation();
        if(delta_input->cudaDeviceID() != layers[i]->cudaDeviceID()) {
            delta_input = &(_delta_cache.at({i+1, layers[i]->cudaDeviceID()}));
        }
        if(activation_input->cudaDeviceID() != layers[i]->cudaDeviceID()) {
            activation_input = &(_activation_cache.at({i-1, layers[i]->cudaDeviceID()}));
        }
        CUDA_CHECK(cudaSetDevice(layers[i]->cudaDeviceID()));
        CHECK_F(layers[i]->cudaDeviceID() == delta_input->cudaDeviceID() && layers[i]->cudaDeviceID() == activation_input->cudaDeviceID(), 
        "All inputs of backward() must be on the same device. Delta is on %i, activation is on %i, layer is on %i.", delta_input->cudaDeviceID(), activation_input->cudaDeviceID(), layers[i]->cudaDeviceID());
	    layers[i]->backward(
            *delta_input,
            *activation_input
        );
        // Copy delta to cache at previous layer, should it be needed.
        if(layers[i]->cudaDeviceID() != layers[i-1]->cudaDeviceID()) {
            CUDA_CHECK(cudaSetDevice(layers[i]->cudaDeviceID()));
            layers[i]->getDelta()->copySync(_delta_cache.at({i, layers[i-1]->cudaDeviceID()}));      
        }
	}

    if (layers.size() > 1) {
        Share<T>* delta_input = layers[1]->getDelta();
        if(delta_input->cudaDeviceID() != layers[0]->cudaDeviceID()) {
            delta_input = &(_delta_cache.at({1, layers[0]->cudaDeviceID()}));
        }
        if(input.cudaDeviceID() != layers[0]->cudaDeviceID()) LOG_S(FATAL) << "Input must be on the same device as layer 0";
        CUDA_CHECK(cudaSetDevice(layers[0]->cudaDeviceID()));
        layers[0]->backward(
            *delta_input,
            input 
        );
    }
}


template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::_backward_pass_pipeline_group(int group_index, Share<T> &deltas) { 
    std::string thread_name = "Bp " + std::to_string(group_index);
    loguru::set_thread_name(thread_name.c_str());
    LOG_F(1, "Starting backward pipeline group, layers %i to %i on GPU %i", std::get<0>(_pipeline_groups.at(group_index)), std::get<1>(_pipeline_groups.at(group_index)), std::get<2>(_pipeline_groups.at(group_index)));
    CHECK_F(MINI_BATCH_SIZE % MICRO_BATCH_SIZE == 0, "MICRO_BATCH_SIZE must be divisible by MINI_BATCH_SIZE.");
    int iter_count = MINI_BATCH_SIZE / MICRO_BATCH_SIZE;
    int start = std::get<0>(_pipeline_groups.at(group_index));
    int end = std::get<1>(_pipeline_groups.at(group_index));
    int gpu_id = std::get<2>(_pipeline_groups.at(group_index));
    CUDA_CHECK(cudaSetDevice(gpu_id));
    for(int iter = 0; iter < iter_count; iter++) {
        CUDA_CHECK(cudaSetDevice(gpu_id));
        // Go backward layer by layer.
        for(int l = end; l >= start; l--) {
            Share<T>* delta_input;
            Share<T>* activation_input;
            if(l == end && l < layers.size() - 1) {
                CHECK_F(layers[l+1]->getDelta()->size() % MINI_BATCH_SIZE == 0, "The size of Share<T> input must be divisible by MINI_BATCH_SIZE");
                int size_per_microbatch = layers[l+1]->getDelta()->size() / MINI_BATCH_SIZE * MICRO_BATCH_SIZE;
                LOG_F(1, "Delta at layer %i's size is %i", l+1, size_per_microbatch);
                CUDA_CHECK(cudaSetDevice(layers[l+1]->cudaDeviceID()));
                // Wait for layer l+1 to finishing copying to l.
                LOG_S(2) << "Waiting for layer " << l+1 << " to finish copying delta to layer " << l;
                CUDA_CHECK(cudaStreamSynchronize(_pipeline_group_streams.at(group_index + 1)));
                LOG_S(2) << "Layer " << l+1 << "  finished copying delta to layer " << l;
                CUDA_CHECK(cudaSetDevice(layers[l]->cudaDeviceID()));
                Share<T>& delta_input_ref = _delta_cache.at({l+1, layers[l]->cudaDeviceID()});
                delta_input = new Share<T>(delta_input_ref, iter * size_per_microbatch, (iter + 1) * size_per_microbatch);
            } else if(l == end && l == layers.size() - 1) {
                // for the last layer in the network, the input is Share<T> &deltas.
                int size_per_microbatch = deltas.size() / MINI_BATCH_SIZE * MICRO_BATCH_SIZE;
                CUDA_CHECK(cudaSetDevice(layers[l]->cudaDeviceID()));
                delta_input = new Share<T>(deltas, iter * size_per_microbatch, (iter + 1) * size_per_microbatch);
            } else {
                int size_per_microbatch = layers[l+1]->getDelta()->size() / MINI_BATCH_SIZE * MICRO_BATCH_SIZE;
                CUDA_CHECK(cudaSetDevice(layers[l+1]->cudaDeviceID()));
                CHECK_F(layers[l+1]->cudaDeviceID() == layers[l]->cudaDeviceID());
                delta_input = new Share<T>(*(layers[l+1]->getDelta()), iter * size_per_microbatch, (iter + 1) * size_per_microbatch);
            }

            if(l == start && l > 0) {
                CHECK_F(layers[l-1]->getActivation()->size() % MINI_BATCH_SIZE == 0, "The size of Share<T> input must be divisible by MINI_BATCH_SIZE");
                int size_per_microbatch = layers[l-1]->getActivation()->size() / MINI_BATCH_SIZE * MICRO_BATCH_SIZE;
                LOG_F(1, "Activation at layer %i's size is %i", l-1, size_per_microbatch);
                // No need to wait, as activation input is already calculated during the forward pass.
                Share<T>& activation_input_ref = _activation_cache.at({l-1, layers[l]->cudaDeviceID()});
                CUDA_CHECK(cudaSetDevice(layers[l]->cudaDeviceID()));
                activation_input = new Share<T>(activation_input_ref, iter * size_per_microbatch, (iter + 1) * size_per_microbatch);
            } else if(l == start && l == 0) {
                CUDA_CHECK(cudaSetDevice(layers[l]->cudaDeviceID()));
                CHECK_F(input.cudaDeviceID() == layers[0]->cudaDeviceID(), "Input must be on the same device as the first layer");
                // No need to wait, as input is always ready.
                int size_per_microbatch = input.size() / MINI_BATCH_SIZE * MICRO_BATCH_SIZE;
                activation_input = new Share<T>(input, iter * size_per_microbatch, (iter + 1) * size_per_microbatch);
            } else {
                int size_per_microbatch = layers[l-1]->getActivation()->size() / MINI_BATCH_SIZE * MICRO_BATCH_SIZE;
                CUDA_CHECK(cudaSetDevice(layers[l-1]->cudaDeviceID()));
                CHECK_F(layers[l-1]->cudaDeviceID() == layers[l]->cudaDeviceID());
                activation_input = new Share<T>(*(layers[l-1]->getActivation()), iter * size_per_microbatch, (iter + 1) * size_per_microbatch);
            }

            CUDA_CHECK(cudaSetDevice(layers[l]->cudaDeviceID()));
            CHECK_F(layers[l]->cudaDeviceID() == delta_input->cudaDeviceID() && layers[l]->cudaDeviceID() == activation_input->cudaDeviceID(), 
            "All inputs of backward() must be on the same device. Delta is on %i, activation is on %i, layer is on %i.", delta_input->cudaDeviceID(), activation_input->cudaDeviceID(), layers[l]->cudaDeviceID());
            layers[l]->backward(
                *delta_input,
                *activation_input,
                iter
            );
            CUDA_CHECK(cudaSetDevice(layers[l]->cudaDeviceID()));
            // TODO: is this necessary?
            CUDA_CHECK(cudaStreamSynchronize(0));
        }
        LOG_F(0, "Finished the %i/%i th microbatch.", iter, iter_count);
        // Copy delta to cache at the GPU of the previous layer.
        if(start > 0) {
            CHECK_F(layers[start]->cudaDeviceID() != layers[start-1]->cudaDeviceID());
            int size_per_microbatch = layers[start]->getDelta()->size() / MINI_BATCH_SIZE * MICRO_BATCH_SIZE;
            CUDA_CHECK(cudaSetDevice(layers[start]->cudaDeviceID()));
            Share<T>& curr_delta_ref = *layers[start]->getDelta();
            Share<T>* curr_delta = new Share<T>(curr_delta_ref, size_per_microbatch * iter, size_per_microbatch * (iter + 1));
            CUDA_CHECK(cudaSetDevice(layers[start-1]->cudaDeviceID()));
            Share<T>& next_delta_ref = _delta_cache.at({start, layers[start-1]->cudaDeviceID()});   
            Share<T>* next_delta = new Share<T>(next_delta_ref, size_per_microbatch * iter, size_per_microbatch * (iter + 1));
            CUDA_CHECK(cudaSetDevice(layers[start]->cudaDeviceID()));
            // Same as in forward pipeline, we only allow one transfer operation to be inflight at a time.
            LOG_F(0, "Waiting for the previous copy, if any, to finished.");
            CUDA_CHECK(cudaStreamSynchronize(_pipeline_group_streams.at(group_index)));
            LOG_F(0, "Previous copy finished, start copying the activation at layer %i to layer %i", start, start-1);
            curr_delta->copyAsync(*next_delta, _pipeline_group_streams.at(group_index));  
        }
    }
}



template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::_backward_pass_pipeline(Share<T> &deltas) {
    CHECK_F(layers.size() > 1, "Can't use pipelining there is only one layer");
    // backwards pass
    CUDA_CHECK(cudaSetDevice(layers[layers.size() - 1]->cudaDeviceID()));    
    if(deltas.cudaDeviceID() != layers[layers.size() - 1]->cudaDeviceID()) {
        LOG_S(FATAL) << "The last delta must be on the same device as the last layer";
    }
    std::vector<std::thread> _pipeline_threads;
    for(int group_idx = 0; group_idx < _pipeline_groups.size(); group_idx++) {
        _pipeline_threads.emplace_back([this](int group_idx, Share<T>& deltas) {this->_backward_pass_pipeline_group(group_idx, deltas);}, group_idx, std::ref(deltas));
    }

    for(int group_idx = 0; group_idx < _pipeline_groups.size(); group_idx++) {
        _pipeline_threads.at(group_idx).join();
    }

    LOG_S(1) << "Finished NN.backward_pass";
}


/*
template<typename T, template<typename, typename...> typename Share>
void NeuralNetwork<T, Share>::printLoss(std::vector<double> &labels, bool cross_entropy) {
    
	layers[i]->forward(*(layers[i-1]->getActivation()));
    DeviceData<T> reconstr  uctedOutput(
    reconstruct(outputData, reconstructedOutput);
    std::vector<double> host_output(reconstructedOutput.size());
    copyToHost(reconstructedOutput, host_output, true);

    std::vector<double> host_expected(expectedOutput.size());
    copyToHost(expectedOutput, host_expected, true);

    double cumulative_error = 0.0;
    for(int i = 0; i < host_output.size(); i++) {
        if (cross_entropy) {
            if (host_expected[i] == 1) {
                cumulative_error -= log2 (host_output[i]);
            }
        } else {
            cumulative_error += fabs(host_output[i] - host_expected[i]);
        }
    }

    if (cross_entropy) {
    	std::cout << "cross entropy loss from expected FW pass results: " << cumulative_error << std::endl;
    } else {
    	std::cout << "cumulative error from expected FW pass results: " << cumulative_error << std::endl;
    }
    
    std::cout << "expected (first 10): ";
    for (int i = 0; i < 10; i++) std::cout << host_expected[i] << " ";
    std::cout << std::endl;

    std::cout << "actual (first 10): ";
    for (int i = 0; i < 10; i++) std::cout << host_output[i] << " ";
    std::cout << std::endl;
}
*/

template class NeuralNetwork<uint32_t, RSS>;
template class NeuralNetwork<uint64_t, RSS>;

template class NeuralNetwork<uint32_t, TPC>;
template class NeuralNetwork<uint64_t, TPC>;

template class NeuralNetwork<uint32_t, FPC>;
template class NeuralNetwork<uint64_t, FPC>;

template class NeuralNetwork<uint32_t, OPC>;
template class NeuralNetwork<uint64_t, OPC>;

