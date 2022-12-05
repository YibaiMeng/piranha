
#pragma once
#include <map>
#include <tuple>
#include <condition_variable>
#include <thread>

#include "NeuralNetConfig.h"
#include "Layer.h"
#include "../globals.h"

template<typename T, template<typename, typename...> typename Share>
class NeuralNetwork {

    public:

        Share<T> input;
        vector<Layer<T, Share> *> layers;
        // pipeline_layers[i][j] = means the first layer of micro batch  
        vector<vector<Layer<T, Share> *>> pipeline_layers;
        NeuralNetwork(NeuralNetConfig* config, int seed);
        ~NeuralNetwork();
    
        void printNetwork();
        void printPipelineGroup();
        void loadSnapshot(std::string path);
        void saveSnapshot(std::string path);

        void forward(std::vector<double> &data);
        void backward(std::vector<double> &labels);
        void forward_pipeline(std::vector<double> &data);


        // Compute the derivative of the softmax of (label - predication) relative to the predication. 
        // Stores the results in deltas.
        void _backward_delta(Share<T> &labels, Share<T> &deltas);
        // starting from the last layer and the deltas for that layer, go back layer by layer.
        void _backward_pass(Share<T> &deltas);
        void _backward_pass_pipeline(Share<T> &deltas);

//    private:
        // _activation_cache.at({layer_id, cuda_id}) represents the cache for the activation of layer_id on GPU cuda_id.
        // May not exist, may not be initalized, and the values may not be consistant.
        std::map<std::pair<int, int>, Share<T>> _activation_cache;
        // _delta_after_layer stores a copy of the delta results *after* this layer if they are on a different device.
        // May not be initalized, and the values may not be consistant.
        std::map<std::pair<int, int>, Share<T>> _delta_cache;
        // (start_layer_index, end_layer_idx, cuda_device_id of group)
        // the input Share is always in the first group.
        std::vector<std::tuple<int, int, int>> _pipeline_groups;
        void _relu_grad(Share<T> &labels, Share<T> &deltas);
        void _relu_norm_grad(Share<T> &labels, Share<T> &deltas);
        void _softmax_grad(Share<T> &labels, Share<T> &deltas);
        void _reveal_softmax_grad(Share<T> &labels, Share<T> &deltas);
        void _adhoc_softmax_grad(Share<T> &labels, Share<T> &deltas);

     private:
        // Each pipeline group executes the follow functions in a separate thread.
        void _forward_pipeline_group(int pipeline_group);
        void _backward_pass_pipeline_group(int pipeline_group, Share<T> &deltas);

        // Each stores a stream for a pipeline group.
        std::vector<cudaStream_t> _pipeline_group_streams;        
};

