
#pragma once

#include "ReLUConfig.h"
#include "Layer.h"
#include "../util/util.cuh"
#include "../util/connect.h"
#include "../globals.h"

extern int partyNum;

template<typename T, template<typename, typename...> typename Share>
class ReLULayer : public Layer<T, Share> {

    private:
        ReLUConfig conf;

        Share<uint8_t> _reluPrime;

        Share<T> _activations;
        Share<T> _deltas;

    public:
        //Constructor and initializer
        ReLULayer(ReLUConfig* conf, int _layerNum, int seed);

        //Functions
        void loadSnapshot(std::string path) override;
        void saveSnapshot(std::string path) override;
        void printLayer() override;
        // batch_offset means only the batch_offset_start th element in inputs will be consider.
        // batch_offset_end is exclusive.
        // if batch_offset is -1, then the whole share will be considered.
        void forward(const Share<T>& input, int micro_batch_idx=-1) override;
        void backward(const Share<T> &delta, const Share<T> &forwardInput, int micro_batch_idx=-1) override;

        //Getters
        Share<T> *getActivation() {return &_activations;};
        Share<T> *getWeights() {return nullptr;};
        Share<T> *getBiases() {return nullptr;};
        Share<T> *getDelta() {return &_deltas;};

        static Profiler relu_profiler;
};

