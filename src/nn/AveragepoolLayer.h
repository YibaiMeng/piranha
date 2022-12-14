
#pragma once

#include "nn/AveragepoolConfig.h"
#include "nn/Layer.h"
#include "util/util.cuh"
#include "util/connect.h"
#include "globals.h"

template<typename T, template<typename, typename...> typename Share>
class AveragepoolLayer : public Layer<T, Share> {

    private:
        AveragepoolConfig conf;

        Share<T> _activations;
        Share<T> _deltas;

    public:
        //Constructor and initializer
        AveragepoolLayer(AveragepoolConfig* conf, int _layerNum, int seed);

        //Functions
        void loadSnapshot(std::string path) override;
        void saveSnapshot(std::string path) override;
        void printLayer() override;
        void forward(const Share<T>& input, int micro_batch_idx=-1) override;
        void backward(const Share<T>& delta, const Share<T> &forwardInput, int micro_batch_idx=-1) override;

        //Getters
        Share<T> *getActivation() {return &_activations;};
        Share<T> *getWeights() {return nullptr;}
        Share<T> *getBiases() {return nullptr;}
        Share<T> *getDelta() {return &_deltas;};

        static Profiler averagepool_profiler;
};
