
#pragma once

#include "nn/CNNConfig.h"
#include "nn/Layer.h"
#include "util/util.cuh"
#include "util/connect.h"
#include "globals.h"

extern int partyNum;

template<typename T, template<typename, typename...> typename Share>
class CNNLayer : public Layer<T, Share> {

    private:
        CNNConfig conf;

        Share<T> weights;

        Share<T> _activations;
        Share<T> _deltas;

    public:
        //Constructor and initializer
        CNNLayer(CNNConfig *conf, int _layerNum, int seed);
        void initialize(int layerNum, int seed);

        //Functions
        void loadSnapshot(std::string path) override;
        void saveSnapshot(std::string path) override;
        void printLayer() override;
        void forward(const Share<T> &input, int micro_batch_idx=-1) override;
        void backward(const Share<T> &delta, const Share<T> &forwardInput, int micro_batch_idx=-1) override;

        //Getters
        Share<T> *getActivation() {return &_activations;};
        Share<T> *getWeights() {return &weights;};
        Share<T> *getBiases() {return nullptr;};
        Share<T> *getDelta() {return &_deltas;};
};

