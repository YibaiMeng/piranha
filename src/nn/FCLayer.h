
#pragma once

#include "FCConfig.h"
#include "Layer.h"
//#include "../mpc/RSS.h"
//#include "../mpc/TPC.h"
//#include "../mpc/FPC.h"
#include "../util/util.cuh"
#include "../util/connect.h"
#include "../globals.h"

extern int partyNum;

template<typename T, template<typename, typename...> typename Share>
class FCLayer : public Layer<T, Share> {

    private:
        FCConfig conf;

        Share<T> weights;
        Share<T> biases;

        Share<T> _activations;
        Share<T> _deltas;

    public:
        //Constructor and initializer
        FCLayer(FCConfig* conf, int _layerNum, int seed);
        void initialize(int layerNum, int seed);

        //Functions
        void loadSnapshot(std::string path) override;
        void saveSnapshot(std::string path) override;
        void printLayer() override;
        void forward(const Share<T> &input, int micro_batch_idx=-1) override;
        void backward(const Share<T> &delta, const Share<T> &forwardInput, int micro_batch_idx=-1) override;

        //Getters
        Share<T>* getActivation() {return &_activations;};
        Share<T>* getWeights() {return &weights;};
        Share<T>* getBiases() {return &biases;};
        Share<T>* getDelta() {return &_deltas;};
};
