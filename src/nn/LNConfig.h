
#pragma once

#include "nn/LayerConfig.h"
#include "globals.h"

using namespace std;

class LNConfig : public LayerConfig {

    public:
        size_t inputSize = 0;
        size_t numBatches = 0;
	    size_t microBatchSize = 0;

        LNConfig(size_t _inputSize, size_t _numBatches, size_t _microBatchSize=0)
        :inputSize(_inputSize),
         numBatches(_numBatches),
         microBatchSize(_microBatchSize),
         LayerConfig("LN")
        {};
};
