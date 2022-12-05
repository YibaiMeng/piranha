
#pragma once

#include "LayerConfig.h"
#include "../globals.h"

class ReLUConfig : public LayerConfig {
	public:
		size_t inputDim = 0;
		size_t batchSize = 0;
	    size_t microBatchSize = 0;

		ReLUConfig(size_t _inputDim, size_t _batchSize, size_t _microBatchSize=0) : LayerConfig("ReLU"),
			inputDim(_inputDim), batchSize(_batchSize), microBatchSize(_microBatchSize) {
			// nothing
		};
};

