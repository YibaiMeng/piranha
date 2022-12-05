
#pragma once

#include "LayerConfig.h"
#include "../globals.h"

using namespace std;

class FCConfig : public LayerConfig
{
public:
	size_t inputDim = 0;
	size_t batchSize = 0;
	size_t microBatchSize = 0;
	size_t outputDim = 0;

	FCConfig(size_t _inputDim, size_t _batchSize, size_t _microBatchSize, size_t _outputDim)
	:inputDim(_inputDim),
	 batchSize(_batchSize), 
	 microBatchSize(_microBatchSize),
	 outputDim(_outputDim),
	 LayerConfig("FC")
	{};
};
