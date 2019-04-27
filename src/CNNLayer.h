
#pragma once
#include "CNNConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
using namespace std;


class CNNLayer : public Layer
{
private:
	CNNConfig conf;
	RSSVectorMyType activations;
	RSSVectorMyType deltas;
	RSSVectorMyType weights;
	RSSVectorMyType biases;

public:
	//Constructor and initializer
	CNNLayer(CNNConfig* conf, int _layerNum);
	void initialize();

	//Functions
	void printLayer() override;
	void forward(const RSSVectorMyType& inputActivation) override;
	void computeDelta(RSSVectorMyType& prevDelta) override;
	void updateEquations(const RSSVectorMyType& prevActivations) override;

	//Getters
	RSSVectorMyType* getActivation() {return &activations;};
	RSSVectorMyType* getDelta() {return &deltas;};
};