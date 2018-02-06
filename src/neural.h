#pragma once
#include "base.h"

#define MAX_LAYERS 10

typedef struct NeuralNet
{
    f64* values;
    f64* weights;
} NeuralNet;

typedef struct NeuralNetDef
{
    i32 layerCount;
    i32 layerNeuronCount[MAX_LAYERS];
    i32 neuronCount;
    i32 neuralNetSize;
    i32 inputNeuronCount;
    i32 synapseTotalCount;
    f64 bias;
} NeuralNetDef;

void makeNeuralNetDef(NeuralNetDef* def, const i32 layerCount, const i32 layerNeuronCount[], f64 bias);
u8* neuralNetAlloc(NeuralNet** nn, const i32 nnCount, const NeuralNetDef* def);
void neuralNetInitRandom(NeuralNet** nn, const i32 nnCount, const NeuralNetDef* def);
void neuralNetPropagate(NeuralNet** nn, const i32 nnCount, const NeuralNetDef* def);
