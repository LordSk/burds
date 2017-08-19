#pragma once
#include "base.h"

typedef struct Neuron
{
   f64 value;
   f64 synapseWeight[10];
} Neuron;

typedef struct NeuralNet
{
    Neuron neurons[1];
} NeuralNet;

typedef struct NeuralNetDef
{
    i32 layerCount;
    i32 layerNeuronCount[10];
} NeuralNetDef;


u8* allocNeuralNets(NeuralNet** nn, const i32 nnCount, const NeuralNetDef* def);
void initNeuralNets(NeuralNet** nn, const i32 nnCount, const NeuralNetDef* def);
void propagateNeuralNets(NeuralNet** nn, const i32 nnCount, const NeuralNetDef* def);
