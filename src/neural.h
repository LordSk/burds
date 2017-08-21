#pragma once
#include "base.h"

#define MAX_SYNAPSE_COUNT 15

typedef struct Neuron
{
   f64 value;
   f64 synapseWeight[MAX_SYNAPSE_COUNT];
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
