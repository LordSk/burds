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

i32 reinsertTruncate(i32 maxBest, i32 nnCount, f64* fitness, NeuralNet** nextGen,
                     NeuralNet** curGen, NeuralNetDef* def);
void crossover(i32 id, i32 parentA, i32 parentB, NeuralNet** nextGen,
               NeuralNet** curGen, NeuralNetDef* def);
i32 selectTournament(const i32 reinsertCount, const i32 tournamentSize, i32 notThisId, const f64* fitness);
i32 mutate(f32 rate, f32 factor, i32 nnCount, NeuralNet** nextGen, NeuralNetDef* def);
