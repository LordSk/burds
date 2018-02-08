#include "neural.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include <stddef.h>

u8* neuralNetAlloc(NeuralNet** nn, const i32 nnCount, const NeuralNetDef* def)
{
    i32 dataSize = nnCount * def->neuralNetSize;
    u8* data = (u8*)_aligned_malloc(dataSize, 8);

    for(i32 i = 0; i < nnCount; ++i) {
        nn[i] = (NeuralNet*)(data + def->neuralNetSize * i);
        nn[i]->values = (f64*)(nn[i] + 1);
        nn[i]->weights = nn[i]->values + def->neuronCount;
    }

    LOG("allocated %d neural nets (layers=%d totalDataSize=%d)", nnCount, def->layerCount, dataSize);
    return data;
}

// set random synapse weight
void neuralNetInitRandom(NeuralNet** nn, const i32 nnCount, const NeuralNetDef* def)
{
    for(i32 i = 0; i < nnCount; ++i) {
        for(i32 s = 0; s < def->synapseTotalCount; ++s) {
            nn[i]->weights[s] = randf64(-1.0, 1.0);
        }
    }
}

void neuralNetPropagate(NeuralNet** nn, const i32 nnCount, const NeuralNetDef* def)
{
    const f64 bias = def->bias;

    // there probably is a way better way to traverse the net and compute values
    for(i32 i = 0; i < nnCount; ++i) {
        f64* neuronPrevValues = nn[i]->values;
        f64* neuronCurValues = nn[i]->values;
        f64* neuronWeights = nn[i]->weights;

        for(i32 l = 1; l < def->layerCount; ++l) {
            const i32 prevLayerNeuronCount = def->layerNeuronCount[l-1];
            neuronCurValues = neuronPrevValues + prevLayerNeuronCount;

            for(i32 n = 0; n < def->layerNeuronCount[l]; ++n) {
                // weighted sum
                i32 s = 0;
                f64 value = 0.0;
                for(; s < prevLayerNeuronCount; ++s) {
                    value += neuronWeights[s] * neuronPrevValues[s];
                }
                value += neuronWeights[s] * bias; // bias
                neuronWeights += prevLayerNeuronCount + 1; // +bias

                 // "activate" value
                //nn[i]->values[neuronCurLayerOff + n] = tanh(value);
                neuronCurValues[n] = 1.0 / (1.0 + exp(-value)); // sigmoid [0.05-0.995]
                //nn[i]->values[neuronCurLayerOff + n] = 1.0 / (1.0 + fabs(value)); // fast sigmoid [0.2-1.0]
            }

            neuronPrevValues += prevLayerNeuronCount;
        }
    }
}

void makeNeuralNetDef(NeuralNetDef* def, const i32 layerCount, const i32 layerNeuronCount[], f64 bias)
{
    assert(layerCount >= 2);

    def->layerCount = layerCount;
    memmove(def->layerNeuronCount, layerNeuronCount, sizeof(i32) * layerCount);

    def->inputNeuronCount = def->layerNeuronCount[0];
    def->neuronCount = def->inputNeuronCount;
    def->neuralNetSize = sizeof(NeuralNet);
    def->synapseTotalCount = 0;

    for(i32 l = 1; l < def->layerCount; ++l) {
        def->neuronCount += def->layerNeuronCount[l];
        // synapses
        i32 s = def->layerNeuronCount[l] * (def->layerNeuronCount[l-1] + 1); // + bias
        def->synapseTotalCount += s;
    }

    def->neuralNetSize += sizeof(f64) * def->neuronCount; // neuron values
    def->neuralNetSize += sizeof(f64) * def->synapseTotalCount;
    def->bias = bias;
}

struct FitnessPair
{
    i32 id;
    f64 fitness;
};

static i32 compareFitness(const void* a, const void* b)
{
    const FitnessPair* fa = (FitnessPair*)a;
    const FitnessPair* fb = (FitnessPair*)b;
    if(fa->fitness > fb->fitness) return -1;
    if(fa->fitness < fb->fitness) return 1;
    return 0;
}

i32 reinsertTruncate(i32 maxBest, i32 nnCount, f64* fitness, NeuralNet** nextGen,
                     NeuralNet** curGen, NeuralNetDef* def)
{
    assert(nnCount < 2048);
    FitnessPair list[2048];
    for(i32 i = 0; i < nnCount; ++i) {
        list[i].id = i;
        list[i].fitness = fitness[i];
    }

    qsort(list, nnCount, sizeof(FitnessPair), compareFitness);

    for(i32 i = 0; i < maxBest; ++i) {
        memmove(nextGen[i], curGen[list[i].id], def->neuralNetSize);
    }

    return maxBest;
}


void crossover(i32 id, i32 parentA, i32 parentB, NeuralNet** nextGen,
               NeuralNet** curGen, NeuralNetDef* def)
{
    for(i32 s = 0; s < def->synapseTotalCount; ++s) {
        // get weight from parent A
        if(xorshift64star() & 1) {
            nextGen[id]->weights[s] = curGen[parentA]->weights[s];
        }
        // get weight from parent B
        else {
            nextGen[id]->weights[s] = curGen[parentB]->weights[s];
        }
    }
}

inline i32 selectRandom(const i32 reinsertCount, i32 notThisId)
{
    i32 r = xorshift64star() % reinsertCount;
    while(r == notThisId) {
        r = xorshift64star() % reinsertCount;
    }
    return r;
}

i32 selectTournament(const i32 reinsertCount, const i32 tournamentSize, i32 notThisId, const f64* fitness)
{
    i32 champion = selectRandom(reinsertCount, notThisId);
    f32 championFitness = fitness[champion];

    for(i32 i = 0; i < tournamentSize; ++i) {
        i32 opponent = selectRandom(reinsertCount, notThisId);
        if(fitness[opponent] > championFitness) {
            champion = opponent;
            championFitness = fitness[opponent];
        }
    }

    return champion;
}


i32 mutate(f32 rate, f32 factor, i32 nnCount, NeuralNet** nextGen, NeuralNetDef* def)
{
    i32 mutationCount = 0;
    for(i32 i = 1; i < nnCount; ++i) {
        for(i32 s = 0; s < def->synapseTotalCount; ++s) {
            // mutate
            if(randf64(0.0, 1.0) < rate) {
                mutationCount++;
                nextGen[i]->weights[s] += randf64(-factor, factor);
            }
        }
    }

    return mutationCount;
}
