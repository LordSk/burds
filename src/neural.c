#include "neural.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

const u64 RAND_MAX3 = (u64)RAND_MAX*3;

inline f64 randf64(f64 min, f64 max)
{
    u64 r = (u64)rand() + (u64)rand() + (u64)rand();
    return min + (f64)r/RAND_MAX3 * (max - min);
}

u8* neuralNetAlloc(NeuralNet** nn, const i32 nnCount, const NeuralNetDef* def)
{
    i32 dataSize = nnCount * def->neuralNetSize;
    u8* data = (u8*)malloc(dataSize);

    for(i32 i = 0; i < nnCount; ++i) {
        nn[i] = (NeuralNet*)(data + def->neuralNetSize * i);
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
    // there probably is a way better way to traverse the net and compute values
    for(i32 i = 0; i < nnCount; ++i) {
        i32 synapseLayerOff = 0;
        i32 neuronCurLayerOff = 0;

        for(i32 l = 1; l < def->layerCount; ++l) {
            const i32 prevLayerNeuronCount = def->layerNeuronCount[l-1];
            const i32 neuronPrevLayerOff = neuronCurLayerOff;
            neuronCurLayerOff += prevLayerNeuronCount;

            for(i32 n = 0; n < def->layerNeuronCount[l]; ++n) {
                f64 value = 0.0;
                // weighted sum
                const i32 synapseNeuronOff = n * prevLayerNeuronCount;
                for(i32 s = 0; s < prevLayerNeuronCount; ++s) {
                    value += nn[i]->weights[synapseLayerOff + synapseNeuronOff + s] *
                            nn[i]->values[neuronPrevLayerOff + s];
                }
                value += nn[i]->weights[synapseLayerOff + synapseNeuronOff + prevLayerNeuronCount] *
                         def->bias; // bias

                 // "activate" value
                //nn[i]->values[neuronCurLayerOff + n] = tanh(value);
                nn[i]->values[neuronCurLayerOff + n] = 1.0 / (1.0 + exp(-value)); // sigmoid [0.05-0.995]
                //nn[i]->values[neuronCurLayerOff + n] = 1.0 / (1.0 + fabs(value)); // fast sigmoid [0.2-1.0]
            }

            synapseLayerOff += prevLayerNeuronCount * def->layerNeuronCount[l];
        }
    }
}

void makeNeuralNetDef(NeuralNetDef* def, const i32 layerCount, const i32 layerNeuronCount[], f32 bias)
{
    assert(layerCount >= 2);

    def->layerCount = layerCount;
    memmove(def->layerNeuronCount, layerNeuronCount, sizeof(i32) * layerCount);

    def->inputNeuronCount = def->layerNeuronCount[0];
    def->neuronCount = def->inputNeuronCount;
    def->neuralNetSize = sizeof(NeuralNet) - sizeof(f64);
    def->synapseTotalCount = 0;

    for(i32 l = 1; l < def->layerCount; ++l) {
        def->neuronCount += def->layerNeuronCount[l];
        // synapses
        i32 s = def->layerNeuronCount[l] * def->layerNeuronCount[l-1] + 1; // + bias
        def->synapseTotalCount += s;
        def->neuralNetSize += s * sizeof(f64);
    }

    def->neuralNetSize += sizeof(f64) * def->neuronCount; // neuron values
    def->bias = bias;
}
