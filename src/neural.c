#include "neural.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

const u64 RAND_MAX3 = (u64)RAND_MAX*3;

inline f64 randf64(f64 min, f64 max)
{
    u64 r = (u64)rand() + (u64)rand() + (u64)rand();
    return min + (f64)r/RAND_MAX3 * (max - min);
}

u8* allocNeuralNets(NeuralNet** nn, const i32 nnCount, const NeuralNetDef* def)
{
    i32 dataSize = nnCount * def->neuralNetSize;
    u8* data = (u8*)malloc(dataSize);

    for(i32 i = 0; i < nnCount; ++i) {
        nn[i] = (NeuralNet*)(data + def->neuralNetSize * i);
    }

    LOG("allocated %d neural nets (layers=%d totalDataSize=%d)", nnCount, def->layerCount, dataSize);
    return data;
}

// set random synapse weight
void initNeuralNets(NeuralNet** nn, const i32 nnCount, const NeuralNetDef* def)
{
    for(i32 i = 0; i < nnCount; ++i) {
        i32 curLayerNeuronIdOff = 0;
        for(i32 l = 1; l < def->layerCount; ++l) {
            const i32 prevLayerNeuronCount = def->layerNeuronCount[l-1];
            curLayerNeuronIdOff += prevLayerNeuronCount;

            for(i32 n = 0; n < def->layerNeuronCount[l]; ++n) {
                for(i32 s = 0; s < prevLayerNeuronCount; ++s) {
                    nn[i]->neurons[curLayerNeuronIdOff + n].synapseWeight[s] = randf64(-1, 1.0);
                }
            }
        }
    }
}

void propagateNeuralNets(NeuralNet** nn, const i32 nnCount, const NeuralNetDef* def)
{
    for(i32 i = 0; i < nnCount; ++i) {
        i32 prevLayerNeuronIdOff = 0;
        i32 curLayerNeuronIdOff = 0;
        for(i32 l = 1; l < def->layerCount; ++l) {
            const i32 prevLayerNeuronCount = def->layerNeuronCount[l-1];
            curLayerNeuronIdOff += prevLayerNeuronCount;

            for(i32 n = 0; n < def->layerNeuronCount[l]; ++n) {
                Neuron* curNeuron = &nn[i]->neurons[curLayerNeuronIdOff + n];

                f64 value = 0.0;
                // weighted sum
                for(i32 s = 0; s < prevLayerNeuronCount; ++s) {
                    value += curNeuron->synapseWeight[s] * nn[i]->neurons[prevLayerNeuronIdOff + s].value;
                }
                 // "activate" value
                curNeuron->value = tanh(value) > 0.0 ? 1.0 : 0.0;
                //curNeuron->value = 1.0 / (1.0 + exp(-value)); // sigmoid [0.05-0.995]
                //curNeuron->value = 1.0 / (1.0 + fabs(value)); // fast sigmoid [0.2-1.0]
            }

            prevLayerNeuronIdOff += prevLayerNeuronCount;
        }
    }
}

void makeNeuralNetDef(NeuralNetDef* def, const i32 layerCount, const i32 layerNeuronCount[])
{
    def->layerCount = layerCount;
    memmove(def->layerNeuronCount, layerNeuronCount, sizeof(i32) * layerCount);
    def->neuronCount = 0;
    for(i32 l = 0; l < def->layerCount; ++l) {
        def->neuronCount += def->layerNeuronCount[l];
    }
    def->neuralNetSize = sizeof(Neuron) * def->neuronCount;
    def->inputNeuronCount = def->layerNeuronCount[0];
    def->synapseTotalCount = (def->neuronCount - def->inputNeuronCount) *
                                       MAX_SYNAPSE_COUNT;
}
