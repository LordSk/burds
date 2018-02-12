#include "neural.h"
#include "base.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include <stddef.h>

#define sigmoid(val) (1.0 / (1.0 + expf(-val)))
#define activate(val) tanh(val)
//#define activate(val) min(max(0, val), 10.0)
//#define output_activate(val, outputCount) ((tanhf(val) + 1.0) * 0.5)
#define pow2(val) (val * val)

inline f64 exp1(f64 x)
{
    x = 1.0 + x / 1000.0;
    x *= x; x *= x; x *= x; x *= x;
    x *= x; x *= x; x *= x; x *= x;
    return x;
}


inline w128d wide_f64_exp1(w128d x)
{
    x = wide_f64_add(wide_f64_mul(x, wide_f64_set1(1.0/1000.0)), wide_f64_set1(1.0));
    x = wide_f64_mul(x, x);
    x = wide_f64_mul(x, x);
    x = wide_f64_mul(x, x);
    x = wide_f64_mul(x, x);
    x = wide_f64_mul(x, x);
    x = wide_f64_mul(x, x);
    x = wide_f64_mul(x, x);
    x = wide_f64_mul(x, x);
    return x;
}

u8* nnAlloc(NeuralNet** nn, const i32 nnCount, const NeuralNetDef* def)
{
    i32 dataSize = nnCount * def->neuralNetSize;
    u8* data = (u8*)_aligned_malloc(dataSize, 8);

    for(i32 i = 0; i < nnCount; ++i) {
        nn[i] = (NeuralNet*)(data + def->neuralNetSize * i);
        nn[i]->values = (f64*)(nn[i] + 1);
        nn[i]->weights = nn[i]->values + def->neuronCount;
        nn[i]->output = nn[i]->weights - def->outputNeuronCount;
    }

    LOG("allocated %d neural nets (layers=%d totalDataSize=%d)", nnCount, def->layerCount, dataSize);
    return data;
}

// set random synapse weight
void nnInitRandom(NeuralNet** nn, const i32 nnCount, const NeuralNetDef* def)
{
    for(i32 i = 0; i < nnCount; ++i) {
        for(i32 s = 0; s < def->weightTotalCount; ++s) {
            nn[i]->weights[s] = randf64(-1.0, 1.0);
        }
    }
}

void nnPropagate(NeuralNet** nn, const i32 nnCount, const NeuralNetDef* def)
{
    const f64 bias = def->bias;
    const i32 layerCount = def->layerCount;

    // there probably is a way better way to traverse the net and compute values
    for(i32 i = 0; i < nnCount; ++i) {
        f64* neuronPrevValues = nn[i]->values;
        f64* neuronCurValues = nn[i]->values;
        f64* neuronWeights = nn[i]->weights;

        for(i32 l = 1; l < layerCount-1; ++l) {
            const i32 prevLayerNeuronCount = def->layerNeuronCount[l-1];
            neuronCurValues = neuronPrevValues + prevLayerNeuronCount;
            const i32 layerNeuronCount = def->layerNeuronCount[l];

            for(i32 n = 0; n < layerNeuronCount; ++n) {
                // weighted sum
                f64 value = 0.0;
                for(i32 s = 0; s < prevLayerNeuronCount; ++s) {
                    value += neuronWeights[s] * neuronPrevValues[s];
                }
                value += bias; // bias

                neuronCurValues[n] = activate(value);
                neuronWeights += prevLayerNeuronCount;
            }

            neuronPrevValues += prevLayerNeuronCount;
        }

        const i32 prevLayerNeuronCount = def->layerNeuronCount[layerCount-2];
        const i32 layerNeuronCount = def->layerNeuronCount[layerCount-1];
        neuronCurValues = neuronPrevValues + prevLayerNeuronCount;
        f64 outputTotal = 0;
        for(i32 n = 0; n < layerNeuronCount; ++n) {
            // weighted sum
            f64 value = 0.0;
            for(i32 s = 0; s < prevLayerNeuronCount; ++s) {
                value += neuronWeights[s] * neuronPrevValues[s];
            }
            value += bias; // bias
            neuronCurValues[n] = value;
            outputTotal += value;
            neuronWeights += prevLayerNeuronCount;
        }

        // softmax
        for(i32 n = 0; n < layerNeuronCount; ++n) {
            neuronCurValues[n] /= outputTotal;
        }
    }
}

void nnMakeDef(NeuralNetDef* def, const i32 layerCount, const i32 layerNeuronCount[], f64 bias)
{
    assert(layerCount >= 2);

    def->layerCount = layerCount;
    memmove(def->layerNeuronCount, layerNeuronCount, sizeof(i32) * layerCount);

    def->inputNeuronCount = def->layerNeuronCount[0];
    def->outputNeuronCount = def->layerNeuronCount[layerCount-1];
    def->neuronCount = def->inputNeuronCount;
    def->neuralNetSize = sizeof(NeuralNet);
    def->weightTotalCount = 0;

    for(i32 l = 1; l < def->layerCount; ++l) {
        def->neuronCount += def->layerNeuronCount[l];
        // synapses
        i32 s = def->layerNeuronCount[l] * def->layerNeuronCount[l-1];
        def->weightTotalCount += s;
    }

    def->neuralNetSize += sizeof(f64) * def->neuronCount; // neuron values
    def->neuralNetSize += sizeof(f64) * def->weightTotalCount;
    def->bias = bias;
}

void rnnMakeDef(RecurrentNeuralNetDef* def, const i32 layerCount, const i32 layerNeuronCount[], f64 bias)
{
    assert(layerCount == 3);

    def->layerCount = layerCount;
    memmove(def->layerNeuronCount, layerNeuronCount, sizeof(i32) * layerCount);

    def->inputNeuronCount = def->layerNeuronCount[0];
    def->outputNeuronCount = def->layerNeuronCount[layerCount-1];
    def->neuronCount = def->inputNeuronCount;
    def->neuralNetSize = sizeof(RecurrentNeuralNet);
    def->weightTotalCount = 0;
    def->hiddenStateNeuronCount = 0;
    def->hiddenStateWeightCount = 0;

    for(i32 l = 1; l < def->layerCount; ++l) {
        def->neuronCount += def->layerNeuronCount[l];
        i32 w = def->layerNeuronCount[l] * def->layerNeuronCount[l-1];
        def->weightTotalCount += w;
    }

    for(i32 l = 1; l < def->layerCount-1; ++l) {
        def->hiddenStateNeuronCount += def->layerNeuronCount[l];
        def->hiddenStateWeightCount += (def->layerNeuronCount[l] * def->layerNeuronCount[l]);
    }

    def->neuronCount += def->hiddenStateNeuronCount;
    def->weightTotalCount += def->hiddenStateWeightCount;

    def->neuralNetSize += sizeof(f64) * def->neuronCount; // neuron values
    def->neuralNetSize += sizeof(f64) * def->weightTotalCount;
    def->neuralNetSize += alignof(RecurrentNeuralNet) - (def->neuralNetSize % alignof(RecurrentNeuralNet));
    def->bias = bias;
}

void rnnAlloc(RecurrentNeuralNet** nn, const i32 nnCount, const RecurrentNeuralNetDef* def)
{
    i32 dataSize = nnCount * def->neuralNetSize;
    u8* data = (u8*)_aligned_malloc(dataSize, alignof(RecurrentNeuralNet));

    for(i32 i = 0; i < nnCount; ++i) {
        nn[i] = (RecurrentNeuralNet*)(data + def->neuralNetSize * i);
        nn[i]->values = (f64*)(nn[i] + 1);
        nn[i]->weights = nn[i]->values + def->neuronCount;
        nn[i]->prevHiddenValues = nn[i]->values + def->neuronCount - def->hiddenStateNeuronCount;
        nn[i]->prevHiddenWeights = nn[i]->weights + def->weightTotalCount - def->hiddenStateWeightCount;
        nn[i]->output = nn[i]->prevHiddenValues - def->outputNeuronCount;
    }

    LOG("allocated %d RNN (layers=%d nnSize=%d totalDataSize=%d)", nnCount, def->layerCount,
        def->neuralNetSize, dataSize);
}


void rnnDealloc(void* ptr)
{
    _aligned_free(ptr);
}

void rnnInitRandom(RecurrentNeuralNet** nn, const i32 nnCount, const RecurrentNeuralNetDef* def)
{
    for(i32 i = 0; i < nnCount; ++i) {
        memset(nn[i]->values, 0, sizeof(nn[i]->values[0]) * def->neuronCount);
        for(i32 s = 0; s < def->weightTotalCount; ++s) {
            nn[i]->weights[s] = randf64(-1.0, 1.0);
        }
    }
}

void rnnPropagate(RecurrentNeuralNet** nn, const i32 nnCount, const RecurrentNeuralNetDef* def)
{
    const f64 bias = def->bias;
    const i32 inputNeuronCount = def->inputNeuronCount;
    const i32 outputNeuronCount = def->outputNeuronCount;
    const i32 hiddenNeuronCount = def->hiddenStateNeuronCount;

    for(i32 i = 0; i < nnCount; ++i) {
        f64* inputValues = nn[i]->values;
        f64* hiddenStateVals = nn[i]->values + inputNeuronCount;
        f64* weights = nn[i]->weights;
        f64* prevHiddenValues = nn[i]->prevHiddenValues;
        f64* prevHiddenWeights = nn[i]->prevHiddenWeights;
        f64* output = nn[i]->output;

        // compute new hidden state
        for(i32 n = 0; n < hiddenNeuronCount; ++n) {
            f64 value = 0.0;
            // input * inputWeights
            for(i32 s = 0; s < inputNeuronCount; ++s) {
                value += weights[s] * inputValues[s];
            }
            // prevSate * prevSateWeights
            for(i32 s = 0; s < hiddenNeuronCount; ++s) {
                value += prevHiddenWeights[s] * prevHiddenValues[s];
            }
            value += bias; // bias
            hiddenStateVals[n] = activate(value);

            weights += inputNeuronCount;
            prevHiddenWeights += hiddenNeuronCount;
        }

        f64 outputTotal = 0;
        for(i32 n = 0; n < outputNeuronCount; ++n) {
            f64 value = 0.0;
            // hiddenState * outputWeights
            for(i32 s = 0; s < hiddenNeuronCount; ++s) {
                value += weights[s] * hiddenStateVals[s];
            }
            value += bias; // bias
            /*output[n] = exp1(clamp(value, -10.0, 10.0));
            assert(output[n] >= 0.0);*/
            output[n] = (tanh(value) + 1.0) * 0.5;
            outputTotal += output[n];
            weights += hiddenNeuronCount;
        }

        // softmax
        /*if(outputTotal == 0) {
            memset(output, 0, sizeof(output[0]) * outputNeuronCount);
            output[0] = 1.0;
        }
        else {
            for(i32 n = 0; n < outputNeuronCount; ++n) {
                output[n] /= outputTotal;
                assert(output[n] >= 0.0 && output[n] <= 1.0);
            }
        }*/


        // "pass on" new hidden state
        memmove(prevHiddenValues, hiddenStateVals, hiddenNeuronCount * sizeof(hiddenStateVals[0]));
    }
}

void rnnPropagateWide(RecurrentNeuralNet** nn, const i32 nnCount, const RecurrentNeuralNetDef* def)
{
#if 0
    const f64 bias = def->bias;
    const i32 inputNeuronCount = def->inputNeuronCount;
    const i32 outputNeuronCount = def->outputNeuronCount;
    const i32 hiddenNeuronCount = def->hiddenStateNeuronCount;

    for(i32 i = 0; i < nnCount; ++i) {
        f64* inputValues = nn[i]->values;
        f64* hiddenStateVals = nn[i]->values + inputNeuronCount;
        f64* weights = nn[i]->weights;
        f64* prevHiddenValues = nn[i]->prevHiddenValues;
        f64* prevHiddenWeights = nn[i]->prevHiddenWeights;
        f64* output = nn[i]->output;

        // compute new hidden state
        for(i32 n = 0; n < hiddenNeuronCount; n += 2) {
            f64 value[2] = {0.0};
            // input * inputWeights
            for(i32 s = 0; s < inputNeuronCount; s += 2) {
                value[0] += weights[s] * inputValues[s];
                value[0] += weights[s+1] * inputValues[s+1];
                value[1] += weights[s+inputNeuronCount] * inputValues[s];
                value[1] += weights[s+inputNeuronCount+1] * inputValues[s+1];
            }
            // prevSate * prevSateWeights
            for(i32 s = 0; s < hiddenNeuronCount; s += 2) {
                value[0] += prevHiddenWeights[s] * prevHiddenValues[s];
                value[0] += prevHiddenWeights[s+1] * prevHiddenValues[s+1];
                value[1] += prevHiddenWeights[s+hiddenNeuronCount] * prevHiddenValues[s];
                value[1] += prevHiddenWeights[s+hiddenNeuronCount+1] * prevHiddenValues[s+1];
            }
            value[0] += bias; // bias
            value[1] += bias; // bias
            hiddenStateVals[n] = activate(value[0]);
            hiddenStateVals[n+1] = activate(value[1]);

            weights += inputNeuronCount * 2;
            prevHiddenWeights += hiddenNeuronCount * 2;
        }

        f64 outputTotal = 0;
        for(i32 n = 0; n < outputNeuronCount; n += 2) {
            f64 value[2] = {0.0};
            // hiddenState * outputWeights
            for(i32 s = 0; s < hiddenNeuronCount; s += 2) {
                value[0] += weights[s] * hiddenStateVals[s];
                value[0] += weights[s+1] * hiddenStateVals[s+1];
                value[1] += weights[s+hiddenNeuronCount] * hiddenStateVals[s];
                value[1] += weights[s+hiddenNeuronCount+1] * hiddenStateVals[s+1];
            }
            value[0] += bias; // bias
            value[1] += bias; // bias
            output[n] = exp1(clamp(value[0], -10.0, 10.0));
            output[n+1] = exp1(clamp(value[1], -10.0, 10.0));
            outputTotal += output[n];
            outputTotal += output[n+1];

            weights += hiddenNeuronCount * 2;
        }

        // softmax
        if(outputTotal == 0) {
            memset(output, 0, sizeof(output[0]) * outputNeuronCount);
            output[0] = 1.0;
        }
        else {
            for(i32 n = 0; n < outputNeuronCount; n += 2) {
                output[n] /= outputTotal;
                output[n+1] /= outputTotal;
            }
        }

        // "pass on" new hidden state
        memmove(prevHiddenValues, hiddenStateVals, hiddenNeuronCount * sizeof(hiddenStateVals[0]));
    }
#else
    assert((def->layerNeuronCount[0] & 1) == 0);
    assert((def->layerNeuronCount[1] & 1) == 0);
    assert((def->layerNeuronCount[2] & 1) == 0);

    const w128d bias = wide_f64_set1(def->bias);
    const w128d zero = wide_f64_zero();
    const w128d valmax = wide_f64_set1(10.0);
    const w128d valmin = wide_f64_set1(-5.0);
    const i32 inputNeuronCountHalf = def->inputNeuronCount / 2;
    const i32 outputNeuronCountHalf = def->outputNeuronCount / 2;
    const i32 hiddenNeuronCountHalf = def->hiddenStateNeuronCount / 2;

    for(i32 i = 0; i < nnCount; ++i) {
        w128d* inputValues = nn[i]->wide.values;
        w128d* hiddenStateVals = nn[i]->wide.values + inputNeuronCountHalf;
        w128d* weights = nn[i]->wide.weights;
        w128d* prevHiddenValues = nn[i]->wide.prevHiddenValues;
        w128d* prevHiddenWeights = nn[i]->wide.prevHiddenWeights;
        w128d* output = nn[i]->wide.output;

        // compute new hidden state
        for(i32 n = 0; n < hiddenNeuronCountHalf; n++) {
            w128d value = wide_f64_zero();
            // input * inputWeights
            for(i32 s = 0; s < inputNeuronCountHalf; s++) {
                value = wide_f64_add(value,
                                     wide_f64_hadd(wide_f64_mul(weights[s],
                                                                inputValues[s]),
                                                   wide_f64_mul((weights+inputNeuronCountHalf)[s],
                                                                inputValues[s])
                                                   )
                                     );

            }
            // prevSate * prevSateWeights
            for(i32 s = 0; s < hiddenNeuronCountHalf; ++s) {
                value = wide_f64_add(value,
                                     wide_f64_hadd(wide_f64_mul(prevHiddenWeights[s],
                                                                prevHiddenValues[s]),
                                                   wide_f64_mul((prevHiddenWeights+hiddenNeuronCountHalf)[s],
                                                                prevHiddenValues[s])
                                                   )
                                     );
            }
            value = wide_f64_add(value, bias); // bias
            hiddenStateVals[n] = wide_f64_min(wide_f64_max(zero, value), valmax); // activate

            weights += inputNeuronCountHalf * 2;
            prevHiddenWeights += hiddenNeuronCountHalf * 2;
        }

        w128d outputTotal = wide_f64_zero();
        for(i32 n = 0; n < outputNeuronCountHalf; ++n) {
            w128d value = wide_f64_zero();
            // hiddenState * outputWeights
            for(i32 s = 0; s < hiddenNeuronCountHalf; ++s) {
                value = wide_f64_add(value,
                                     wide_f64_hadd(wide_f64_mul(weights[s],
                                                                hiddenStateVals[s]),
                                                   wide_f64_mul((weights+hiddenNeuronCountHalf)[s],
                                                                hiddenStateVals[s])
                                                   )
                                     );
            }
            value = wide_f64_add(value, bias); // bias
            // clamp -10.0, 10.0
            value = wide_f64_max(value, valmin);
            value = wide_f64_min(value, valmax);
            output[n] = wide_f64_exp1(value);
            outputTotal = wide_f64_add(outputTotal, wide_f64_hadd(output[n], zero));

            weights += hiddenNeuronCountHalf * 2;
        }

        // softmax
        f64 fOutTotal = ((f64*)&outputTotal)[0];
        if(fOutTotal == 0) {
            memset(output, 0, sizeof(output[0]) * outputNeuronCountHalf);
            ((f64*)&output[0])[0] = 1.0;
        }
        else {
            w128d invOutputTotal = wide_f64_set1(1.0/fOutTotal);
            for(i32 n = 0; n < outputNeuronCountHalf; ++n) {
                output[n] = wide_f64_mul(output[n], invOutputTotal);
            }
        }

        // "pass on" new hidden state
        memmove(prevHiddenValues, hiddenStateVals, hiddenNeuronCountHalf * sizeof(hiddenStateVals[0]));
    }
#endif
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

i32 reinsertTruncateNN(i32 maxBest, i32 nnCount, f64* fitness, NeuralNet** nextGen,
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

i32 reinsertTruncateRNN(i32 maxBest, i32 nnCount, f64* fitness, RecurrentNeuralNet** nextGen,
                        RecurrentNeuralNet** curGen, RecurrentNeuralNetDef* def)
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


i32 reinsertTruncateRNNSpecies(i32 maxBest, i32 nnCount, f64* fitness, RecurrentNeuralNet** nextGen,
                               RecurrentNeuralNet** curGen, RecurrentNeuralNetDef* def, u8* curSpecies,
                               u8* nextSpecies)
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
        nextSpecies[i] = curSpecies[list[i].id];
    }

    return maxBest;
}


void crossover(f64* outWeights, f64* parentBWeights, f64* parentAWeights, i32 weightCount)
{
    for(i32 s = 0; s < weightCount; ++s) {
        // get weight from parent A
        if(xorshift64star() & 1) {
            outWeights[s] = parentAWeights[s];
        }
        // get weight from parent B
        else {
            outWeights[s] = parentBWeights[s];
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

i32 selectTournamentSpecies(const i32 count, i32 tries, i32 notThisId, const f64* fitness,
                            const u8* speciesTags, const u8 thisTag)
{
    i32 champion = selectRandom(count, notThisId);
    while(speciesTags[champion] != thisTag && tries--) {
        champion = selectRandom(count, notThisId);
    }

    f32 championFitness = fitness[champion];
    for(i32 i = 0; i < tries; ++i) {
        i32 opponent = selectRandom(count, notThisId);
        while(speciesTags[opponent] != thisTag && tries--) {
            opponent = selectRandom(count, notThisId);
        }
        if(fitness[opponent] > championFitness) {
            champion = opponent;
            championFitness = fitness[opponent];
        }
    }

    return champion;
}


i32 mutateNN(f32 rate, f32 factor, i32 nnCount, NeuralNet** nextGen, NeuralNetDef* def)
{
    i32 mutationCount = 0;
    for(i32 i = 0; i < nnCount; ++i) {
        for(i32 s = 0; s < def->weightTotalCount; ++s) {
            // mutate
            if(randf64(0.0, 1.0) < rate) {
                mutationCount++;
                nextGen[i]->weights[s] += randf64(-factor, factor);
            }
        }
    }

    return mutationCount;
}

i32 mutateRNN(f32 rate, f32 factor, i32 nnCount, RecurrentNeuralNet** nextGen, RecurrentNeuralNetDef* def)
{
    i32 mutationCount = 0;
    for(i32 i = 1; i < nnCount; ++i) {
        for(i32 s = 0; s < def->weightTotalCount; ++s) {
            // mutate
            if(randf64(0.0, 1.0) < rate) {
                mutationCount++;
                nextGen[i]->weights[s] += randf64(-factor, factor);
            }
        }
    }

    return mutationCount;
}

void testPropagateNN()
{
    f64 inputs[2] = { randf64(-5.0, 5.0), randf64(-5.0, 5.0) };
    f64 values[3] = {0, 0, 0};
    f64 bias = 1.0;
    f64 weights1[3 * 2] = { randf64(-1.0, 1.0), randf64(-1.0, 1.0), randf64(-1.0, 1.0),
                            randf64(-1.0, 1.0), randf64(-1.0, 1.0), randf64(-1.0, 1.0) };
    f64 weights2[3 * 2] = { randf64(-1.0, 1.0), randf64(-1.0, 1.0), randf64(-1.0, 1.0),
                            randf64(-1.0, 1.0), randf64(-1.0, 1.0), randf64(-1.0, 1.0) };
    f64 output[2] = {0, 0};

    for(i32 i = 0; i < 3; ++i) {
        values[i] += inputs[0] * weights1[i*2];
        values[i] += inputs[1] * weights1[i*2+1];
        values[i] += bias;
        values[i] = activate(values[i]);
    }

    f64 outputTotal = 0;
    for(i32 i = 0; i < 2; ++i) {
        output[i] += values[0] * weights2[i*3];
        output[i] += values[1] * weights2[i*3+1];
        output[i] += values[2] * weights2[i*3+2];
        output[i] += bias;
        outputTotal += output[i];
    }

    for(i32 i = 0; i < 2; ++i) {
        output[i] /= outputTotal;
    }

    NeuralNetDef def;
    const i32 layers[] = {2, 3, 2};
    nnMakeDef(&def, sizeof(layers) / sizeof(layers[0]), layers, 1.0);

    NeuralNet* nn;
    nnAlloc(&nn, 1, &def);

    assert(def.neuronCount == 7);
    assert(def.weightTotalCount == (3 * 2 + 3 * 2));
    nn->values[0] = inputs[0];
    nn->values[1] = inputs[1];
    memmove(nn->weights, weights1, sizeof(weights1));
    memmove(nn->weights + 6, weights2, sizeof(weights2));

    nnPropagate(&nn, 1, &def);

    for(i32 i = 0; i < 3; ++i) {
        assert(nn->values[2 + i] == values[i]);
    }
    assert(nn->output[0] == output[0]);
    assert(nn->output[1] == output[1]);

    _aligned_free(nn);
}

void testPropagateRNN()
{
    assert(exp(-5.0) >= 0.0);
    f64 inputs[2] = { randf64(0.0, 5.0), randf64(0.0, 5.0) };
    f64 hiddenVals[4] = {0};
    f64 prevHiddenVals[4] = {randf64(0.0, 1.0), randf64(0.0, 1.0)};
    f64 prevHiddenWeights[4 * 4] = {randf64(0.0, 1.0), randf64(0.0, 1.0),
                                    randf64(0.0, 1.0), randf64(0.0, 1.0),
                                    randf64(0.0, 1.0), randf64(0.0, 1.0),
                                    randf64(0.0, 1.0), randf64(0.0, 1.0),
                                    randf64(0.0, 1.0), randf64(0.0, 1.0),
                                    randf64(0.0, 1.0), randf64(0.0, 1.0),
                                    randf64(0.0, 1.0), randf64(0.0, 1.0),
                                    randf64(0.0, 1.0), randf64(0.0, 1.0)};
    f64 bias = 1.0;
    f64 weights1[4 * 2] = { randf64(0.0, 1.0), randf64(0.0, 1.0),
                            randf64(0.0, 1.0), randf64(0.0, 1.0),
                            randf64(0.0, 1.0), randf64(0.0, 1.0),
                            randf64(0.0, 1.0), randf64(0.0, 1.0)};
    f64 weights2[2 * 4] = { randf64(0.0, 1.0), randf64(0.0, 1.0),
                            randf64(0.0, 1.0), randf64(0.0, 1.0),
                            randf64(0.0, 1.0), randf64(0.0, 1.0),
                            randf64(0.0, 1.0), randf64(0.0, 1.0) };
    f64 output[2] = {0, 0};


    for(i32 i = 0; i < 4; ++i) {
        hiddenVals[i] += inputs[0] * weights1[i*2];
        hiddenVals[i] += inputs[1] * weights1[i*2+1];
        hiddenVals[i] += prevHiddenVals[0] * prevHiddenWeights[i*4];
        hiddenVals[i] += prevHiddenVals[1] * prevHiddenWeights[i*4+1];
        hiddenVals[i] += prevHiddenVals[2] * prevHiddenWeights[i*4+2];
        hiddenVals[i] += prevHiddenVals[3] * prevHiddenWeights[i*4+3];
        hiddenVals[i] += bias;
        hiddenVals[i] = activate(hiddenVals[i]);
    }

    f64 outputTotal = 0;
    for(i32 i = 0; i < 2; ++i) {
        output[i] += hiddenVals[0] * weights2[i*4];
        output[i] += hiddenVals[1] * weights2[i*4+1];
        output[i] += hiddenVals[2] * weights2[i*4+2];
        output[i] += hiddenVals[3] * weights2[i*4+3];
        output[i] += bias;
        output[i] = exp1(clamp(output[i], -10.0, 10.0));
        outputTotal += output[i];
    }

    for(i32 i = 0; i < 2; ++i) {
        output[i] /= outputTotal;
    }

    RecurrentNeuralNetDef def;
    const i32 layers[] = {2, 4, 2};
    rnnMakeDef(&def, sizeof(layers) / sizeof(layers[0]), layers, 1.0);

    RecurrentNeuralNet* nn;
    rnnAlloc(&nn, 1, &def);

    assert(def.neuronCount == 8 + 4);
    assert(def.weightTotalCount == (4 * 2 + 4 * 4 + 2 * 4));
    memmove(nn->values, inputs, sizeof(inputs));
    memmove(nn->weights, weights1, sizeof(weights1));
    memmove(nn->weights + array_count(weights1), weights2, sizeof(weights2));
    memmove(nn->prevHiddenValues, prevHiddenVals, sizeof(prevHiddenVals));
    memmove(nn->prevHiddenWeights, prevHiddenWeights, sizeof(prevHiddenWeights));

    rnnPropagate(&nn, 1, &def);

    for(i32 i = 0; i < 4; ++i) {
        assert(fabs(nn->values[2 + i] - hiddenVals[i]) < 0.0001);
    }
    assert(fabs(nn->output[0] - output[0]) < 0.0001);
    assert(fabs(nn->output[1] - output[1]) < 0.0001);

    rnnDealloc(nn);
}

void testPropagateRNNWide()
{
    RecurrentNeuralNetDef def;
    const i32 layers[] = {8, 16, 4};
    rnnMakeDef(&def, array_count(layers), layers, 1.0);

    RecurrentNeuralNet* nn[2];
    rnnAlloc(nn, 2, &def);

    rnnInitRandom(&nn[0], 1, &def);
    nn[0]->values[0] = randf64(0, 1.0);
    nn[0]->values[1] = randf64(0, 1.0);
    nn[0]->values[2] = randf64(0, 1.0);
    nn[0]->values[3] = randf64(0, 1.0);
    nn[0]->values[4] = randf64(0, 1.0);
    nn[0]->values[5] = randf64(0, 1.0);
    nn[0]->values[6] = randf64(0, 1.0);
    nn[0]->values[7] = randf64(0, 1.0);

    memmove(nn[1], nn[0], def.neuralNetSize);

    rnnPropagate(&nn[0], 1, &def);
    rnnPropagateWide(&nn[1], 1, &def);

    for(i32 i = 0; i < def.neuronCount; ++i) {
        assert(fabs(nn[0]->values[i] - nn[1]->values[i]) < 0.0001);
    }

    rnnDealloc(nn[0]);
}
