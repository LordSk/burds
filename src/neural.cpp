#include "neural.h"
#include "base.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include <stddef.h>
#include "imgui/imgui.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui/imgui_internal.h"

#define stack_arr(type, count) ((type*)alloca(sizeof(type) * count))
#define zero_arr(arr, count) (memset(arr, 0, sizeof(arr[0]) * count))

#define sigmoid(val) (1.0 / (1.0 + expf(-val)))
//#define hidden_activate(val) min(max(0, val), 10.0)
#define hidden_activate(val) tanh(val)
#define output_activate(val) ((tanh(val) + 1.0) * 0.5)
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

inline w128d wide_f64_abs(w128d src)
{
    w128d zero = wide_f64_zero();
    w128d minus1 = wide_f64_set1(-1.0);
    w128d ltMask = wide_f64_less_than(src, zero);
    w128d srcAbs = wide_f64_mul(src, minus1);
    srcAbs = wide_f64_blendv(src, srcAbs, ltMask);
    return srcAbs;
}

inline w128d wide_f64_tanh(w128d src)
{
    // x = abs(src[i]);
    w128d zero = wide_f64_zero();
    w128d minus1 = wide_f64_set1(-1.0);
    w128d ltMask = wide_f64_less_than(src, zero);
    w128d x = wide_f64_mul(src, minus1);
    x = wide_f64_blendv(src, x, ltMask);

    w128d one = wide_f64_set1(1.0);
    w128d f1 = wide_f64_set1(0.5658);
    w128d f2 = wide_f64_set1(0.1430);
    // x*x*x*x*0.1430
    w128d e1 = wide_f64_mul(f2, x);
    e1 = wide_f64_mul(e1, x);
    e1 = wide_f64_mul(e1, x);
    e1 = wide_f64_mul(e1, x);
    // x*x*0.5658
    w128d e2 = wide_f64_mul(f1, x);
    e2 = wide_f64_mul(e2, x);
    // e = 1 + x + e1 + e2
    w128d e = wide_f64_add(e1, e2);
    e = wide_f64_add(e, x);
    e = wide_f64_add(e, one);
    // out[i] = (src[i] > 0 ? 1 : -1)*(e - 1/e)/(e + 1/e);
    w128d div1e = wide_f64_div(one, e);
    w128d out = wide_f64_div(wide_f64_sub(e, div1e), wide_f64_add(e, div1e));
    w128d m1 = wide_f64_blendv(one, minus1, ltMask);
    out = wide_f64_mul(out, m1);
    return out;
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

                neuronCurValues[n] = hidden_activate(value);
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
    assert(layerCount >= 2);
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
#ifdef CONF_DEBUG
        memset(nn[i], 0xAB+i, def->neuralNetSize);
#endif
        nn[i]->values = (f64*)(nn[i] + 1);
        assert((intptr_t)nn[i]->values - (intptr_t)nn[i] == sizeof(RecurrentNeuralNet));
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


void rnnCopy(RecurrentNeuralNet* dest, RecurrentNeuralNet* src, RecurrentNeuralNetDef* def)
{
    const i32 weightTotalCount = def->weightTotalCount;
    const i32 neuronCount = def->neuronCount;

    for(i32 i = 0; i < weightTotalCount; ++i) {
        dest->weights[i] = src->weights[i];
    }
    for(i32 i = 0; i < neuronCount; ++i) {
        dest->values[i] = src->values[i];
    }
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
    const i32 layerCount = def->layerCount;
    const i32 inputNeuronCount = def->inputNeuronCount;
    const i32 outputNeuronCount = def->outputNeuronCount;
    const i32 hiddenStateNeuronCount = def->hiddenStateNeuronCount;

    for(i32 i = 0; i < nnCount; ++i) {
        f64* prevLayerVals = nn[i]->values;
        f64* hiddenStateVals = nn[i]->values + inputNeuronCount;
        f64* weights = nn[i]->weights;
        f64* prevHiddenValues = nn[i]->prevHiddenValues;
        f64* prevHiddenWeights = nn[i]->prevHiddenWeights;
        f64* output = nn[i]->output;

        // compute new hidden state
        for(i32 l = 1; l < layerCount-1; ++l) {
            const i32 prevNeuronCount = def->layerNeuronCount[l-1];
            const i32 hiddenNeuronCount = def->layerNeuronCount[l];

            for(i32 n = 0; n < hiddenNeuronCount; ++n) {
                f64 value = 0.0;
                // prevLayervals * prevLayerWeights
                for(i32 s = 0; s < prevNeuronCount; ++s) {
                    value += weights[s] * prevLayerVals[s];
                }
                // prevSate * prevSateWeights
                for(i32 s = 0; s < hiddenNeuronCount; ++s) {
                    value += prevHiddenWeights[s] * prevHiddenValues[s];
                }
                value += bias; // bias
                hiddenStateVals[n] = hidden_activate(value);

                weights += prevNeuronCount;
                prevHiddenWeights += hiddenNeuronCount;
            }


            prevLayerVals += prevNeuronCount;
            hiddenStateVals += hiddenNeuronCount;
            prevHiddenValues += hiddenNeuronCount;
        }

        //f64 outputTotal = 0;
        const i32 prevNeuronCount = def->layerNeuronCount[layerCount-2];
        f64* lastHiddenVals = output - prevNeuronCount;
        for(i32 n = 0; n < outputNeuronCount; ++n) {
            f64 value = 0.0;
            // hiddenState * outputWeights
            for(i32 s = 0; s < prevNeuronCount; ++s) {
                value += weights[s] * lastHiddenVals[s];
            }
            value += bias; // bias
            output[n] = output_activate(value);

            weights += prevNeuronCount;
        }

        // "pass on" new hidden state
        hiddenStateVals = nn[i]->values + inputNeuronCount;
        prevHiddenValues = nn[i]->prevHiddenValues;
        memmove(prevHiddenValues, hiddenStateVals, hiddenStateNeuronCount * sizeof(hiddenStateVals[0]));
    }
}

void rnnPropagateWide(RecurrentNeuralNet** nn, const i32 nnCount, const RecurrentNeuralNetDef* def)
{
    for(i32 l = 0; l < def->layerCount; ++l) {
        assert((def->layerNeuronCount[l] & 1) == 0);
    }

#if 0
    const f64 bias = def->bias;
    const i32 layerCount = def->layerCount;
    const i32 inputNeuronCount = def->inputNeuronCount;
    const i32 outputNeuronCount = def->outputNeuronCount;
    const i32 hiddenStateNeuronCount = def->hiddenStateNeuronCount;

    for(i32 i = 0; i < nnCount; ++i) {
        f64* prevLayerVals = nn[i]->values;
        f64* hiddenStateVals = nn[i]->values + inputNeuronCount;
        f64* weights = nn[i]->weights;
        f64* prevHiddenValues = nn[i]->prevHiddenValues;
        f64* prevHiddenWeights = nn[i]->prevHiddenWeights;
        f64* output = nn[i]->output;

        // compute new hidden state
        for(i32 l = 1; l < layerCount-1; ++l) {
            const i32 prevNeuronCount = def->layerNeuronCount[l-1];
            const i32 hiddenNeuronCount = def->layerNeuronCount[l];

            for(i32 n = 0; n < hiddenNeuronCount; n += 2) {
                f64 value[2] = {0.0};
                // input * inputWeights
                for(i32 s = 0; s < prevNeuronCount; s += 2) {
                    value[0] += weights[s] * prevLayerVals[s];
                    value[0] += weights[s+1] * prevLayerVals[s+1];
                    value[1] += weights[s+prevNeuronCount] * prevLayerVals[s];
                    value[1] += weights[s+prevNeuronCount+1] * prevLayerVals[s+1];
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
                hiddenStateVals[n] = hidden_activate(value[0]);
                hiddenStateVals[n+1] = hidden_activate(value[1]);

                weights += prevNeuronCount * 2;
                prevHiddenWeights += hiddenNeuronCount * 2;
            }

            prevLayerVals += prevNeuronCount;
            hiddenStateVals += hiddenNeuronCount;
            prevHiddenValues += hiddenNeuronCount;
        }

        const i32 prevNeuronCount = def->layerNeuronCount[layerCount-2];
        f64* lastHiddenVals = output - prevNeuronCount;
        for(i32 n = 0; n < outputNeuronCount; n += 2) {
            f64 value[2] = {0.0};
            // hiddenState * outputWeights
            for(i32 s = 0; s < prevNeuronCount; s += 2) {
                value[0] += weights[s] * lastHiddenVals[s];
                value[0] += weights[s+1] * lastHiddenVals[s+1];
                value[1] += weights[s+prevNeuronCount] * lastHiddenVals[s];
                value[1] += weights[s+prevNeuronCount+1] * lastHiddenVals[s+1];
            }
            value[0] += bias; // bias
            value[1] += bias; // bias
            output[n] = output_activate(value[0]);
            output[n+1] = output_activate(value[1]);

            weights += prevNeuronCount * 2;
        }

        // "pass on" new hidden state
        hiddenStateVals = nn[i]->values + inputNeuronCount;
        prevHiddenValues = nn[i]->prevHiddenValues;
        memmove(prevHiddenValues, hiddenStateVals, hiddenStateNeuronCount * sizeof(hiddenStateVals[0]));
    }
#else
    const w128d bias = wide_f64_set1(def->bias);
    const w128d zero = wide_f64_zero();
    const w128d one = wide_f64_set1(1.0);
    const w128d half = wide_f64_set1(0.5);
    const w128d valmax = wide_f64_set1(10.0);
    const w128d valmin = wide_f64_set1(-5.0);
    const i32 layerCount = def->layerCount;
    const i32 inputNeuronCountHalf = def->inputNeuronCount / 2;
    const i32 outputNeuronCountHalf = def->outputNeuronCount / 2;
    const i32 hiddenStateNeuronCount = def->hiddenStateNeuronCount;

    for(i32 i = 0; i < nnCount; ++i) {
        w128d* prevLayerVals = nn[i]->wide.values;
        w128d* hiddenStateVals = nn[i]->wide.values + inputNeuronCountHalf;
        w128d* weights = nn[i]->wide.weights;
        w128d* prevHiddenValues = nn[i]->wide.prevHiddenValues;
        w128d* prevHiddenWeights = nn[i]->wide.prevHiddenWeights;
        w128d* output = nn[i]->wide.output;

        // compute new hidden state
        for(i32 l = 1; l < layerCount-1; ++l) {
            const i32 prevNeuronCountHalf = def->layerNeuronCount[l-1] / 2;
            const i32 hiddenNeuronCountHalf = def->layerNeuronCount[l] / 2;

            for(i32 n = 0; n < hiddenNeuronCountHalf; n++) {
                w128d value = wide_f64_zero();
                // input * inputWeights
                for(i32 s = 0; s < prevNeuronCountHalf; s++) {
                    value = wide_f64_add(value,
                             wide_f64_hadd(wide_f64_mul(weights[s],
                                                        prevLayerVals[s]),
                                           wide_f64_mul((weights+prevNeuronCountHalf)[s],
                                                        prevLayerVals[s])
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
                hiddenStateVals[n] = wide_f64_tanh(value); // activate

                weights += prevNeuronCountHalf * 2;
                prevHiddenWeights += hiddenNeuronCountHalf * 2;
            }

            prevLayerVals += prevNeuronCountHalf;
            hiddenStateVals += hiddenNeuronCountHalf;
            prevHiddenValues += hiddenNeuronCountHalf;
        }

        const i32 prevNeuronCountHalf = def->layerNeuronCount[layerCount-2] / 2;
        w128d* lastHiddenVals = output - prevNeuronCountHalf;
        for(i32 n = 0; n < outputNeuronCountHalf; ++n) {
            w128d value = wide_f64_zero();
            // hiddenState * outputWeights
            for(i32 s = 0; s < prevNeuronCountHalf; ++s) {
                value = wide_f64_add(value,
                                     wide_f64_hadd(wide_f64_mul(weights[s],
                                                                lastHiddenVals[s]),
                                                   wide_f64_mul((weights+prevNeuronCountHalf)[s],
                                                                lastHiddenVals[s])
                                                   )
                                     );
            }
            value = wide_f64_add(value, bias); // bias
            // activate
            value = wide_f64_tanh(value);
            output[n] = wide_f64_mul(wide_f64_add(value, one), half);

            weights += prevNeuronCountHalf * 2;
        }

        // "pass on" new hidden state
        f64* hiddenStateVals1 = nn[i]->values + inputNeuronCountHalf * 2;
        f64* prevHiddenValues1 = nn[i]->prevHiddenValues;
        memmove(prevHiddenValues1, hiddenStateVals1, hiddenStateNeuronCount * sizeof(hiddenStateVals1[0]));
    }
#endif
}


struct FitnessPair
{
    i32 id;
    f64 fitness;
};

static i32 compareFitnessDesc(const void* a, const void* b)
{
    const FitnessPair* fa = (FitnessPair*)a;
    const FitnessPair* fb = (FitnessPair*)b;
    if(fa->fitness > fb->fitness) return -1;
    if(fa->fitness < fb->fitness) return 1;
    return 0;
}

static i32 compareFitnessAsc(const void* a, const void* b)
{
    const FitnessPair* fa = (FitnessPair*)a;
    const FitnessPair* fb = (FitnessPair*)b;
    if(fa->fitness < fb->fitness) return -1;
    if(fa->fitness > fb->fitness) return 1;
    return 0;
}

i32 selectRoulette(const i32 count, f64* fitness, f64 totalFitness)
{
    assert(totalFitness > 0);
    f64 r = randf64(0.0, totalFitness);
    f64 s = 0.0;
    for(i32 j = 0; j < count; ++j) {
        s += fitness[j];
        if(r < s) {
            return j;
        }
    }
    assert(0);
    return -1;
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

    qsort(list, nnCount, sizeof(FitnessPair), compareFitnessDesc);

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

    qsort(list, nnCount, sizeof(FitnessPair), compareFitnessDesc);

    for(i32 i = 0; i < maxBest; ++i) {
        memmove(nextGen[i], curGen[list[i].id], def->neuralNetSize);
    }

    return maxBest;
}


i32 reinsertTruncateRNNSpecies(i32 maxBest, GeneticEnvRnn* env)
{
    const i32 popCount = env->populationCount;
    const f64* fitness = env->fitness;
    RecurrentNeuralNet** curGen = env->curPopRNN;
    RecurrentNeuralNet** nextGen = env->nextPopRNN;
    const i32 nnSize = env->rnnDef->neuralNetSize;
    u8* curSpecies = env->curSpeciesTags;
    u8* nextSpecies = env->nextSpeciesTags;

    assert(popCount < 2048);
    FitnessPair list[2048];

    for(i32 i = 0; i < popCount; ++i) {
        list[i].id = i;
        list[i].fitness = fitness[i];
    }

    qsort(list, popCount, sizeof(FitnessPair), compareFitnessDesc);

    for(i32 i = 0; i < maxBest; ++i) {
        memmove(nextGen[i], curGen[list[i].id], nnSize);
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
    i32 r = randi64(0, reinsertCount);
    while(r == notThisId) {
        r = randi64(0, reinsertCount);
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
    for(i32 i = 0; i < 15 && tries > 0; ++i) {
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
        values[i] = hidden_activate(values[i]);
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
    const i32 PASSES = 4;
    assert(exp(-5.0) >= 0.0);
    f64 inputs[2] = { randf64(0.0, 1.0), randf64(0.0, 1.0) };
    f64 hiddenVals[4] = {0};
    f64 hiddenVals2[2] = {0};
    f64 prevHiddenVals[4] = {randf64(0.0, 1.0), randf64(0.0, 1.0),
                             randf64(0.0, 1.0), randf64(0.0, 1.0)};
    f64 prevHiddenVals2[2] = {randf64(0.0, 1.0), randf64(0.0, 1.0)};
    f64 prevHiddenWeights[4 * 4] = {
        randf64(0.0, 1.0), randf64(0.0, 1.0), randf64(0.0, 1.0), randf64(0.0, 1.0),
        randf64(0.0, 1.0), randf64(0.0, 1.0), randf64(0.0, 1.0), randf64(0.0, 1.0),
        randf64(0.0, 1.0), randf64(0.0, 1.0), randf64(0.0, 1.0), randf64(0.0, 1.0),
        randf64(0.0, 1.0), randf64(0.0, 1.0), randf64(0.0, 1.0), randf64(0.0, 1.0)};

    f64 prevHiddenWeights2[2 * 2] = {randf64(0.0, 1.0), randf64(0.0, 1.0),
                                     randf64(0.0, 1.0), randf64(0.0, 1.0)};
    f64 bias = 1.0;
    f64 weights1[4 * 2] = { randf64(0.0, 1.0), randf64(0.0, 1.0),
                            randf64(0.0, 1.0), randf64(0.0, 1.0),
                            randf64(0.0, 1.0), randf64(0.0, 1.0),
                            randf64(0.0, 1.0), randf64(0.0, 1.0)};

    f64 weights2[4 * 2] = { randf64(0.0, 1.0), randf64(0.0, 1.0),
                            randf64(0.0, 1.0), randf64(0.0, 1.0),
                            randf64(0.0, 1.0), randf64(0.0, 1.0),
                            randf64(0.0, 1.0), randf64(0.0, 1.0)};

    f64 outWeights[2 * 2] = { randf64(0.0, 1.0), randf64(0.0, 1.0),
                              randf64(0.0, 1.0), randf64(0.0, 1.0)};
    f64 output[2] = {0, 0};

    /*for(i32 i = 0; i < 2; ++i) {
        output[i] /= outputTotal;
    }*/

    RecurrentNeuralNetDef def;
    const i32 layers[] = {2, 4, 2, 2};
    rnnMakeDef(&def, array_count(layers), layers, 1.0);

    RecurrentNeuralNet* nn;
    rnnAlloc(&nn, 1, &def);

    assert(def.neuronCount == 10 + 6);
    assert(def.weightTotalCount == (4 * 2 + 4 * 4 + 4 * 2 + 2 * 2 + 2 * 2));
    memmove(nn->values, inputs, sizeof(inputs));

    f64* nnWeights = nn->weights;
    memmove(nnWeights, weights1, sizeof(weights1));
    nnWeights += array_count(weights1);
    memmove(nnWeights, weights2, sizeof(weights2));
    nnWeights += array_count(weights2);
    memmove(nnWeights, outWeights, sizeof(outWeights));

    memmove(nn->prevHiddenValues, prevHiddenVals, sizeof(prevHiddenVals));
    memmove(nn->prevHiddenValues + array_count(prevHiddenVals), prevHiddenVals2, sizeof(prevHiddenVals2));
    memmove(nn->prevHiddenWeights, prevHiddenWeights, sizeof(prevHiddenWeights));
    memmove(nn->prevHiddenWeights + array_count(prevHiddenWeights),
            prevHiddenWeights2, sizeof(prevHiddenWeights2));


    for(i32 p = 0; p < PASSES; ++p) {
        for(i32 i = 0; i < 4; ++i) {
            f64 value = 0.0;
            value += inputs[0] * weights1[i*2];
            value += inputs[1] * weights1[i*2+1];
            value += prevHiddenVals[0] * prevHiddenWeights[i*4];
            value += prevHiddenVals[1] * prevHiddenWeights[i*4+1];
            value += prevHiddenVals[2] * prevHiddenWeights[i*4+2];
            value += prevHiddenVals[3] * prevHiddenWeights[i*4+3];
            value += bias;
            hiddenVals[i] = hidden_activate(value);
        }

        for(i32 i = 0; i < 2; ++i) {
            f64 value = 0.0;
            value += hiddenVals[0] * weights2[i*4];
            value += hiddenVals[1] * weights2[i*4+1];
            value += hiddenVals[2] * weights2[i*4+2];
            value += hiddenVals[3] * weights2[i*4+3];
            value += prevHiddenVals2[0] * prevHiddenWeights2[i*2];
            value += prevHiddenVals2[1] * prevHiddenWeights2[i*2+1];
            value += bias;
            hiddenVals2[i] = hidden_activate(value);
        }

        for(i32 i = 0; i < 2; ++i) {
            f64 value = 0.0;
            value += hiddenVals2[0] * outWeights[i*2];
            value += hiddenVals2[1] * outWeights[i*2+1];
            value += bias;
            output[i] = output_activate(value);
        }

        memmove(prevHiddenVals, hiddenVals, sizeof(hiddenVals));
        memmove(prevHiddenVals2, hiddenVals2, sizeof(hiddenVals2));
    }

    for(i32 p = 0; p < PASSES; ++p) {
        rnnPropagate(&nn, 1, &def);
    }

    for(i32 i = 0; i < 4; ++i) {
        assert(fabs(nn->values[2 + i] - hiddenVals[i]) < 0.0001);
    }
    for(i32 i = 0; i < 2; ++i) {
        assert(fabs(nn->values[6 + i] - hiddenVals2[i]) < 0.0001);
    }
    assert(fabs(nn->output[0] - output[0]) < 0.0001);
    assert(fabs(nn->output[1] - output[1]) < 0.0001);

    rnnDealloc(nn);
}

void testPropagateRNNWide()
{
    const i32 PASSES = 3;
    RecurrentNeuralNetDef def;
    const i32 layers[] = {2, 4, 6, 2};
    rnnMakeDef(&def, array_count(layers), layers, 1.0);

    RecurrentNeuralNet* nn[2];
    rnnAlloc(nn, 2, &def);

    assert((intptr_t)nn[1] - (intptr_t)nn[0] == def.neuralNetSize);

    rnnInitRandom(&nn[0], 1, &def);
    nn[0]->values[0] = randf64(0, 5.0);
    nn[0]->values[1] = randf64(0, 5.0);

    rnnCopy(nn[1], nn[0], &def);

    for(i32 p = 0; p < PASSES; ++p) {
        rnnPropagate(&nn[0], 1, &def);
        rnnPropagateWide(&nn[1], 1, &def);
    }

    for(i32 i = 0; i < def.neuronCount; ++i) {
        LOG("val[%d] = %.6f val2[%d] = %.6f", i, nn[0]->values[i], i, nn[1]->values[i]);
        assert(fabs(nn[0]->values[i] - nn[1]->values[i]) < 0.01);
    }

    rnnDealloc(nn[0]);
}

void testWideTanh()
{
    constexpr i32 TEST_COUNT = 16;
    f64 input[TEST_COUNT];
    w128d winput[TEST_COUNT/2];

    for(i32 i = 0; i < TEST_COUNT; ++i) {
        input[i] = randf64(-5.0, 5.0);
    }

    memmove(winput, input, sizeof(winput));

    for(i32 i = 0; i < TEST_COUNT; ++i) {
        input[i] = tanh(input[i]);
    }

    for(i32 i = 0; i < TEST_COUNT/2; ++i) {
        winput[i] = wide_f64_tanh(winput[i]);
    }

    f64* notWinput = (f64*)winput;

    for(i32 i = 0; i < TEST_COUNT; ++i) {
        LOG("val[%d] = %.6f val2[%d] = %.6f", i, input[i], i, notWinput[i]);
        assert(fabs(input[i] - notWinput[i]) < 0.01);
    }
}

void generateSpeciesTags(u8* tags, const i32 tagCount, const i32 bitCount)
{
    for(i32 i = 0; i < tagCount; ++i) {
        tags[i] = randi64(0, (1 << bitCount)-1);
    }
}

void evolutionSSS1(GeneticEnvRnn* env)
{
    assert(env->speciesBits <= 8);
    const i32 POP_COUNT = env->populationCount;
    const i32 SUBPOP_MAX_COUNT = (1 << env->speciesBits);

    f64* indivFitness = env->fitness;
    RecurrentNeuralNet** curPopNN = env->curPopRNN;
    RecurrentNeuralNet** nextPopNN = env->nextPopRNN;
    u8* curPopTag = env->curSpeciesTags;
    u8* nextPopTag = env->nextSpeciesTags;
    const i32 weightTotalCount = env->rnnDef->weightTotalCount;
    const i32 neuralNetSize = env->rnnDef->neuralNetSize;

    i32* subPopIndivCount = stack_arr(i32,SUBPOP_MAX_COUNT);
    memset(subPopIndivCount, 0, SUBPOP_MAX_COUNT * sizeof(i32));
    f64* normFitness = stack_arr(f64,POP_COUNT);
    const i32 parentCount = POP_COUNT * 0.3;
    i32* parents = stack_arr(i32,parentCount);

    for(i32 i = 0; i < POP_COUNT; ++i) {
        u8 tag = curPopTag[i];
        subPopIndivCount[tag]++;
    }

    // fitness sharing
    f64 totalFitness = 0.0;
    for(i32 i = 0; i < POP_COUNT; ++i) {
        normFitness[i] = indivFitness[i] / subPopIndivCount[curPopTag[i]];
        totalFitness += normFitness[i];
    }

    // select parents
    i32 rouletteMisses = 0;
    u8* chosenAsParent = stack_arr(u8,POP_COUNT);
    memset(chosenAsParent, 0, sizeof(u8)*POP_COUNT);
    for(i32 p = 0; p < parentCount; ++p) {
        i32 tries = 200;
        bool found = false;
        while(tries-- && !found) {
            i32 s = selectRoulette(POP_COUNT, normFitness, totalFitness);
            if(!chosenAsParent[s]) {
                parents[p] = s;
                chosenAsParent[s] = 1;
                found = true;
            }
        }
        if(!found) {
            parents[p] = selectRoulette(POP_COUNT, normFitness, totalFitness);
            rouletteMisses++;
        }
    }

    i32 noMatesFoundCount = 0;
    RecurrentNeuralNet** potentialMates = stack_arr(RecurrentNeuralNet*,parentCount);
    f64* pmFitness = stack_arr(f64,parentCount);

    for(i32 i = 0; i < POP_COUNT; ++i) {
        const i32 parentA = randi64(0, parentCount-1);
        const i32 idA = parents[parentA];
        const u8 tagA = curPopTag[idA];

        // find same sub pop mates
        i32 potentialMatesCount = 0;
        i32 pmTotalFitness = 0.0;
        for(i32 j = 0; j < parentCount; ++j) {
            i32 idB = parents[j];
            if(parentA != j && tagA == curPopTag[idB]) {
                i32 pmId = potentialMatesCount++;
                potentialMates[pmId] = curPopNN[idB];
                pmFitness[pmId] = normFitness[idB];
                pmTotalFitness += pmFitness[pmId];
            }
        }

        if(potentialMatesCount < 1) {
            noMatesFoundCount++;
            memmove(nextPopNN[i], curPopNN[idA], neuralNetSize);
            nextPopTag[i] = tagA;
        }
        else {
            RecurrentNeuralNet* mateA = curPopNN[idA];
            i32 mateBId = selectRoulette(potentialMatesCount, pmFitness, pmTotalFitness);
            RecurrentNeuralNet* mateB = potentialMates[mateBId];
            crossover(nextPopNN[i]->weights, mateA->weights, mateB->weights, weightTotalCount);
            nextPopTag[i] = tagA;
        }
    }

    LOG("noMatesFoundCount=%d rouletteMisses=%d", noMatesFoundCount, rouletteMisses);

    // mutate
    const f32 mutationRate = 0.005f;
    i32 mutationCount = 0;
    for(i32 i = 0; i < POP_COUNT; ++i) {
        f64 mutationFactor = ((f64)i/POP_COUNT) * 0.5;
        for(i32 s = 0; s < weightTotalCount; ++s) {
            if(randf64(0.0, 1.0) < mutationRate) {
                mutationCount++;
                nextPopNN[i]->weights[s] += randf64(-mutationFactor, mutationFactor);
            }
        }
    }

    // diffuse
    const f32 mutationTagRate = 0.005f;
    i32 tagMutations = 0;
    /*for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(randf64(0.0, 1.0) < mutationTagRate) {
            tagMutations++;
            i32 bit = randi64(0, BIRD_TAG_BITS-1);
            nextSpeciesTag[i] ^= 1 << bit;
        }
    }*/

    LOG("mutationCount=%d tagMutations=%d", mutationCount, tagMutations);

    memmove(curPopNN[0], nextPopNN[0], neuralNetSize * POP_COUNT);
    memmove(curPopTag, nextPopTag, sizeof(curPopTag[0]) * POP_COUNT);
}

void ImGui_NeuralNet(NeuralNet* nn, NeuralNetDef* def)
{
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    if (window->SkipItems)
        return;

    constexpr i32 cellsPerLine = 10;
    const ImVec2 cellSize(10, 10);
    i32 lines = def->neuronCount / cellsPerLine + 1;
    ImVec2 size(cellsPerLine * cellSize.x, lines * cellSize.y);

    ImVec2 pos = window->DC.CursorPos;
    const ImRect bb(pos, pos + size);
    ImGui::ItemSize(bb);

    for(i32 i = 0; i < def->neuronCount; ++i) {
        f32 w = clamp(nn->values[i] * 0.5, 0.0, 1.0);
        u32 color = 0xff000000 | ((u8)(0xff*w) << 16)| ((u8)(0xff*w) << 8)| ((u8)(0xff*w));
        i32 column = i % cellsPerLine;
        i32 line = i / cellsPerLine;
        ImVec2 offset(column * cellSize.x, line * cellSize.y);
        ImGui::RenderFrame(pos + offset, pos + offset + cellSize, color, false, 0);
    }
}

void ImGui_RecurrentNeuralNet(RecurrentNeuralNet* nn, RecurrentNeuralNetDef* def)
{
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    if (window->SkipItems)
        return;

    constexpr i32 cellsPerLine = 14;
    const ImVec2 cellSize(10, 10);
    i32 lines = def->neuronCount / cellsPerLine + 1;
    ImVec2 size(cellsPerLine * cellSize.x, lines * cellSize.y);

    ImVec2 pos = window->DC.CursorPos;
    const ImRect bb(pos, pos + size);
    ImGui::ItemSize(bb);

    for(i32 i = 0; i < def->neuronCount; ++i) {
        i32 isNormalVal = i < (def->neuronCount - def->hiddenStateNeuronCount);
        f32 w = clamp(nn->values[i] * 0.5, 0.0, 1.0);
        u32 color = 0xff000000 | ((u8)(0xff*w) << 16)| ((u8)(0xff*w*isNormalVal) << 8)| ((u8)(0xff*w));
        i32 column = i % cellsPerLine;
        i32 line = i / cellsPerLine;
        ImVec2 offset(column * cellSize.x, line * cellSize.y);
        ImGui::RenderFrame(pos + offset, pos + offset + cellSize, color, false, 0);
    }
}

void ImGui_SubPopWindow(const GeneticEnvRnn* env, const ImVec4* subPopColors)
{
    const i32 POP_COUNT = env->populationCount;
    const i32 SUBPOP_MAX_COUNT = 1 << env->speciesBits;
    const u8* curSpeciesTag = env->curSpeciesTags;
    const f64* fitness = env->fitness;

    f64* totalFitness = stack_arr(f64,SUBPOP_MAX_COUNT);
    f64* maxFitness = stack_arr(f64,SUBPOP_MAX_COUNT);
    f64* avgFitness = stack_arr(f64,SUBPOP_MAX_COUNT);
    i32* subPopIndivCount = stack_arr(i32,SUBPOP_MAX_COUNT);
    zero_arr(maxFitness,SUBPOP_MAX_COUNT);
    zero_arr(totalFitness,SUBPOP_MAX_COUNT);
    zero_arr(avgFitness,SUBPOP_MAX_COUNT);
    zero_arr(subPopIndivCount,SUBPOP_MAX_COUNT);
    f64 maxTotal = 0;
    f64 maxMaxFitness = 0;
    f64 maxAvg = 0;
    i32 maxCount = 0;

    for(i32 i = 0; i < POP_COUNT; ++i) {
        maxFitness[curSpeciesTag[i]] = max(fitness[i], maxFitness[curSpeciesTag[i]]);
        totalFitness[curSpeciesTag[i]] += fitness[i];
        subPopIndivCount[curSpeciesTag[i]]++;
    }
    for(i32 i = 0; i < SUBPOP_MAX_COUNT; ++i) {
        maxTotal = max(totalFitness[i], maxTotal);
        maxMaxFitness = max(maxFitness[i], maxMaxFitness);
        avgFitness[i] = totalFitness[i]/subPopIndivCount[i];
        maxAvg = max(avgFitness[i], maxAvg);
        maxCount = max(subPopIndivCount[i], maxCount);
    }

    ImGui::Begin("Sub populations");

    if(ImGui::CollapsingHeader("Population count")) {
        for(i32 i = 0; i < SUBPOP_MAX_COUNT; ++i) {
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, subPopColors[i]);
            char buff[64];
            sprintf(buff, "%d", subPopIndivCount[i]);
            ImGui::ProgressBar(subPopIndivCount[i]/(f32)maxCount, ImVec2(-1,0), buff);
            ImGui::PopStyleColor(1);
        }
    }

    if(ImGui::CollapsingHeader("Total fitness")) {
        for(i32 i = 0; i < SUBPOP_MAX_COUNT; ++i) {
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, subPopColors[i]);
            ImGui::ProgressBar(totalFitness[i]/maxTotal);
            ImGui::PopStyleColor(1);
        }
    }

    if(ImGui::CollapsingHeader("Average fitness")) {
        for(i32 i = 0; i < SUBPOP_MAX_COUNT; ++i) {
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, subPopColors[i]);
            ImGui::ProgressBar(avgFitness[i]/maxAvg);
            ImGui::PopStyleColor(1);
        }
    }

    if(ImGui::CollapsingHeader("Max fitness")) {
        for(i32 i = 0; i < SUBPOP_MAX_COUNT; ++i) {
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, subPopColors[i]);
            ImGui::ProgressBar(maxFitness[i]/maxMaxFitness);
            ImGui::PopStyleColor(1);
        }
    }

    ImGui::End();
}
