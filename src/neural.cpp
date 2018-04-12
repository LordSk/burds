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

RnnSpeciation::~RnnSpeciation()
{
    if(speciesRep) {
        rnnDealloc(speciesRep);
    }
}

#define sigmoid(val) (1.0 / (1.0 + expf(-val)))
//#define hidden_activate(val) min(max(0, val), 10.0)
#define hidden_activate(val) tanh(clamp(val, -10.0, 10.0))
#define output_activate(val) tanh(clamp(val, -10.0, 10.0))
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

void rnnAlloc(RecurrentNeuralNet** nn, const i32 nnCount, const RecurrentNeuralNetDef& def)
{
    i32 dataSize = nnCount * def.neuralNetSize;
    u8* data = (u8*)_aligned_malloc(dataSize, alignof(RecurrentNeuralNet));

    for(i32 i = 0; i < nnCount; ++i) {
        nn[i] = (RecurrentNeuralNet*)(data + def.neuralNetSize * i);
#ifdef CONF_DEBUG
        memset(nn[i], 0xAB+i, def.neuralNetSize);
#endif
        nn[i]->values = (f64*)(nn[i] + 1);
        assert((intptr_t)nn[i]->values - (intptr_t)nn[i] == sizeof(RecurrentNeuralNet));
        nn[i]->weights = nn[i]->values + def.neuronCount;
        nn[i]->prevHiddenValues = nn[i]->values + def.neuronCount - def.hiddenStateNeuronCount;
        nn[i]->prevHiddenWeights = nn[i]->weights + def.weightTotalCount - def.hiddenStateWeightCount;
        nn[i]->output = nn[i]->prevHiddenValues - def.outputNeuronCount;
    }

    LOG("allocated %d RNN (layers=%d nnSize=%d totalDataSize=%d)", nnCount, def.layerCount,
        def.neuralNetSize, dataSize);
}


void rnnDealloc(RecurrentNeuralNet** nn)
{
    _aligned_free(nn[0]);
}


void rnnCopy(RecurrentNeuralNet* dest, RecurrentNeuralNet* src, const RecurrentNeuralNetDef& def)
{
    const i32 weightTotalCount = def.weightTotalCount;
    const i32 neuronCount = def.neuronCount;

    for(i32 i = 0; i < weightTotalCount; ++i) {
        dest->weights[i] = src->weights[i];
    }
    for(i32 i = 0; i < neuronCount; ++i) {
        dest->values[i] = src->values[i];
    }
}

static f64 compatibilityDistance(const f64* weightA, const f64* weightB, const i32 weightCount)
{
    f64 totalWeightDiff = 0.0;
    for(i32 i = 0; i < weightCount; ++i) {
        totalWeightDiff += fabs(weightA[i] - weightB[i]);
    }

    f64 avgWeightDiff = totalWeightDiff / weightCount;
    return avgWeightDiff;
}

void rnnInit(RecurrentNeuralNet** nn, const i32 popCount, const RecurrentNeuralNetDef& def)
{
    const i32 weightTotalCount = def.weightTotalCount;
    const i32 neuronCount = def.neuronCount;
    for(i32 i = 0; i < popCount; ++i) {
        memset(nn[i]->values, 0, sizeof(nn[i]->values[0]) * neuronCount);
        for(i32 s = 0; s < weightTotalCount; ++s) {
            nn[i]->weights[s] = randf64(-1.0, 1.0);
        }
    }
}

void rnnSpeciationInit(RnnSpeciation* speciation, i32* species, RecurrentNeuralNet** nn,
                       const i32 popCount, const RecurrentNeuralNetDef& rnnDef)
{
    // reset speciation first
    const f64 compT = speciation->compT;
    *speciation = {};
    speciation->compT = compT;

    rnnAlloc(speciation->speciesRep, RNN_MAX_SPECIES, rnnDef);
    mem_zero(speciation->speciesPopCount);

    RecurrentNeuralNet** speciesRep = speciation->speciesRep;
    i32* speciesPopCount = speciation->speciesPopCount;
    i32 speciesCount = 0;
    f64 biggestDist = 0.0;

    const i32 weightTotalCount = rnnDef.weightTotalCount;

    for(i32 i = 0; i < popCount; ++i) {
        RecurrentNeuralNet* nni = nn[i];

        bool found = false;
        for(i32 s = 0; s < speciesCount; ++s) {
            if(speciesPopCount[s] == 0) continue;

            f64 dist = compatibilityDistance(speciesRep[s]->weights, nni->weights, weightTotalCount);
            biggestDist = max(dist, biggestDist);
            if(dist < compT) {
                species[i] = s;
                speciesPopCount[s]++;
                found = true;
                break;
            }
        }

        if(!found) {
            assert(speciesCount < RNN_MAX_SPECIES);
            i32 sid = speciesCount++;
            rnnCopy(speciesRep[sid], nni, rnnDef);
            speciesPopCount[sid] = 1;
            species[i] = sid;
        }
    }

    LOG("initial speciesCount: %d", speciesCount);
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
                f64 value = bias; // bias
                // prevLayervals * prevLayerWeights
                for(i32 s = 0; s < prevNeuronCount; ++s) {
                    value += weights[s] * prevLayerVals[s];
                }
                // prevSate * prevSateWeights
                for(i32 s = 0; s < hiddenNeuronCount; ++s) {
                    value += prevHiddenWeights[s] * prevHiddenValues[s];
                }
                value = hidden_activate(value);
                assert(value == value); // nan check
                hiddenStateVals[n] = value;

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
            f64 value = bias; // bias
            // hiddenState * outputWeights
            for(i32 s = 0; s < prevNeuronCount; ++s) {
                value += weights[s] * lastHiddenVals[s];
            }
            value = output_activate(value);
            assert(value == value);  // nan check
            output[n] = value;

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
    i32 species;
    f64 fitness;
};

static i32 compareFitnessDesc(const void* a, const void* b)
{
    const FitnessPair* fa = (FitnessPair*)a;
    const FitnessPair* fb = (FitnessPair*)b;
    if(fa->species > fb->species) return -1;
    if(fa->species < fb->species) return 1;
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
    //assert(totalFitness > 0);
    f64 r = randf64(0.0, totalFitness);
    f64 s = 0.0;
    for(i32 j = 0; j < count; ++j) {
        s += fitness[j];
        if(r < s) {
            return j;
        }
    }
    assert(0);
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

    qsort(list, nnCount, sizeof(FitnessPair), compareFitnessDesc);

    for(i32 i = 0; i < maxBest; ++i) {
        memmove(nextGen[i], curGen[list[i].id], def->neuralNetSize);
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

inline i32 selectRandom(const i32 count, i32 notThisId)
{
    i32 r = randi64(0, count);
    while(r == notThisId) {
        r = randi64(0, count);
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
    rnnMakeDef(&def, arr_count(layers), layers, 1.0);

    RecurrentNeuralNet* nn;
    rnnAlloc(&nn, 1, def);

    assert(def.neuronCount == 10 + 6);
    assert(def.weightTotalCount == (4 * 2 + 4 * 4 + 4 * 2 + 2 * 2 + 2 * 2));
    memmove(nn->values, inputs, sizeof(inputs));

    f64* nnWeights = nn->weights;
    memmove(nnWeights, weights1, sizeof(weights1));
    nnWeights += arr_count(weights1);
    memmove(nnWeights, weights2, sizeof(weights2));
    nnWeights += arr_count(weights2);
    memmove(nnWeights, outWeights, sizeof(outWeights));

    memmove(nn->prevHiddenValues, prevHiddenVals, sizeof(prevHiddenVals));
    memmove(nn->prevHiddenValues + arr_count(prevHiddenVals), prevHiddenVals2, sizeof(prevHiddenVals2));
    memmove(nn->prevHiddenWeights, prevHiddenWeights, sizeof(prevHiddenWeights));
    memmove(nn->prevHiddenWeights + arr_count(prevHiddenWeights),
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

    rnnDealloc(&nn);
}

void testPropagateRNNWide()
{
    const i32 PASSES = 3;
    RecurrentNeuralNetDef def;
    const i32 layers[] = {2, 4, 6, 2};
    rnnMakeDef(&def, arr_count(layers), layers, 1.0);

    RecurrentNeuralNet* nn[2];
    rnnAlloc(nn, 2, def);

    assert((intptr_t)nn[1] - (intptr_t)nn[0] == def.neuralNetSize);

    rnnInit(&nn[0], 1, def);
    nn[0]->values[0] = randf64(0, 5.0);
    nn[0]->values[1] = randf64(0, 5.0);

    rnnCopy(nn[1], nn[0], def);

    for(i32 p = 0; p < PASSES; ++p) {
        rnnPropagate(&nn[0], 1, &def);
        rnnPropagateWide(&nn[1], 1, &def);
    }

    for(i32 i = 0; i < def.neuronCount; ++i) {
        LOG("val[%d] = %.6f val2[%d] = %.6f", i, nn[0]->values[i], i, nn[1]->values[i]);
        assert(fabs(nn[0]->values[i] - nn[1]->values[i]) < 0.01);
    }

    rnnDealloc(nn);
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

void rnnEvolve(RnnGeneticEnv* env, bool verbose)
{
    const i32 popCount = env->popCount;
    RnnSpeciation& speciation = *env->speciation;
    assert(speciation.speciesRep[0]);

    f64* fitness = env->fitness;
    RecurrentNeuralNet** curGenNN = env->curGenRNN;
    RecurrentNeuralNet** nextGenNN = env->nextGenRNN;
    i32* curGenSpecies = env->curGenSpecies;
    i32* nextGenSpecies = env->nextGenSpecies;
    const RecurrentNeuralNetDef& rnnDef = *env->rnnDef;
    const i32 weightTotalCount = rnnDef.weightTotalCount;
    const i32 hiddenStateNeuronCount = rnnDef.hiddenStateNeuronCount;

    f64 speciesMaxFitness[RNN_MAX_SPECIES] = {0};

    for(i32 i = 0; i < popCount; ++i) {
        const i32 s = curGenSpecies[i];
        assert(s >= 0 && s < RNN_MAX_SPECIES);
        speciesMaxFitness[s] = max(fitness[i], speciesMaxFitness[s]);
    }

    // species stagnation
    u8* deleteSpecies = stack_arr(u8,RNN_MAX_SPECIES);
    const i32 stagnationT = 15;
    u16* specStagnation = speciation.stagnation;
    f64* specStagMaxFitness = speciation.maxFitness;

    for(i32 s = 0; s < RNN_MAX_SPECIES; ++s) {
        deleteSpecies[s] = false;
        if(speciesMaxFitness[s] <= specStagMaxFitness[s]) {
            specStagnation[s]++;

            if(specStagnation[s] > stagnationT) {
                if(verbose) LOG("NEAT> species %x stagnating (%d)", s, specStagnation[s]);
                deleteSpecies[s] = true;
                specStagnation[s] = 0;
                specStagMaxFitness[s] = 0.0;
            }
        }
        else {
            specStagMaxFitness[s] = speciesMaxFitness[s];
            specStagnation[s] = 0;
        }
    }

    // keep best species always
    f64 bestMaxFitness = 0.0;
    i32 bestSpecies = -1;
    for(i32 s = 0; s < RNN_MAX_SPECIES; ++s) {
        if(speciesMaxFitness[s] > bestMaxFitness) {
            bestMaxFitness = speciesMaxFitness[s];
            bestSpecies = s;
        }
    }
    assert(bestSpecies != -1);
    deleteSpecies[bestSpecies] = false;
    specStagnation[bestSpecies] = 0;

    FitnessPair* fpair = stack_arr(FitnessPair,popCount);
    memset(fpair, 0, sizeof(FitnessPair) * popCount);
    for(i32 i = 0; i < popCount; ++i) {
        fpair[i] = { i, curGenSpecies[i], fitness[i] };
    }
    qsort(fpair, popCount, sizeof(FitnessPair), compareFitnessDesc);

    // fitness sharing
    i32* speciesPopCount = speciation.speciesPopCount;
    f64* normFitness = stack_arr(f64,popCount);
    for(i32 i = 0; i < popCount; ++i) {
        normFitness[i] = fitness[i] * 10000.0 / speciesPopCount[curGenSpecies[i]];
    }

    // parents
    i32* speciesParentCount = stack_arr(i32,RNN_MAX_SPECIES);
    memset(speciesParentCount, 0, RNN_MAX_SPECIES * sizeof(i32));
    f64* parentFitness = stack_arr(f64,popCount);
    f64 parentTotalFitness = 0.0;
    i32 parentCount = 0;

    for(i32 i = 0; i < popCount; ++i) {
        const i32 id = fpair[i].id;
        const i32 species = fpair[i].species;
        if(deleteSpecies[species]) continue;

        if(speciesParentCount[species] < max(speciesPopCount[species] / 2, 1)) {
            speciesParentCount[species]++;
            const i32 pid = parentCount++;
            rnnCopy(nextGenNN[pid], curGenNN[id], rnnDef);
            nextGenSpecies[pid] = species;
            parentFitness[pid] = normFitness[id];
            parentTotalFitness += parentFitness[pid];
        }
    }

    // move parent to current pop array
    for(i32 i = 0; i < parentCount; ++i) {
        rnnCopy(curGenNN[i], nextGenNN[i], rnnDef);
    }
    memmove(curGenSpecies, nextGenSpecies, sizeof(curGenSpecies[0]) * parentCount);

    // copy champion of each species unchanged
    i32 championCount = 0;
    i32 champCheckSpec = -1;
    for(i32 i = 0; i < parentCount; ++i) {
        const i32 spec = curGenSpecies[i];
        if(spec != champCheckSpec && speciesPopCount[spec] > 4) {
            rnnCopy(nextGenNN[popCount - 1 - (championCount++)], curGenNN[i], rnnDef);
            champCheckSpec = spec;
        }
    }

    const i32 popCountMinusChamps = popCount - championCount;

    i32 noMatesFoundCount = 0;
    RecurrentNeuralNet** potentialMates = stack_arr(RecurrentNeuralNet*,parentCount);
    f64* pmFitness = stack_arr(f64,parentCount);

    for(i32 i = 0; i < popCountMinusChamps; ++i) {
        const i32 idA = selectRoulette(parentCount, parentFitness, parentTotalFitness);
        const i32 speciesA = curGenSpecies[idA];

        // copy 25% (no crossover)
        if(randf64(0.0, 1.0) < 0.25) {
            rnnCopy(nextGenNN[i], curGenNN[idA], rnnDef);
            nextGenSpecies[i] = speciesA;
            continue;
        }

        // find same sub pop mates
        i32 potentialMatesCount = 0;
        i32 pmTotalFitness = 0.0;
        for(i32 j = 0; j < parentCount; ++j) {
            const i32 idB = j;
            if(idA != idB && speciesA == curGenSpecies[idB]) {
                i32 pmId = potentialMatesCount++;
                potentialMates[pmId] = curGenNN[idB];
                pmFitness[pmId] = parentFitness[idB];
                pmTotalFitness += pmFitness[pmId];
            }
        }

        if(potentialMatesCount < 1) {
            noMatesFoundCount++;
            rnnCopy(nextGenNN[i], curGenNN[idA], rnnDef);
            nextGenSpecies[i] = speciesA;
        }
        else {
            RecurrentNeuralNet* mateA = curGenNN[idA];
            i32 mateBId = selectRoulette(potentialMatesCount, pmFitness, pmTotalFitness);
            RecurrentNeuralNet* mateB = potentialMates[mateBId];

            // A is the fittest
            if(parentFitness[idA] < parentFitness[mateBId]) {
                RecurrentNeuralNet* tmp = mateA;
                mateA = mateB;
                mateB = tmp;
            }

            crossover(nextGenNN[i]->weights, mateA->weights, mateB->weights, weightTotalCount);
            nextGenSpecies[i] = speciesA;
        }
    }

    LOG("noMatesFoundCount=%d", noMatesFoundCount);

    // mutate
    const f64 mutationRate = 0.05;
    const f64 mutationStep = 0.5;
    const f64 mutationResetWeight = 0.10;

    i32 mutationCount = 0;
    for(i32 i = 0; i < popCountMinusChamps; ++i) {
        for(i32 s = 0; s < weightTotalCount; ++s) {
            if(randf64(0.0, 1.0) < mutationRate) {
                mutationCount++;

                if(randf64(0.0, 1.0) < mutationResetWeight) {
                    nextGenNN[i]->weights[s] = randf64(-1.0, 1.0);
                }
                else {
                    nextGenNN[i]->weights[s] += randf64(-mutationStep, mutationStep);
                }
            }
        }
    }

    LOG("mutationCount=%d", mutationCount);

    for(i32 i = 0; i < popCount; ++i) {
        rnnCopy(curGenNN[i], nextGenNN[i], rnnDef);
    }
    memmove(curGenSpecies, nextGenSpecies, sizeof(curGenSpecies[0]) * popCount);

    for(i32 i = 0; i < popCount; ++i) {
        memset(curGenNN[i]->prevHiddenValues, 0,
               sizeof(curGenNN[i]->prevHiddenValues[0]) * hiddenStateNeuronCount);
    }

    // speciation
    u8 speciesPrevExisted[RNN_MAX_SPECIES] = {0};
    for(i32 s = 0; s < RNN_MAX_SPECIES; ++s) {
        speciesPrevExisted[s] = (speciesPopCount[s] != 0);
    }

    RecurrentNeuralNet** speciesRep = speciation.speciesRep;
    mem_zero(speciation.speciesPopCount); // reset species population count
    speciesPopCount = speciation.speciesPopCount;
    f64 biggestDist = 0.0;

    const f64 compT = speciation.compT;

    for(i32 i = 0; i < popCount; ++i) {
        RecurrentNeuralNet* nni = curGenNN[i];

        bool found = false;
        for(i32 s = 0; s < RNN_MAX_SPECIES; ++s) {
            if(!speciesPrevExisted[s] && speciesPopCount[s] == 0) continue;

            f64 dist = compatibilityDistance(speciesRep[s]->weights, nni->weights, weightTotalCount);
            biggestDist = max(dist, biggestDist);
            if(dist < compT) {
                curGenSpecies[i] = s;
                speciesPopCount[s]++;
                found = true;
                break;
            }
        }

        if(!found) {
            // find a species slot
            i32 sid = -1;
            for(i32 s = 0; s < RNN_MAX_SPECIES; ++s) {
                if(!speciesPrevExisted[s] && speciesPopCount[s] == 0) {
                    sid = s;
                    break;
                }
            }

            rnnCopy(speciesRep[sid], nni, rnnDef);
            speciesPopCount[sid] = 1;
            curGenSpecies[i] = sid;
        }
    }
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

void ImGui_SubPopWindow(const RnnGeneticEnv* env, const ImVec4* subPopColors)
{
    const i32 POP_COUNT = env->popCount;
    const i32 speciesCount = RNN_MAX_SPECIES;
    const i32* curSpeciesTag = env->curGenSpecies;
    const f64* fitness = env->fitness;

    f64* totalFitness = stack_arr(f64,speciesCount);
    f64* maxFitness = stack_arr(f64,speciesCount);
    f64* avgFitness = stack_arr(f64,speciesCount);
    i32* subPopIndivCount = stack_arr(i32,speciesCount);
    arr_zero(maxFitness,speciesCount);
    arr_zero(totalFitness,speciesCount);
    arr_zero(avgFitness,speciesCount);
    arr_zero(subPopIndivCount,speciesCount);
    f64 maxTotal = 0;
    f64 maxMaxFitness = 0;
    f64 maxAvg = 0;
    i32 maxCount = 0;

    for(i32 i = 0; i < POP_COUNT; ++i) {
        maxFitness[curSpeciesTag[i]] = max(fitness[i], maxFitness[curSpeciesTag[i]]);
        totalFitness[curSpeciesTag[i]] += fitness[i];
        subPopIndivCount[curSpeciesTag[i]]++;
    }
    for(i32 i = 0; i < speciesCount; ++i) {
        maxTotal = max(totalFitness[i], maxTotal);
        maxMaxFitness = max(maxFitness[i], maxMaxFitness);
        avgFitness[i] = totalFitness[i]/subPopIndivCount[i];
        maxAvg = max(avgFitness[i], maxAvg);
        maxCount = max(subPopIndivCount[i], maxCount);
    }

    ImGui::Begin("Sub populations");

    if(ImGui::CollapsingHeader("Population count")) {
        for(i32 i = 0; i < speciesCount; ++i) {
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, subPopColors[i]);
            char buff[64];
            sprintf(buff, "%d", subPopIndivCount[i]);
            ImGui::ProgressBar(subPopIndivCount[i]/(f32)maxCount, ImVec2(-1,0), buff);
            ImGui::PopStyleColor(1);
        }
    }

    if(ImGui::CollapsingHeader("Total fitness")) {
        for(i32 i = 0; i < speciesCount; ++i) {
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, subPopColors[i]);
            ImGui::ProgressBar(totalFitness[i]/maxTotal);
            ImGui::PopStyleColor(1);
        }
    }

    if(ImGui::CollapsingHeader("Average fitness")) {
        for(i32 i = 0; i < speciesCount; ++i) {
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, subPopColors[i]);
            ImGui::ProgressBar(avgFitness[i]/maxAvg);
            ImGui::PopStyleColor(1);
        }
    }

    if(ImGui::CollapsingHeader("Max fitness")) {
        for(i32 i = 0; i < speciesCount; ++i) {
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, subPopColors[i]);
            ImGui::ProgressBar(maxFitness[i]/maxMaxFitness);
            ImGui::PopStyleColor(1);
        }
    }

    ImGui::End();
}
