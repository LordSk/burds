#pragma once
#include "base.h"
#include <intrin.h>
#include <emmintrin.h>

#define NN_MAX_LAYERS 10
#define RNN_MAX_SPECIES 1024

// wide types
typedef __m128d w128d;

#define wide_f64_zero() _mm_setzero_pd()
#define wide_f64_set1(f) _mm_set1_pd(f)
#define wide_f64_add(wa, wb) _mm_add_pd(wa, wb)
#define wide_f64_hadd(wa, wb) _mm_hadd_pd(wa, wb)
#define wide_f64_sub(wa, wb) _mm_sub_pd(wa, wb)
#define wide_f64_mul(wa, wb) _mm_mul_pd(wa, wb)
#define wide_f64_div(wa, wb) _mm_div_pd(wa, wb)
#define wide_f64_min(wa, wb) _mm_min_pd(wa, wb)
#define wide_f64_max(wa, wb) _mm_max_pd(wa, wb)
#define wide_f64_and(wa, wb) _mm_and_pd(wa, wb)
#define wide_f64_blendv(wa, wb, mask) _mm_blendv_pd(wa, wb, mask)
#define wide_f64_less_than(wa, wb) _mm_cmplt_pd(wa, wb)

inline void outputNormalizeTanh(f64* out, const i32 count)
{
    for(i32 i = 0; i < count; i++) {
        out[i] = (out[i] + 1.0) * 0.5;
    }
}

inline void outputNormalizeReLu(f64* out, const i32 count)
{
    f64 m = 0.0;
    for(i32 i = 0; i < count; i++) {
        m = max(m, out[i]);
    }
    if(m < 1.0) m = 1.0;
    for(i32 i = 0; i < count; i++) {
        out[i] = out[i] / m;
    }
}

struct NeuralNet
{
    f64* values;
    f64* weights;
    f64* output;

    inline void setInputs(f64* inputArr, const i32 count) {
        memmove(values, inputArr, sizeof(values[0]) * count);
    }
};

struct NeuralNetDef
{
    i32 layerCount;
    i32 layerNeuronCount[NN_MAX_LAYERS];
    i32 neuronCount;
    i32 neuralNetSize;
    i32 inputNeuronCount;
    i32 outputNeuronCount;
    i32 weightTotalCount;
    f64 bias;
};

struct NnSpeciation
{
    NeuralNet* speciesRep[RNN_MAX_SPECIES] = {0};
    i32 speciesPopCount[RNN_MAX_SPECIES] = {0};
    u16 stagnation[RNN_MAX_SPECIES] = {0};
    f64 maxFitness[RNN_MAX_SPECIES] = {0};
    f64 compT = 0.7; // compatibility threshold

    ~NnSpeciation();
};

struct NnEvolutionParams
{
    i32 popCount;
    f64* fitness;
    NeuralNet** curGenRNN;
    NeuralNet** nextGenRNN;
    NeuralNetDef* rnnDef;
    i32* curGenSpecies;
    i32* nextGenSpecies;
    NnSpeciation* speciation;
    f64 mutationRate = 2.0;
    f64 mutationStep = 0.5;
    f64 mutationReset = 0.1;
};

void nnMakeDef(NeuralNetDef* def, const i32 layerCount, const i32 layerNeuronCount[], f64 bias);
u8* nnAlloc(NeuralNet** nn, const i32 nnCount, const NeuralNetDef& def);
void nnDealloc(NeuralNet** nn);
void nnCopy(NeuralNet* dest, NeuralNet* src, const NeuralNetDef& def);
void nnInit(NeuralNet** nn, const i32 nnCount, const NeuralNetDef& def);
void nnSpeciationInit(NnSpeciation* speciation, i32* species, NeuralNet** nn,
                      const i32 popCount, const NeuralNetDef& nnDef);

void nnPropagate(NeuralNet** nn, const i32 nnCount, const NeuralNetDef& def);

void nnCrossover(f64* outWeights, f64* parentBWeights, f64* parentAWeights, i32 weightCount);
void nnEvolve(NnEvolutionParams* params, bool verbose = false);

union alignas(w128d) RecurrentNeuralNet
{
    struct {
    f64* values;
    f64* weights;
    f64* prevHiddenValues;
    f64* prevHiddenWeights;
    f64* output;
    };

    struct {
    w128d* values;
    w128d* weights;
    w128d* prevHiddenValues;
    w128d* prevHiddenWeights;
    w128d* output;
    } wide;

    inline void setInputs(f64* inputArr, const i32 count) {
        memmove(values, inputArr, sizeof(values[0]) * count);
    }
};

struct RecurrentNeuralNetDef
{
    i32 layerCount;
    i32 layerNeuronCount[NN_MAX_LAYERS];
    i32 neuralNetSize;
    i32 neuronCount;
    i32 inputNeuronCount;
    i32 outputNeuronCount;
    i32 weightTotalCount;
    i32 hiddenStateNeuronCount;
    i32 hiddenStateWeightCount;
    f64 bias;
};

struct RnnSpeciation
{
    RecurrentNeuralNet* speciesRep[RNN_MAX_SPECIES] = {0};
    i32 speciesPopCount[RNN_MAX_SPECIES] = {0};
    u16 stagnation[RNN_MAX_SPECIES] = {0};
    f64 maxFitness[RNN_MAX_SPECIES] = {0};
    f64 compT = 0.7; // compatibility threshold

    ~RnnSpeciation();
};

struct RnnEvolutionParams
{
    i32 popCount;
    f64* fitness;
    RecurrentNeuralNet** curGenRNN;
    RecurrentNeuralNet** nextGenRNN;
    RecurrentNeuralNetDef* rnnDef;
    i32* curGenSpecies;
    i32* nextGenSpecies;
    RnnSpeciation* speciation;
    f64 mutationRate = 2.0;
    f64 mutationStep = 0.5;
    f64 mutationReset = 0.1;
};

void rnnMakeDef(RecurrentNeuralNetDef* def, const i32 layerCount, const i32 layerNeuronCount[], f64 bias);
void rnnAlloc(RecurrentNeuralNet** nn, const i32 nnCount, const RecurrentNeuralNetDef& def);
void rnnDealloc(RecurrentNeuralNet** nn);
void rnnCopy(RecurrentNeuralNet* dest, RecurrentNeuralNet* src, const RecurrentNeuralNetDef& def);
void rnnInit(RecurrentNeuralNet** nn, const i32 popCount, const RecurrentNeuralNetDef& def);
void rnnSpeciationInit(RnnSpeciation* speciation, i32* species, RecurrentNeuralNet** nn,
                       const i32 popCount, const RecurrentNeuralNetDef& rnnDef);
void rnnPropagate(RecurrentNeuralNet** nn, const i32 nnCount, const RecurrentNeuralNetDef& def);
void rnnPropagateWide(RecurrentNeuralNet** nn, const i32 nnCount, const RecurrentNeuralNetDef& def);

void rnnCrossover(f64* outWeights, f64* parentBWeights, f64* parentAWeights, i32 weightCount);
void rnnEvolve(RnnEvolutionParams* params, bool verbose = false);

void testWideTanh();
void testPropagateNN();
void testPropagateRNN();
void testPropagateRNNWide();

void ImGui_NeuralNet(const NeuralNet* nn, const NeuralNetDef& def);
void ImGui_RecurrentNeuralNet(const RecurrentNeuralNet* nn, const RecurrentNeuralNetDef& def);
void ImGui_SubPopWindow(const RnnEvolutionParams* env, const struct ImVec4* subPopColors);
