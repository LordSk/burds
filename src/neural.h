#pragma once
#include "base.h"
#include <intrin.h>
#include <emmintrin.h>

#define MAX_LAYERS 10

// wide types
typedef __m128 wf32;
typedef __m128d w128d;

#define wide_f64_zero() _mm_setzero_pd()
#define wide_f64_set1(f) _mm_set1_pd(f)
#define wide_f64_add(wa, wb) _mm_add_pd(wa, wb)
#define wide_f64_hadd(wa, wb) _mm_hadd_pd(wa, wb)
#define wide_f64_mul(wa, wb) _mm_mul_pd(wa, wb)
#define wide_f64_min(wa, wb) _mm_min_pd(wa, wb)
#define wide_f64_max(wa, wb) _mm_max_pd(wa, wb)

struct NeuralNet
{
    f64* values;
    f64* weights;
    f64* output;
};

struct NeuralNetDef
{
    i32 layerCount;
    i32 layerNeuronCount[MAX_LAYERS];
    i32 neuronCount;
    i32 neuralNetSize;
    i32 inputNeuronCount;
    i32 outputNeuronCount;
    i32 weightTotalCount;
    f64 bias;
};

void nnMakeDef(NeuralNetDef* def, const i32 layerCount, const i32 layerNeuronCount[], f64 bias);
u8* nnAlloc(NeuralNet** nn, const i32 nnCount, const NeuralNetDef* def);
void nnInitRandom(NeuralNet** nn, const i32 nnCount, const NeuralNetDef* def);
void nnPropagate(NeuralNet** nn, const i32 nnCount, const NeuralNetDef* def);

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
};

struct RecurrentNeuralNetDef
{
    i32 layerCount;
    i32 layerNeuronCount[MAX_LAYERS];
    i32 neuralNetSize;
    i32 neuronCount;
    i32 inputNeuronCount;
    i32 outputNeuronCount;
    i32 weightTotalCount;
    i32 hiddenStateNeuronCount;
    i32 hiddenStateWeightCount;
    f64 bias;
};

void rnnMakeDef(RecurrentNeuralNetDef* def, const i32 layerCount, const i32 layerNeuronCount[], f64 bias);
void rnnAlloc(RecurrentNeuralNet** nn, const i32 nnCount, const RecurrentNeuralNetDef* def);
void rnnDealloc(void* ptr);
void rnnInitRandom(RecurrentNeuralNet** nn, const i32 nnCount, const RecurrentNeuralNetDef* def);
void rnnPropagate(RecurrentNeuralNet** nn, const i32 nnCount, const RecurrentNeuralNetDef* def);
void rnnPropagateWide(RecurrentNeuralNet** nn, const i32 nnCount, const RecurrentNeuralNetDef* def);

i32 reinsertTruncateNN(i32 maxBest, i32 nnCount, f64* fitness, NeuralNet** nextGen,
                       NeuralNet** curGen, NeuralNetDef* def);
i32 reinsertTruncateRNN(i32 maxBest, i32 nnCount, f64* fitness, RecurrentNeuralNet** nextGen,
                        RecurrentNeuralNet** curGen, RecurrentNeuralNetDef* def);
i32 reinsertTruncateRNNSpecies(i32 maxBest, i32 nnCount, f64* fitness, RecurrentNeuralNet** nextGen,
                               RecurrentNeuralNet** curGen, RecurrentNeuralNetDef* def, u8* curSpecies,
                               u8* nextSpecies);
void crossover(f64* outWeights, f64* parentBWeights,
               f64* parentAWeights, i32 weightCount);
i32 selectTournament(const i32 reinsertCount, const i32 tournamentSize, i32 notThisId, const f64* fitness);
i32 selectTournamentSpecies(const i32 count, i32 tries, i32 notThisId, const f64* fitness,
                            const u8* speciesTags, const u8 thisTag);
i32 mutateNN(f32 rate, f32 factor, i32 nnCount, NeuralNet** nextGen, NeuralNetDef* def);
i32 mutateRNN(f32 rate, f32 factor, i32 nnCount, RecurrentNeuralNet** nextGen, RecurrentNeuralNetDef* def);

void testPropagateNN();
void testPropagateRNN();
void testPropagateRNNWide();

void ImGui_NeuralNet(NeuralNet* nn, NeuralNetDef* def);
void ImGui_RecurrentNeuralNet(RecurrentNeuralNet* nn, RecurrentNeuralNetDef* def);
