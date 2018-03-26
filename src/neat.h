#pragma once
#include "base.h"

struct Gene
{
    i32 historicalMarker;
    i16 nodeIn;
    i16 nodeOut;
    f64 weight;
};

#define NEAT_MAX_GENES 512
#define NEAT_MAX_NODES 128
#define NEAT_MAX_LAYERS 64
#define NEAT_MAX_SPECIES 1024

struct NodePos
{
    i8 layer;
    u8 vpos;
};

struct Genome
{
    Gene genes[NEAT_MAX_GENES];
    u8 geneDisabled[NEAT_MAX_GENES];
    u8 layerNodeCount[NEAT_MAX_LAYERS];
    NodePos nodePos[NEAT_MAX_NODES];
    i32 nodeOriginMarker[NEAT_MAX_NODES];
    i32 layerCount = 0;
    i32 geneCount = 0;
    i16 inputNodeCount = 0;  // TODO: this is constant, no need to store it in EVERY genome
    i16 outputNodeCount = 0; // same here
    i32 totalNodeCount = 0;
    i32 species;
};

struct NeatNN
{
    struct Computation {
        i16 nodeIn, nodeOut;
        f64 weight;
    };

    f64* nodeValues;
    Computation* computations; // sorted
    i32 computationsCount;
};

struct NeatEvolutionParams
{
    f64 compC1 = 1.0; // compatibility distance DISJOINT factor
    f64 compC2 = 1.0; // compatibility distance EXCESS factor
    f64 compC3 = 0.4; // compatibility distance WEIGHT factor
    f64 compT  = 0.6; // compatibility threshold
    f64 crossoverKeepDisabled = 0.75; // crossover chance to keep a gene disabled
    f64 mutateWeight = 0.8; // mutation chance to alter a connection weight
    f64 mutateResetWeight = 0.1; // mutation chance ton reset a weight if chosen for alteration
    f64 mutateAddConn = 0.05; // mutation chance to add a connection between 2 nodes
    f64 mutateAddNode = 0.03; // mutation chance to split an existing connection to add a node in between
};

struct NeatSpeciesStagnation
{
    u16 stagnation[NEAT_MAX_SPECIES] = {0};
    f64 maxFitness[NEAT_MAX_SPECIES] = {0};
};

void neatGenomeAlloc(Genome** genomes, const i32 count);
void neatGenomeDealloc(void* ptr);

void neatGenomeInit(Genome** genomes, const i32 count, i32 inputCount, i32 outputCount);
void neatGenomeMakeNN(Genome** genomes, const i32 count, NeatNN** nn, bool verbose = false);
void neatNnPropagate(NeatNN** nn, const i32 nnCount);
void neatNnDealloc(void* ptr);

void neatEvolve(Genome** genomes, Genome** nextGenomes, f64* fitness, const i32 popCount,
                NeatSpeciesStagnation* speciesStagn, const NeatEvolutionParams& params,
                bool verbose = false);

void neatTestTryReproduce(const Genome& g1, const Genome& g2);
void neatTestCrossover(const Genome* parentA, const Genome* parentB, Genome* dest);
f64 neatTestCompability(const Genome* ga, const Genome* gb, const NeatEvolutionParams& params);
