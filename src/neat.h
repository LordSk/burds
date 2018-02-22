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

struct NodePos
{
    i8 layer;
    i8 vpos;
};

struct Genome
{
    Gene genes[NEAT_MAX_GENES];
    u8 geneDisabled[NEAT_MAX_GENES];
    u8 layerNodeCount[NEAT_MAX_LAYERS];
    NodePos nodePos[NEAT_MAX_NODES];
    i32 layerCount = 0;
    i32 geneCount = 0;
    i32 inputNodeCount = 0;  // TODO: this is constant, no need to store it in EVERY genome
    i32 outputNodeCount = 0; // same here
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

void neatGenomeAlloc(Genome** genomes, const i32 count);
void neatGenomeDealloc(void* ptr);

void neatGenomeInit(Genome** genomes, const i32 count, i32 inputCount, i32 outputCount);
void neatGenomeMakeNN(Genome** genomes, const i32 count, NeatNN** nn);
void neatNnDealloc(void* ptr);

void neatEvolve(Genome** genomes, Genome** nextGenomes, f64* fitness, const i32 popCount);

void neatTestTryReproduce(const Genome& g1, const Genome& g2);
void neatTestCrossover(const Genome* parentA, const Genome* parentB, Genome* dest);
