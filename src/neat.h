#pragma once
#include "base.h"
#include "neural.h"

struct Gene
{
    i32 innovationNumber;
    i16 nodeIn;
    i16 nodeOut;
    f64 weight;
};

#define MAX_GENES 512
#define MAX_NODES 128

struct Genome
{
    Gene genes[MAX_GENES];
    u8 geneDisabled[MAX_GENES];
    u8 nodeLayer[MAX_NODES];
    i32 layerCount = 0;
    i32 geneCount = 0;
    i32 inputNodeCount = 0;  // TODO: this is constant, no need to store it in EVERY genome
    i32 outputNodeCount = 0; // same here
    i32 totalNodeCount = 0;
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

i32 newInnovationNumber();
void neatGenomeAlloc(Genome** genomes, const i32 count);
void neatGenomeDealloc(void* ptr);

void neatGenomeInit(Genome** genomes, const i32 count, i32 inputCount, i32 outputCount);
void neatGenomeMakeNN(Genome** genomes, const i32 count, NeatNN** nn);
void neatNnDealloc(void* ptr);

void neatEvolve(Genome** genomes, Genome** nextGenomes, f64* fitness, const i32 count);
