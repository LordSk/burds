#pragma once
#include "base.h"
#include "neural.h"

struct Gene
{
    i32 innovationNumber;
    i16 nodeIn;
    i16 nodeOut;
    i32 disabled;
    f64 weight;
};

#define MAX_GENES 512

struct Genome
{
    Gene genes[MAX_GENES];
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
