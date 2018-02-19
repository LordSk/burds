#include "neat.h"
#include <stdlib.h>
#include <assert.h>

i32 newInnovationNumber()
{
    static i32 number = 1;
    return number++;
}

void neatGenomeAlloc(Genome** genomes, const i32 count)
{
    i64 size = sizeof(Genome) * count;
    u8* block = (u8*)malloc(size);

    for(i32 i = 0; i < count; ++i) {
        genomes[i] = (Genome*)(block + sizeof(Genome) * i);
    }

    LOG("NEAT> allocated %d genomes, size=%lld", count, size);
}

void neatGenomeDealloc(void* ptr)
{
    free(ptr);
}

void neatGenomeInit(Genome** genomes, const i32 count, i32 inputCount, i32 outputCount)
{
    assert(inputCount > 0);
    assert(outputCount > 0);

    for(i32 i = 0; i < count; ++i) {
        Genome& g = *genomes[i];
        g.inputNodeCount = inputCount;
        g.outputNodeCount = outputCount;
        g.geneCount = 0;

        for(i16 in = 0; in < inputCount; ++in) {
            for(i16 out = 0; out < outputCount; ++out) {
                i32 gid = g.geneCount++;
                g.genes[gid] = { gid, in, (i16)(inputCount + out), 0, randf64(-1.0, 1.0) };
            }
        }
    }
}

i32 compareNeatNNComputations(void const* a, void const* b)
{
    const NeatNN::Computation& ca = *(NeatNN::Computation*)a;
    const NeatNN::Computation& cb = *(NeatNN::Computation*)b;
    if(ca.nodeOut < cb.nodeOut) return -1;
    if(ca.nodeOut > cb.nodeOut) return 1;
    if(ca.nodeIn < cb.nodeIn) return -1;
    if(ca.nodeIn > cb.nodeIn) return 1;
    return 0;
}

void neatGenomeMakeNN(Genome** genomes, const i32 count, NeatNN** nn)
{
    i64 size = 0;

    for(i32 i = 0; i < count; ++i) {
        Genome& g = *genomes[i];
        g.totalNodeCount = g.inputNodeCount + g.outputNodeCount;

        for(i32 j = 0; j < g.geneCount; ++j) {
            if(g.genes[j].nodeOut >= g.totalNodeCount) {
                g.totalNodeCount = g.genes[j].nodeOut + 1;
            }
        }
    }

    i32 nnSize[2048];
    assert(count <= 2048);
    for(i32 i = 0; i < count; ++i) {
        nnSize[i] = 0;
        nnSize[i] += sizeof(NeatNN);
        nnSize[i] += sizeof(NeatNN::nodeValues[0]) * genomes[i]->totalNodeCount;
        nnSize[i] += sizeof(NeatNN::computations[0]) * genomes[i]->geneCount;
        size += nnSize[i];
    }

    u8* block = (u8*)malloc(size);

    for(i32 i = 0; i < count; ++i) {
        Genome& g = *genomes[i];
        nn[i] = (NeatNN*)block;
        nn[i]->nodeValues = (f64*)(nn[i] + 1);
        nn[i]->computations = (NeatNN::Computation*)(nn[i]->nodeValues + g.totalNodeCount);

        const i32 geneCount = g.geneCount;
        for(i32 j = 0; j < geneCount; ++j) {
            const Gene& gene = g.genes[j];
            if(gene.disabled) continue;
            nn[i]->computations[j] = { gene.nodeIn, gene.nodeOut, gene.weight };
        }

        qsort(nn[i]->computations, geneCount, sizeof(NeatNN::Computation), compareNeatNNComputations);

        nn[i]->computationsCount = geneCount;
        block += nnSize[i];
    }

    LOG("NEAT> allocated %d NeatNN, size=%lld", count, size);
}

void neatNnDealloc(void* ptr)
{
    free(ptr);
}
