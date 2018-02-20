#include "neat.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static i32 g_innovationNumber = 0;

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
        g.totalNodeCount = inputCount + outputCount;
        g_innovationNumber = g.totalNodeCount; // TODO: have the user prove his one global innovation number
        g.layerCount = 2;
        g.geneCount = 0;
        mem_zero(g.geneDisabled);

        for(i16 n = 0; n < g.inputNodeCount; ++n) {
            g.nodeLayer[n] = 0;
        }
        for(i16 n = 0; n < g.inputNodeCount; ++n) {
            g.nodeLayer[n+g.inputNodeCount] = 1;
        }

        for(i16 in = 0; in < inputCount; ++in) {
            for(i16 out = 0; out < outputCount; ++out) {
                i32 gid = g.geneCount++;
                g.genes[gid] = { gid, in, (i16)(inputCount + out), randf64(-1.0, 1.0) };
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
            if(g.geneDisabled[j]) continue;
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

void neatEvolve(Genome** genomes, Genome** nextGenomes, f64* fitness, const i32 count)
{
#if 0
    FitnessPair fpair[2048];
    assert(count < 2048);

    for(i32 i = 0; i < count; ++i) {
        fpair[i] = { i, fitness[i] };
    }

    qsort(fpair, count, sizeof(FitnessPair), compareFitnessDesc);

    // eliminate worst half of the population
    const i32 bestCount = count * 0.5;

    for(i32 i = 0; i < bestCount; ++i) {
        memmove(nextGenomes[i], genomes[fpair[i].id], sizeof(Genome));
    }

#ifdef CONF_DEBUG
    memset(genomes[0], 0xAB, sizeof(Genome) * count);
#endif

    for(i32 i = 0; i < bestCount; ++i) {
        memmove(genomes[i], nextGenomes[i], sizeof(Genome));
    }

    // compute new totalNodeCount
    for(i32 i = 0; i < count; ++i) {
        Genome& g = *genomes[i];
        g.totalNodeCount = g.inputNodeCount + g.outputNodeCount;

        for(i32 j = 0; j < g.geneCount; ++j) {
            if(g.genes[j].nodeOut >= g.totalNodeCount) {
                g.totalNodeCount = g.genes[j].nodeOut + 1;
            }
        }
    }
#endif

    // test mutate
    for(i32 i = 0; i < count; ++i) {
        Genome& g = *genomes[i];

        // change weight
        if(randf64(0.0, 1.0) < 0.5) {
            i32 gid = randi64(0, g.geneCount-1);

            // add to weight (80%) or reset weight (20%)
            if(randf64(0.0, 1.0) < 0.8) {
                g.genes[gid].weight += randf64(-0.5, 0.5);
            }
            else {
                g.genes[gid].weight = randf64(-1.0, 1.0);
            }
        }

        // add connection
        if(randf64(0.0, 1.0) < 0.5) {
            const i32 outputLayer = g.layerCount-1;

            i16 nodeIn = randi64(0, g.totalNodeCount-1);
            while(g.nodeLayer[nodeIn] == outputLayer) {
                nodeIn = randi64(0, g.totalNodeCount-1);
            }

            i16 nodeOut = randi64(g.inputNodeCount, g.totalNodeCount-1);
            while(g.nodeLayer[nodeOut] <= g.nodeLayer[nodeIn]) {
                nodeOut = randi64(g.inputNodeCount, g.totalNodeCount-1);
            }

            // prevent connection overlapping
            bool found = false;
            for(i32 j = 0; j < g.geneCount; ++j) {
                if(g.genes[j].nodeIn == nodeIn &&
                   g.genes[j].nodeOut == nodeOut) {
                    found = true;
                    break;
                }
            }

            if(!found) {
                i32 gid = g.geneCount++;
                assert(gid < MAX_GENES);
                Gene& gene = g.genes[gid];
                gene.innovationNumber = g_innovationNumber++;
                gene.nodeIn = nodeIn;
                gene.nodeOut = nodeOut;
                gene.weight = randf64(-1.0, 1.0);
            }
        }

        // split connection -> 2 new connections (new node)
        if(randf64(0.0, 1.0) < 0.5) {
            i32 splitId = randi64(0, g.geneCount-1);
            g.geneDisabled[splitId] = true;
            const i16 splitNodeIn = g.genes[splitId].nodeIn;
            const i16 splitNodeOut = g.genes[splitId].nodeOut;

            i16 newNodeId = g.totalNodeCount++;
            assert(newNodeId < MAX_NODES);

            i32 con1 = g.geneCount++;
            assert(con1 < MAX_GENES);
            g.genes[con1] = { g_innovationNumber++, splitNodeIn,
                              newNodeId, 1.0 };

            i32 con2 = g.geneCount++;
            assert(con2 < MAX_GENES);
            g.genes[con2] = { g_innovationNumber++, newNodeId,
                              splitNodeOut, g.genes[splitId].weight };

            // add a new layer if needed
            if(g.nodeLayer[splitNodeIn] + 1 == g.nodeLayer[splitNodeOut]) {
                g.layerCount++;
                const i32 expandFrom = g.nodeLayer[splitNodeOut];

                for(i32 n = 0; n < g.totalNodeCount; ++n) {
                    if(g.nodeLayer[n] >= expandFrom) {
                        g.nodeLayer[n]++;
                    }
                }
            }

            g.nodeLayer[newNodeId] = g.nodeLayer[splitNodeIn] + 1;
        }
    }
}
