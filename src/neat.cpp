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
        mem_zero(g.layerNodeCount);
        mem_zero(g.nodePos);

        for(i16 n = 0; n < g.inputNodeCount; ++n) {
            g.nodePos[n].layer = 0;
            g.nodePos[n].vpos = g.layerNodeCount[0]++;
        }
        for(i16 n = 0; n < inputCount; ++n) {
            g.nodePos[n+inputCount].layer = 1;
            g.nodePos[n+inputCount].vpos = g.layerNodeCount[1]++;
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

struct StructuralConnection
{
    NodePos in;
    NodePos out;
};

i32 g_innNumPool[2048];
StructuralConnection g_posPool[2048];
i32 g_structPoolCount;
i32 g_structMatchesFound;

static void resetStructuralChanges()
{
    g_structPoolCount = 0;
    g_structMatchesFound = 0;
}

// I think this way of checking if a structural change is equivalent works
// TODO: test it
static i32 newInnovationNumber(const NodePos& nodePosIn, const NodePos& nodePosOut)
{
    i32 innovationNumber = -1;
    const i32 poolCount2 = g_structPoolCount;

    for(i32 j = 0; j < g_structPoolCount; ++j) {
        if(g_posPool[j].in.layer == nodePosIn.layer &&
           g_posPool[j].in.vpos == nodePosIn.vpos &&
           g_posPool[j].out.layer == nodePosOut.layer &&
           g_posPool[j].out.vpos == nodePosOut.vpos) {
            g_structMatchesFound++;
            return g_innNumPool[j];
        }
    }

    if(innovationNumber == -1) {
        innovationNumber = g_innovationNumber++;
    }

    assert(poolCount2 < 2048);
    i32 pid = g_structPoolCount++;
    g_posPool[pid] = { nodePosIn, nodePosOut };
    g_innNumPool[pid] = innovationNumber;
    return innovationNumber;
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

    // save this evolution pass structural changes and use it to check if
    // a new structural change has already been assigned an innovation number
    resetStructuralChanges();

    i32 newConnectionMutationCount = 0;
    i32 newNodeMutationCount = 0;

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
            while(g.nodePos[nodeIn].layer == outputLayer) {
                nodeIn = randi64(0, g.totalNodeCount-1);
            }

            i16 nodeOut = randi64(g.inputNodeCount, g.totalNodeCount-1);
            while(g.nodePos[nodeOut].layer <= g.nodePos[nodeIn].layer) {
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
                assert(gid < NEAT_MAX_GENES);
                Gene& gene = g.genes[gid];
                gene.nodeIn = nodeIn;
                gene.nodeOut = nodeOut;
                gene.weight = randf64(-1.0, 1.0);
                gene.innovationNumber = newInnovationNumber(g.nodePos[nodeIn], g.nodePos[nodeOut]);
                newConnectionMutationCount++;
            }
        }

        // split connection -> 2 new connections (new node)
        if(randf64(0.0, 1.0) < 0.5) {
            i32 splitId = randi64(0, g.geneCount-1);
            g.geneDisabled[splitId] = true;
            const i16 splitNodeIn = g.genes[splitId].nodeIn;
            const i16 splitNodeOut = g.genes[splitId].nodeOut;

            i16 newNodeId = g.totalNodeCount++;
            assert(newNodeId < NEAT_MAX_NODES);

            i32 con1 = g.geneCount++;
            assert(con1 < NEAT_MAX_GENES);
            g.genes[con1] = { -1, splitNodeIn,
                              newNodeId, 1.0 };
            g.genes[con1].innovationNumber = newInnovationNumber(g.nodePos[g.genes[con1].nodeIn],
                                                                 g.nodePos[g.genes[con1].nodeOut]);

            i32 con2 = g.geneCount++;
            assert(con2 < NEAT_MAX_GENES);
            g.genes[con2] = { -1, newNodeId,
                              splitNodeOut, g.genes[splitId].weight };
            g.genes[con2].innovationNumber = newInnovationNumber(g.nodePos[g.genes[con2].nodeIn],
                                                                 g.nodePos[g.genes[con2].nodeOut]);

            // add a new layer if needed
            if(g.nodePos[splitNodeIn].layer + 1 == g.nodePos[splitNodeOut].layer) {
                assert(g.layerCount < NEAT_MAX_LAYERS);
                g.layerCount++;
                const i32 expandFrom = g.nodePos[splitNodeOut].layer;

                for(i32 n = 0; n < g.totalNodeCount; ++n) {
                    if(g.nodePos[n].layer >= expandFrom) {
                        g.nodePos[n].layer++;
                    }
                }

                memmove(&g.layerNodeCount[expandFrom+1],
                        &g.layerNodeCount[expandFrom],
                        (g.layerCount-expandFrom) * sizeof(g.layerNodeCount[0]));
                g.layerNodeCount[expandFrom] = 0;
            }

            i32 layer = g.nodePos[splitNodeIn].layer + 1;
            g.nodePos[newNodeId].layer = layer;
            g.nodePos[newNodeId].vpos = g.layerNodeCount[layer]++;
            newNodeMutationCount++;
        }
    }

    LOG("NEAT> mutations - connections=%d nodes=%d", newConnectionMutationCount, newNodeMutationCount);
    LOG("NEAT> structural matches=%d", g_structMatchesFound);
}
