#include "neat.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <malloc.h>

#define NEAT_MAX_SPECIES 1024

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
        g.species = 0;
        mem_zero(g.geneDisabled);
        mem_zero(g.layerNodeCount);
        mem_zero(g.nodePos);

        for(i16 n = 0; n < inputCount; ++n) {
            g.nodePos[n].layer = 0;
            g.nodePos[n].vpos = g.layerNodeCount[0]++;
        }
        for(i16 n = 0; n < outputCount; ++n) {
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
    i16 nodeIn;
    i16 nodeOut;
    NodePos posIn;
    NodePos posOut;
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
static i32 newInnovationNumber(i16 nodeIn, i16 nodeOut, const NodePos& nodePosIn, const NodePos& nodePosOut)
{
    i32 innovationNumber = -1;
    const i32 poolCount2 = g_structPoolCount;

    for(i32 j = 0; j < g_structPoolCount; ++j) {
        if(g_posPool[j].nodeIn == nodeIn &&
           g_posPool[j].nodeOut == nodeOut &&
           g_posPool[j].posIn.layer == nodePosIn.layer &&
           g_posPool[j].posIn.vpos == nodePosIn.vpos &&
           g_posPool[j].posOut.layer == nodePosOut.layer &&
           g_posPool[j].posOut.vpos == nodePosOut.vpos) {
            g_structMatchesFound++;
            return g_innNumPool[j];
        }
    }

    if(innovationNumber == -1) {
        innovationNumber = g_innovationNumber++;
    }

    assert(poolCount2 < 2048);
    i32 pid = g_structPoolCount++;
    g_posPool[pid] = { nodeIn, nodeOut, nodePosIn, nodePosOut };
    g_innNumPool[pid] = innovationNumber;
    return innovationNumber;
}

// geneCount -> larger genome geneCount
static f64 compatibilityDistance(const Gene* genes1, const Gene* genes2, i32 geneCount1, i32 geneCount2,
                                 f64 c1, f64 c2, f64 c3)
{
    i32 N = max(geneCount1, geneCount2);
    //if(N < 20) N = 1;

    // l: largest s: smallest
    const i32 geneCountL = geneCount1 > geneCount2 ? geneCount1 : geneCount2;
    const i32 geneCountS = geneCount1 <= geneCount2 ? geneCount1 : geneCount2;
    const Gene* genesL = geneCount1 > geneCount2 ? genes1 : genes2;
    const Gene* genesS = geneCount1 <= geneCount2 ? genes1 : genes2;

    i32 maxHistMarkL = 0;
    i32 maxHistMarkS = 0;

    for(i32 i = 0; i < geneCountL; ++i) {
        maxHistMarkL = max(maxHistMarkL, genesL[i].historicalMarker);
    }

    for(i32 i = 0; i < geneCountS; ++i) {
        maxHistMarkS = max(maxHistMarkS, genesS[i].historicalMarker);
    }

    const i32 maxCommonHistMark = min(maxHistMarkS, maxHistMarkL);

    i32 disjoint = 0;
    i32 excess = 0;
    f64 totalWeightDiff = 0.0;
    i32 matches = 0;

    u8 resultL[NEAT_MAX_GENES] = {0};
    u8 resultS[NEAT_MAX_GENES] = {0};

    enum {
        DISJOINT=0,
        MATCHING,
        EXCESS,
    };

    // compare L to S
    for(i32 l = 0; l < geneCountL; ++l) {
        const i32 innNumL = genesL[l].historicalMarker;
        if(innNumL > maxCommonHistMark) {
            resultL[l] = EXCESS;
            continue;
        }

        for(i32 s = 0; s < geneCountS; ++s) {
            const i32 innNumS = genesS[s].historicalMarker;
            if(innNumS > maxCommonHistMark) {
                resultS[s] = EXCESS;
                continue;
            }

            if(innNumL == genesS[s].historicalMarker) {
                totalWeightDiff += fabs(genesL[l].weight - genesS[s].weight);
                matches++;
                resultL[l] = MATCHING;
                resultS[s] = MATCHING;
                break;
            }
        }
    }

    for(i32 l = 0; l < geneCountL; ++l) {
        if(resultL[l] == DISJOINT) disjoint++;
        else if(resultL[l] == EXCESS) excess++;
    }

    for(i32 s = 0; s < geneCountS; ++s) {
        if(resultS[s] == DISJOINT) disjoint++;
        else if(resultS[s] == EXCESS) excess++;
    }

    f64 avgWeightDiff = totalWeightDiff / matches;

    f64 dist = (c1 * disjoint)/N + (c2 * excess)/N + c3 * avgWeightDiff;
    assert(dist < 100.0);
    return dist;
}

static i32 compareGenesAsc(const void* a, const void* b)
{
    const Gene& ga = *(Gene*)a;
    const Gene& gb = *(Gene*)b;
    if(ga.historicalMarker < gb.historicalMarker) return -1;
    if(ga.historicalMarker > gb.historicalMarker) return 1;
    return 0;
}

static i32 selectRoulette(const i32 count, f64* fitness, f64 totalFitness)
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

static void crossover(Genome* dest, const Genome* parentA, const Genome* parentB)
{
    constexpr f64 geneStayDisabledChance = 0.75;

    // NOTE: parentA is the most fit
    const i32 geneCountA = parentA->geneCount;
    const i32 geneCountB = parentB->geneCount;
    i32& geneCountOut = dest->geneCount;
    geneCountOut = 0;
    const Gene* genesA = parentA->genes;
    const Gene* genesB = parentB->genes;
    Gene* genesOut = dest->genes;
    const u8* geneDisabledA = parentA->geneDisabled;
    const u8* geneDisabledB = parentB->geneDisabled;
    u8* geneDisabledOut = dest->geneDisabled;
    mem_zero(dest->geneDisabled);

    i32 maxHistMarkA = 0;
    i32 maxHistMarkB = 0;

    for(i32 a = 0; a < geneCountA; ++a) {
        maxHistMarkA = max(maxHistMarkA, genesA[a].historicalMarker);
    }

    for(i32 b = 0; b < geneCountB; ++b) {
        maxHistMarkB = max(maxHistMarkB, genesB[b].historicalMarker);
    }

    const i32 maxCommonHistMark = min(maxHistMarkB, maxHistMarkA);

    enum {
        DISJOINT=0,
        MATCHING,
        EXCESS,
    };

    u8 resultA[NEAT_MAX_GENES] = {0};

    // compare A to B
    for(i32 a = 0; a < geneCountA; ++a) {
        const i32 innNumL = genesA[a].historicalMarker;
        if(innNumL > maxCommonHistMark) {
            resultA[a] = EXCESS;
            continue;
        }

        for(i32 b = 0; b < geneCountB; ++b) {
            const i32 innNumS = genesB[b].historicalMarker;
            if(innNumS > maxCommonHistMark) {
                // EXCESS
                continue;
            }

            // matching
            if(innNumL == genesB[b].historicalMarker) {
                resultA[a] = MATCHING;

                // choose at random one or the other
                const i32 gid = geneCountOut++;
                if(randf64(0.0, 1.0) < 0.5) {
                    genesOut[gid] = genesA[a];
                }
                else {
                    genesOut[gid] = genesB[b];
                }
                geneDisabledOut[gid] = randf64(0.0, 1.0) < geneStayDisabledChance ?
                            (geneDisabledA[a] | geneDisabledB[b]) : 0;
                break;
            }
        }
    }

    // inherit disjoint and excess genes from more fit parent
    for(i32 a = 0; a < geneCountA; ++a) {
        if(resultA[a] == DISJOINT || resultA[a] == EXCESS) {
            const i32 gid = geneCountOut++;
            genesOut[gid] = genesA[a];
            geneDisabledOut[gid] = randf64(0.0, 1.0) < geneStayDisabledChance ? geneDisabledA[a] : 0;
        }
    }

    // reconstruct metadata
    const i32 inputCount = parentA->inputNodeCount;
    const i32 outputCount = parentA->outputNodeCount;
    dest->inputNodeCount = inputCount;
    dest->outputNodeCount = outputCount;
    dest->totalNodeCount = inputCount + outputCount;
    dest->layerCount = 2;
    dest->species = parentA->species;
    mem_zero(dest->layerNodeCount);
    mem_zero(dest->nodePos);

    for(i16 n = 0; n < inputCount; ++n) {
        dest->nodePos[n].layer = 0;
    }
    for(i16 n = 0; n < outputCount; ++n) {
        dest->nodePos[n+inputCount].layer = 1;
    }

    const i32 geneCountOut2 = geneCountOut;

    // bubble sort genes
    bool sorting = true;
    while(sorting) {
        sorting = false;
        for(i32 i = 1; i < geneCountOut2; ++i) {
            if(genesOut[i-1].historicalMarker > genesOut[i].historicalMarker) {
                sorting = true;
                // swap
                Gene geneTemp = genesOut[i-1];
                genesOut[i-1] = genesOut[i];
                genesOut[i] = geneTemp;

                u8 disabledTemp = geneDisabledOut[i-1];
                geneDisabledOut[i-1] = geneDisabledOut[i];
                geneDisabledOut[i] = disabledTemp;
            }
        }
    }

    for(i32 i = 0; i < geneCountOut2; ++i) {
        const i16 nodeOut = genesOut[i].nodeOut;
        if(nodeOut >= dest->totalNodeCount) {
            dest->totalNodeCount = nodeOut+1;
        }
    }

    // compute node position
    const i32 nodeCount = dest->totalNodeCount;
    const i32 baseNodeCount = inputCount + outputCount;
    NodePos* nodePos = dest->nodePos;

    for(i32 i = 0; i < geneCountOut2; ++i) {
        const Gene& ge = genesOut[i];
        const i16 nodeIn = ge.nodeIn;
        const i16 nodeOut = ge.nodeOut;
        if(nodeOut < baseNodeCount) continue;

        i32 layer = nodePos[nodeIn].layer + 1;
        if(layer == dest->layerCount-1) {
            dest->layerCount++;
            for(i32 n2 = 0; n2 < nodeOut; ++n2) {
                if(nodePos[n2].layer >= layer) {
                    nodePos[n2].layer++;
                }
            }
        }
        nodePos[nodeOut].layer = layer;
    }

    u8* layerNodeCount = dest->layerNodeCount;
    for(i32 n = 0; n < nodeCount; ++n) {
        nodePos[n].vpos = layerNodeCount[nodePos[n].layer]++;
    }
}

void neatEvolve(Genome** genomes, Genome** nextGenomes, f64* fitness, const i32 popCount,
                const NeatEvolutionParams& params)
{
    assert(genomes);
    assert(nextGenomes);
    assert(fitness);
    assert(popCount > 0);

    timept t0 = timeGet();

#if 1
    FitnessPair* fpair = stack_arr(FitnessPair,popCount);

    for(i32 i = 0; i < popCount; ++i) {
        fpair[i] = { i, fitness[i] };
    }

    qsort(fpair, popCount, sizeof(FitnessPair), compareFitnessDesc);

    // eliminate worst half of the population
    const i32 parentCount = popCount * 0.5;

    for(i32 i = 0; i < parentCount; ++i) {
        memmove(nextGenomes[i], genomes[fpair[i].id], sizeof(Genome));
    }

#ifdef CONF_DEBUG
    memset(genomes[0], 0xAB, sizeof(Genome) * popCount);
#endif

    for(i32 i = 0; i < parentCount; ++i) {
        memmove(genomes[i], nextGenomes[i], sizeof(Genome));
    }

    // speciation
    Gene* speciesRepGenes[NEAT_MAX_SPECIES];
    i32 speciesRepGeneCount[NEAT_MAX_SPECIES];
    i32 speciesPopCount[NEAT_MAX_SPECIES] = {0};
    i32 speciesCount = 0;
    f64 biggestDist = 0.0;

    const f64 c1 = params.compC1;
    const f64 c2 = params.compC2;
    const f64 c3 = params.compC3;
    const f64 compatibilityThreshold = params.compT;

    for(i32 i = 0; i < parentCount; ++i) {
        Genome& g = *genomes[i];

        bool found = false;
        for(i32 s = 0; s < speciesCount; ++s) {
            f64 dist = compatibilityDistance(g.genes, speciesRepGenes[s], g.geneCount,
                                             speciesRepGeneCount[s], c1, c2, c3);
            biggestDist = max(dist, biggestDist);
            if(dist < compatibilityThreshold) {
                g.species = s;
                speciesPopCount[s]++;
                found = true;
                break;
            }
        }

        if(!found) {
            assert(speciesCount < NEAT_MAX_SPECIES);
            i32 sid = speciesCount++;
            speciesRepGenes[sid] = g.genes;
            speciesRepGeneCount[sid] = g.geneCount;
        }
    }

    LOG("NEAT> species count: %d (biggestDist=%g)", speciesCount, biggestDist);

    // fitness sharing
    f64* normFitness = stack_arr(f64,parentCount);
    for(i32 i = 0; i < parentCount; ++i) {
        normFitness[i] = fpair[i].fitness / speciesPopCount[i];
    }

    // crossover
    i32 noMatesFoundCount = 0;
    Genome** potentialMates = stack_arr(Genome*,parentCount);
    f64* pmFitness = stack_arr(f64,parentCount);

    for(i32 i = 0; i < popCount; ++i) {
        const i32 idA = randi64(0, parentCount-1);
        const Genome* mateA = genomes[idA];
        const i32 speciesA = mateA->species;

        // find same species mates
        i32 potentialMatesCount = 0;
        i32 pmTotalFitness = 0.0;
        for(i32 j = 0; j < parentCount; ++j) {
            if(idA == j) continue;

            Genome* mate = genomes[j];
            if(mate->species == speciesA) {
                i32 pmId = potentialMatesCount++;
                potentialMates[pmId] = mate;
                pmFitness[pmId] = normFitness[j];
                pmTotalFitness += pmFitness[pmId];
            }
        }

        if(potentialMatesCount < 1) {
            noMatesFoundCount++;
            memmove(nextGenomes[i], mateA, sizeof(Genome));
        }
        else {
            const i32 idB = selectRoulette(potentialMatesCount, pmFitness, pmTotalFitness);
            const Genome* mateB = genomes[idB];

            // parentA is the most fit
            const Genome* parentA = mateA;
            const Genome* parentB = mateB;
            if(normFitness[idA] < normFitness[idB]) {
                parentA = mateB;
                parentB = mateA;
            }
            crossover(nextGenomes[i], parentA, parentB);
        }
    }

    LOG("NEAT> noMatesFoundCount=%d", noMatesFoundCount);
#endif

#if 1
    // Mutation

    // save this evolution pass structural changes and use it to check if
    // a new structural change has already been assigned an innovation number
    resetStructuralChanges();

    i32 newConnectionMutationCount = 0;
    i32 newNodeMutationCount = 0;

    for(i32 i = 0; i < popCount; ++i) {
        Genome& g = *nextGenomes[i];
        //Genome& g = *genomes[i];

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
                gene.historicalMarker = newInnovationNumber(nodeIn, nodeOut, g.nodePos[nodeIn],
                                                            g.nodePos[nodeOut]);
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
            g.genes[con1].historicalMarker = newInnovationNumber(g.genes[con1].nodeIn,
                                                                 g.genes[con1].nodeOut,
                                                                 g.nodePos[g.genes[con1].nodeIn],
                                                                 g.nodePos[g.genes[con1].nodeOut]);

            i32 con2 = g.geneCount++;
            assert(con2 < NEAT_MAX_GENES);
            g.genes[con2] = { -1, newNodeId,
                              splitNodeOut, g.genes[splitId].weight };
            g.genes[con2].historicalMarker = newInnovationNumber(g.genes[con2].nodeIn,
                                                                 g.genes[con2].nodeOut,
                                                                 g.nodePos[g.genes[con2].nodeIn],
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

        qsort(g.genes, g.geneCount, sizeof(Gene), compareGenesAsc);
    }

    LOG("NEAT> mutations - connections=%d nodes=%d", newConnectionMutationCount, newNodeMutationCount);
    LOG("NEAT> structural matches=%d", g_structMatchesFound);
#endif

    memmove(genomes[0], nextGenomes[0], sizeof(Genome) * popCount);

    LOG("NEAT> evolution took %.3fs", timeToMicrosec(timeGet() - t0)/1000000.0);
}

void neatTestTryReproduce(const Genome& g1, const Genome& g2)
{
    const Gene* genes1 = g1.genes;
    const Gene* genes2 = g2.genes;
    i32 geneCount1 = g1.geneCount;
    i32 geneCount2 = g2.geneCount;

    i32 N = max(geneCount1, geneCount2);
    if(N < 20) N = 1;

    // l: largest s: smallest
    const i32 geneCountL = geneCount1 > geneCount2 ? geneCount1 : geneCount2;
    const i32 geneCountS = geneCount1 <= geneCount2 ? geneCount1 : geneCount2;
    const Gene* genesL = geneCount1 > geneCount2 ? genes1 : genes2;
    const Gene* genesS = geneCount1 <= geneCount2 ? genes1 : genes2;

    i32 maxHistMarkL = 0;
    i32 maxHistMarkS = 0;

    for(i32 i = 0; i < geneCountL; ++i) {
        maxHistMarkL = max(maxHistMarkL, genesL[i].historicalMarker);
    }

    for(i32 i = 0; i < geneCountS; ++i) {
        maxHistMarkS = max(maxHistMarkS, genesS[i].historicalMarker);
    }

    const i32 maxCommonHistMark = min(maxHistMarkS, maxHistMarkL);

    i32 disjoint = 0;
    i32 excess = 0;
    f64 totalWeightDiff = 0.0;
    i32 matches = 0;

    // compare L to S
    for(i32 i = 0; i < geneCountL; ++i) {
        bool found = false;
        const i32 innovationNumber = genesL[i].historicalMarker;

        for(i32 j = 0; j < geneCountS; ++j) {
            if(innovationNumber == genesS[j].historicalMarker) {
                totalWeightDiff += fabs(genesS[i].weight - genesL[j].weight);
                matches++;
                found = true;
                break;
            }
        }

        if(!found) {
            if(innovationNumber <= maxCommonHistMark) {
                disjoint++;
            }
            else {
                excess++;
            }
        }
    }

    // compare S to L
    for(i32 i = 0; i < geneCountS; ++i) {
        bool found = false;
        const i32 innovationNumber = genesS[i].historicalMarker;

        for(i32 j = 0; j < geneCountL; ++j) {
            if(innovationNumber == genesL[j].historicalMarker) {
                found = true;
                break;
            }
        }

        if(!found) {
            if(innovationNumber <= maxCommonHistMark) {
                disjoint++;
            }
            else {
                excess++;
            }
        }
    }

    f64 avgWeightDiff = totalWeightDiff / matches;

    f64 c1 = 1.0;
    f64 c2 = 1.0;
    f64 c3 = 0.4;
    f64 dist = (c1 * disjoint)/N + (c2 * excess)/N + c3 * avgWeightDiff;

    LOG("NEAT> TestTryReproduce:");
    LOG("- maxCommonHistMark: %d", maxCommonHistMark);
    LOG("- matches: %d", matches);
    LOG("- disjoint: %d", disjoint);
    LOG("- excess: %d", excess);
    LOG("- avgWeightDiff: %g", avgWeightDiff);
    LOG("- dist: %g", dist);
}

void neatTestCrossover(const Genome* parentA, const Genome* parentB, Genome* dest)
{
    crossover(dest, parentA, parentB);
}
