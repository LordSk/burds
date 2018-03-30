#include "neat.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <malloc.h>

#define activation(x) tanh(x)
//#define activation(x) (1.0/(1.0+exp(-4.9*x)))

NeatSpeciation::~NeatSpeciation()
{
    if(speciesRep) {
        free(speciesRep);
    }
}

static i32 g_innovationNumber = 0;

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

    u8* resultL = stack_arr(u8,geneCountL);
    u8* resultS = stack_arr(u8,geneCountS);
    // DISJOINT by default
    memset(resultL, 0, geneCountL);
    memset(resultS, 0, geneCountS);

    enum {
        DISJOINT=0,
        MATCHING,
        EXCESS,
    };

    // excess
    for(i32 l = 0; l < geneCountL; ++l) {
        const i32 innNumL = genesL[l].historicalMarker;
        if(innNumL > maxCommonHistMark) {
            resultL[l] = EXCESS;
        }
    }
    for(i32 s = 0; s < geneCountS; ++s) {
        const i32 innNumS = genesS[s].historicalMarker;
        if(innNumS > maxCommonHistMark) {
            resultS[s] = EXCESS;
        }
    }

    // matches
    for(i32 l = 0; l < geneCountL; ++l) {
        if(resultL[l] != DISJOINT) continue;

        for(i32 s = 0; s < geneCountS; ++s) {
            if(resultS[s] != DISJOINT) continue;

            if(genesL[l].historicalMarker == genesS[s].historicalMarker) {
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

void neatGenomeAlloc(Genome** genomes, const i32 count)
{
    i64 size = sizeof(Genome) * count;
    u8* block = (u8*)malloc(size);

    for(i32 i = 0; i < count; ++i) {
        genomes[i] = (Genome*)(block + sizeof(Genome) * i);
    }

    LOG("NEAT> allocated %d genomes, size=%lld", count, size);
}

void neatGenomeDealloc(Genome** genomes)
{
    free(genomes[0]);
}

void neatGenomeInit(Genome** genomes, const i32 popCount, i32 inputCount, i32 outputCount,
                    const NeatEvolutionParams& params, NeatSpeciation* speciation)
{
    assert(inputCount > 0);
    assert(outputCount > 0);

    for(i32 i = 0; i < popCount; ++i) {
        Genome& g = *genomes[i];
        g.inputNodeCount = inputCount;
        g.outputNodeCount = outputCount;
        g.totalNodeCount = inputCount + outputCount;
        g_innovationNumber = g.totalNodeCount; // TODO: have the user prove his one global innovation number
        g.geneCount = 0;
        g.species = -1;
        mem_zero(g.geneDisabled);
        mem_zero(g.nodePos);

        for(i16 n = 0; n < g.totalNodeCount; ++n) {
            g.nodeOriginMarker[n] = n;
        }

        for(i16 in = 0; in < inputCount; ++in) {
            for(i16 out = 0; out < outputCount; ++out) {
                i32 gid = g.geneCount++;
                g.genes[gid] = { gid, in, (i16)(inputCount + out), randf64(-1.0, 1.0) };
            }
        }
    }

    // speciation
    assert(speciation->speciesRep == nullptr);
    speciation->speciesRep = (Genome*)malloc(sizeof(Genome) * NEAT_MAX_SPECIES);
    mem_zero(speciation->speciesPopCount);

    Genome* speciesRep = speciation->speciesRep;
    i32* speciesPopCount = speciation->speciesPopCount;
    i32& speciesCount = speciation->speciesCount;
    speciesCount = 0;
    f64 biggestDist = 0.0;

    const f64 c1 = params.compC1;
    const f64 c2 = params.compC2;
    const f64 c3 = params.compC3;
    const f64 compatibilityThreshold = params.compT;

    for(i32 i = 0; i < popCount; ++i) {
        Genome& g = *genomes[i];

        bool found = false;
        for(i32 s = 0; s < speciesCount; ++s) {
            if(speciesPopCount[s] == 0) continue;

            f64 dist = compatibilityDistance(g.genes, speciesRep[s].genes, g.geneCount,
                                             speciesRep[s].geneCount, c1, c2, c3);
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
            speciesRep[sid] = g;
            speciesPopCount[sid] = 1;
            g.species = sid;
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

static void sortComputationsByDependency(NeatNN::Computation* computations, const i32 compCount,
                                         i16 firstOutNode, i16 lastOutNodePlusOne, const i32 nodeCount)
{
    i16* dependList = stack_arr(i16,nodeCount);
    i16* nextDependList = stack_arr(i16,nodeCount);
    i32 nextDependListCount = 0;
    i32 sortCompCount = compCount;

    // first populate depend list (output nodes)
    for(i16 o = firstOutNode; o != lastOutNodePlusOne; ++o) {
        nextDependList[nextDependListCount++] = o;
    }

    while(nextDependListCount > 0) {
        assert(nextDependListCount <= nodeCount);

        // quick sort depend list (descending)
        qsort(nextDependList, nextDependListCount, sizeof(nextDependList[0]),
             [](const void* pa, const void* pb) {
                i16 a = *(i16*)pa;
                i16 b = *(i16*)pb;
                if(a < b) return 1;
                if(a > b) return -1;
                return 0;
             }
        );

        memmove(dependList, nextDependList, sizeof(dependList[0]) * nextDependListCount);
        const i32 dependListCount = nextDependListCount;
        nextDependListCount = 0;

        for(i32 d = 0; d < dependListCount; ++d) {
            const i16 dependNodeId = dependList[d];

            // bubble sort (nodeOut == dependNodeId -> goes to end)
            bool bubble = true;
            while(bubble) {
                bubble = false;
                for(i32 s = 1; s < sortCompCount; ++s) {
                    if(computations[s-1].nodeOut == dependNodeId &&
                       computations[s].nodeOut != dependNodeId) {
                        // swap
                        NeatNN::Computation temp = computations[s];
                        computations[s] = computations[s-1];
                        computations[s-1] = temp;
                        bubble = true;
                    }
                }
            }

            // find how many we sorted down
            i32 where = -1;
            for(i32 s = sortCompCount-1; s >= 0; s--) {
                if(computations[s].nodeOut != dependNodeId) {
                    where = s + 1;
                    break;
                }
            }

            // shorten sortCount, prepare next dependency batch
            if(where != -1 && where != sortCompCount) {
                // get all the nodes to depend on
                for(i32 t = where; t < sortCompCount; ++t) {
                    const i16 nodeIn = computations[t].nodeIn;

                    // add unique nodeId entry
                    bool found = false;
                    for(i32 nd = 0; nd < nextDependListCount; ++nd) {
                        if(nextDependList[nd] == nodeIn) {
                            found = true;
                            break;
                        }
                    }

                    if(!found) {
                        nextDependList[nextDependListCount++] = nodeIn;
                    }
                }
                sortCompCount = where;
            }
        }
    }
}

static void sortGenesByHistoricalMarker(Genome* genome)
{
    Genome& g = *genome;
    Gene* genes = g.genes;
    u8* geneDisabled = g.geneDisabled;
    const i32 geneCount = g.geneCount;

    bool sorting = true;
    while(sorting) {
        sorting = false;
        for(i32 i = 1; i < geneCount; ++i) {
            if(genes[i-1].historicalMarker > genes[i].historicalMarker) {
                sorting = true;
                // swap
                Gene geneTemp = genes[i-1];
                genes[i-1] = genes[i];
                genes[i] = geneTemp;

                u8 disabledTemp = geneDisabled[i-1];
                geneDisabled[i-1] = geneDisabled[i];
                geneDisabled[i] = disabledTemp;
            }
        }
    }
}

void neatGenomeMakeNN(Genome** genomes, const i32 count, NeatNN** nn, bool verbose)
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
    memset(block, 0, size);

    for(i32 i = 0; i < count; ++i) {
        Genome& g = *genomes[i];
        nn[i] = (NeatNN*)block;
        nn[i]->nodeValues = (f64*)(nn[i] + 1);
        nn[i]->computations = (NeatNN::Computation*)(nn[i]->nodeValues + g.totalNodeCount);
        nn[i]->nodeCount = g.totalNodeCount;

        const i32 geneCount = g.geneCount;
        i32 compCount = 0;
        for(i32 j = 0; j < geneCount; ++j) {
            const Gene& gene = g.genes[j];
            if(g.geneDisabled[j]) continue;
            assert(gene.nodeIn >= 0 && gene.nodeIn < g.totalNodeCount);
            assert(gene.nodeOut >= 0 && gene.nodeOut < g.totalNodeCount);
            nn[i]->computations[compCount++] = { gene.nodeIn, gene.nodeOut, gene.weight };
        }

        // FIXME: some "backwards" do not propagate as expected (doing 2 propagates yield a different result)
        sortComputationsByDependency(nn[i]->computations, compCount,
                                     g.inputNodeCount, g.inputNodeCount + g.outputNodeCount,
                                     g.totalNodeCount);

        nn[i]->computationsCount = compCount;
        block += nnSize[i];

        if(verbose) {
            for(i32 c = 0; c < compCount; ++c) {
                const auto& comp = nn[i]->computations[c];
                LOG("#%d computation[%d] = { %d, %d, %g }", i, c, comp.nodeIn, comp.nodeOut, comp.weight);
            }
        }
    }

    if(verbose) LOG("NEAT> allocated %d NeatNN, size=%lld", count, size);
}

void neatNnPropagate(NeatNN** nn, const i32 nnCount)
{
    for(i32 i = 0; i < nnCount; ++i) {
        const i32 compCount = nn[i]->computationsCount;
        const NeatNN::Computation* computations = nn[i]->computations;
        f64* nodeValues = nn[i]->nodeValues;

        i16 curNoteOut = computations[0].nodeOut;
        f64 value = 1.0; // +bias

        for(i32 c = 0; c < compCount; ++c) {
            const NeatNN::Computation& comp = computations[c];
            if(curNoteOut == comp.nodeOut) {
                value += comp.weight * nodeValues[comp.nodeIn];
            }
            else {
                nodeValues[curNoteOut] = activation(clamp(value, -10.0, 10.0));
                curNoteOut = comp.nodeOut;
                value = comp.weight * nodeValues[comp.nodeIn] + 1.0; // +bias
            }
        }
        nodeValues[curNoteOut] = activation(clamp(value, -10.0, 10.0));
    }
}

void neatNnDealloc(NeatNN** nn)
{
    free(nn[0]);
}

struct FitnessPair
{
    i32 id;
    i32 species;
    f64 fitness;
};

static i32 compareFitnessDesc(const void* a, const void* b)
{
    const FitnessPair* fa = (FitnessPair*)a;
    const FitnessPair* fb = (FitnessPair*)b;

    // by species first
    if(fa->species < fb->species) return -1;
    if(fa->species > fb->species) return 1;

    // then by fitness
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
    i32 nodeInMarker;
    i32 nodeOutMarker;
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

// I think this way of checking structural change equivalence works
// TODO: test it
static i32 newInnovationNumber(i16 nodeIn, i16 nodeOut, const i32* nodeOrignMarkers)
{
    //return g_innovationNumber++;

    i32 innovationNumber = -1;
    const i32 poolCount2 = g_structPoolCount;
    const i32 nodeInMarker = nodeOrignMarkers[nodeIn];
    const i32 nodeOutMarker = nodeOrignMarkers[nodeOut];

    for(i32 j = 0; j < g_structPoolCount; ++j) {
        if(g_posPool[j].nodeIn == nodeIn &&
           g_posPool[j].nodeOut == nodeOut &&
           g_posPool[j].nodeInMarker == nodeInMarker &&
           g_posPool[j].nodeOutMarker == nodeOutMarker) {
            g_structMatchesFound++;
            return g_innNumPool[j];
        }
    }

    if(innovationNumber == -1) {
        innovationNumber = g_innovationNumber++;
    }

    assert(poolCount2 < 2048);
    i32 pid = g_structPoolCount++;
    g_posPool[pid] = { nodeIn, nodeOut, nodeInMarker, nodeOutMarker };
    g_innNumPool[pid] = innovationNumber;
    return innovationNumber;
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
            //LOG("total=%g s=%g r=%g j=%d", totalFitness, s, r, j);
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
    const Gene* genesA = parentA->genes;
    const Gene* genesB = parentB->genes;
    const u8* geneDisabledA = parentA->geneDisabled;
    const u8* geneDisabledB = parentB->geneDisabled;
    i32& geneCountOut = dest->geneCount;
    geneCountOut = 0;
    Gene* genesOut = dest->genes;
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
    u8 resultB[NEAT_MAX_GENES] = {0};
    u8 genesOutParent[NEAT_MAX_GENES];

    // compare A to B
    for(i32 a = 0; a < geneCountA; ++a) {
        const i32 innNumL = genesA[a].historicalMarker;
        if(innNumL > maxCommonHistMark) {
            resultA[a] = EXCESS;
            continue;
        }

        for(i32 b = 0; b < geneCountB; ++b) {
            if(resultB[b] != DISJOINT) continue;

            const i32 innNumS = genesB[b].historicalMarker;
            if(innNumS > maxCommonHistMark) {
                resultB[b] = EXCESS;
                continue;
            }

            // matching
            if(innNumL == genesB[b].historicalMarker) {
                resultA[a] = MATCHING;
                resultB[b] = MATCHING;

                // choose at random one or the other
                const i32 gid = geneCountOut++;
                if(randf64(0.0, 1.0) < 0.5) {
                    genesOut[gid] = genesA[a];
                    genesOutParent[gid] = 0;
                }
                else {
                    genesOut[gid] = genesB[b];
                    genesOutParent[gid] = 1;
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
            genesOutParent[gid] = 0;
            geneDisabledOut[gid] = randf64(0.0, 1.0) < geneStayDisabledChance ? geneDisabledA[a] : 0;
        }
    }
    // still inherit not common disjoint from less fit parent
    for(i32 b = 0; b < geneCountB; ++b) {
        if(resultB[b] == DISJOINT) {
            const i32 markB = genesB[b].historicalMarker;

            bool found = false;
            for(i32 c = 0; c < geneCountOut; c++) {
                const i32 markC = genesOut[c].historicalMarker;
                if(markC == markB) {
                    found = true;
                    break;
                }
            }

            if(!found) {
                const i32 gid = geneCountOut++;
                genesOut[gid] = genesB[b];
                genesOutParent[gid] = 1;
                geneDisabledOut[gid] = randf64(0.0, 1.0) < geneStayDisabledChance ? geneDisabledB[b] : 0;
            }
        }
    }

    const i32 geneCountOut2 = geneCountOut;

    // TODO: purge same connections
    constexpr auto cmpGenes = [](const Gene& a, const Gene& b, u8 parentA, u8 parentB) {
        if(a.nodeIn < b.nodeIn) return -1;
        if(a.nodeIn > b.nodeIn) return 1;
        if(a.nodeOut < b.nodeOut) return -1;
        if(a.nodeOut > b.nodeOut) return 1;
        if(parentA < parentB) return -1;
        if(parentA > parentB) return 1;
        return 0;
    };

    // bubble sort by nodeIn/nodeOut
    bool sorting = true;
    while(sorting) {
        sorting = false;
        for(i32 i = 1; i < geneCountOut2; ++i) {
            if(cmpGenes(genesOut[i-1], genesOut[i], genesOutParent[i-1], genesOutParent[i]) == 1) {
                sorting = true;
                // swap
                Gene geneTmp = genesOut[i-1];
                genesOut[i-1] = genesOut[i];
                genesOut[i] = geneTmp;
                u8 disTmp = geneDisabledOut[i-1];
                geneDisabledOut[i-1] = geneDisabledOut[i];
                geneDisabledOut[i] = disTmp;
                u8 parentTmp = genesOutParent[i-1];
                genesOutParent[i-1] = genesOutParent[i];
                genesOutParent[i] = parentTmp;
            }
        }
    }

    i16 checkNodeIn = -1;
    i16 checkNodeOut = -1;
    i32 sameConnCount = 0;
    Gene* genesPurged = stack_arr(Gene,geneCountOut2);
    u8* genesPurgedDisabled = stack_arr(u8,geneCountOut2);
    i32 genesPurgedCount = 0;

    for(i32 i = 0; i < geneCountOut2; ++i) {
        const Gene& ge = genesOut[i];
        if(checkNodeIn == ge.nodeIn && checkNodeOut == ge.nodeOut) {
            sameConnCount++;
        }
        else {
            checkNodeIn = ge.nodeIn;
            checkNodeOut = ge.nodeOut;
            const i32 gid = genesPurgedCount++;
            genesPurged[gid] = ge;
            genesPurgedDisabled[gid] = geneDisabledOut[i];
        }
    }

    /*if(sameConnCount) {
       LOG("crossover same connections: %d", sameConnCount);
    }*/

    memmove(genesOut, genesPurged, sizeof(genesOut[0]) * genesPurgedCount);
    const i32 geneOutCountFinal = genesPurgedCount;

    // reconstruct metadata
    const i32 inputCount = parentA->inputNodeCount;
    const i32 outputCount = parentA->outputNodeCount;
    dest->inputNodeCount = inputCount;
    dest->outputNodeCount = outputCount;
    dest->species = parentA->species;
    dest->totalNodeCount = inputCount + outputCount;
    mem_zero(dest->nodePos);

    for(i32 i = 0; i < geneOutCountFinal; ++i) {
        const i16 nodeIn = genesOut[i].nodeIn;
        const i16 nodeOut = genesOut[i].nodeOut;
        if(nodeIn >= dest->totalNodeCount) {
            dest->totalNodeCount = nodeIn+1;
        }
        if(nodeOut >= dest->totalNodeCount) {
            dest->totalNodeCount = nodeOut+1;
        }
    }
}

void neatGenomeSpeciation(Genome** genomes, const i32 popCount, i32* speciesPopCount)
{

}

void neatEvolve(Genome** genomes, Genome** nextGenomes, f64* fitness, const i32 popCount,
                NeatSpeciation* neatSpec, const NeatEvolutionParams& params, bool verbose)
{
    assert(genomes);
    assert(nextGenomes);
    assert(fitness);
    assert(popCount > 0);

    timept t0 = timeGet();

    f64 speciesMaxFitness[NEAT_MAX_SPECIES];
    i32* speciesPopCount = neatSpec->speciesPopCount;
    i32& speciesCount = neatSpec->speciesCount;

    for(i32 i = 0; i < popCount; ++i) {
        const i32 s = genomes[i]->species;
        assert(s >= 0 && s < speciesCount); // forgot to neatGenomeInit() ?
        speciesMaxFitness[s] = max(fitness[i], speciesMaxFitness[s]);
    }

    // species stagnation
    u8* deleteSpecies = stack_arr(u8,speciesCount);
#if 1
    const i32 stagnationT = params.speciesStagnationMax;
    u16* specStagnation = neatSpec->stagnation;
    f64* specStagMaxFitness = neatSpec->maxFitness;

    for(i32 s = 0; s < speciesCount; ++s) {
        deleteSpecies[s] = false;
        if(speciesMaxFitness[s] <= specStagMaxFitness[s]) {
            specStagnation[s]++;

            if(specStagnation[s] > stagnationT) {
                if(verbose) LOG("NEAT> species %x stagnating (%d)", s, specStagnation[s]);
                deleteSpecies[s] = true;
                specStagnation[s] = 0;
            }
        }
        else {
            specStagMaxFitness[s] = speciesMaxFitness[s];
            specStagnation[s] = 0;
        }
    }
#endif

    // keep best species always
    f64 bestMaxFitness = 0.0;
    i32 bestSpecies = -1;
    for(i32 s = 0; s < speciesCount; ++s) {
        if(speciesMaxFitness[s] > bestMaxFitness) {
            bestMaxFitness = speciesMaxFitness[s];
            bestSpecies = s;
        }
    }
    assert(bestSpecies != -1);
    deleteSpecies[bestSpecies] = false;
    specStagnation[bestSpecies] = 0;

#if 1
    FitnessPair* fpair = stack_arr(FitnessPair,popCount);
    for(i32 i = 0; i < popCount; ++i) {
        fpair[i] = { i, genomes[i]->species, fitness[i] };
    }

    qsort(fpair, popCount, sizeof(FitnessPair), compareFitnessDesc);

    // eliminate worst half of each species
    i32* speciesPopCountHalf = stack_arr(i32,speciesCount);
    for(i32 s = 0; s < speciesCount; ++s) {
        speciesPopCountHalf[s] = max(speciesPopCount[s] / 2, 1);
    }

    f64* parentFitness = stack_arr(f64,popCount);
    i32 parentCount = 0;
    i32 curSpecies = genomes[fpair[0].id]->species;
    i32 curSpeciesPopCount = 0;

    for(i32 i = 0; i < popCount; ++i) {
        Genome* g = genomes[fpair[i].id];
        const f64 fit = fpair[i].fitness;
        const i32 spec = g->species;
        if(deleteSpecies[spec]) continue; // do not copy over stagnating species

        if(spec == curSpecies) {
            if(curSpeciesPopCount < speciesPopCountHalf[spec]) {
                curSpeciesPopCount++;
                i32 pid = parentCount++;
                memmove(nextGenomes[pid], g, sizeof(Genome));
                parentFitness[pid] = fit;
            }
        }
        else {
            curSpecies = spec;
            curSpeciesPopCount = 1;
            i32 pid = parentCount++;
            memmove(nextGenomes[pid], g, sizeof(Genome));
            parentFitness[pid] = fit;
        }
    }

    if(parentCount == 0) {
        assert(0);
        /*if(verbose) LOG("NEAT> all species stagnated");
        Genome* g = genomes[0];
        neatGenomeInit(genomes, popCount, g->inputNodeCount, g->outputNodeCount);
        *speciesStagn = {};
        return;*/
    }

#ifdef CONF_DEBUG
    memset(genomes[0], 0xAB, sizeof(Genome) * popCount);
#endif

    memmove(genomes[0], nextGenomes[0], sizeof(Genome) * parentCount);

    // all parents are in genomes[0,parentCount]

    // copy champion of each species unchanged
    i32 championCount = 0;
    i32 champCheckSpec = -1;
    for(i32 i = 0; i < parentCount; ++i) {
        Genome* g = genomes[i];
        const i32 spec = g->species;
        if(spec != champCheckSpec && speciesPopCount[spec] > 4) {
            memmove(nextGenomes[popCount - 1 - (championCount++)], g, sizeof(Genome));
            champCheckSpec = spec;
        }
    }

    const i32 popCountMinusChamps = popCount - championCount;

    // fitness sharing
    f64* normFitness = stack_arr(f64,parentCount);
    f64 totalNormFitness = 0.0;
    for(i32 i = 0; i < parentCount; ++i) {
        normFitness[i] = parentFitness[i] / speciesPopCount[genomes[i]->species];
        totalNormFitness += normFitness[i];
    }

    // crossover
    i32 noMatesFoundCount = 0;
    Genome** potentialMates = stack_arr(Genome*,parentCount);
    f64* pmFitness = stack_arr(f64,parentCount);

    for(i32 i = 0; i < popCountMinusChamps; ++i) {
        //const i32 idA = randi64(0, parentCount-1);
        const i32 idA = selectRoulette(parentCount, normFitness, totalNormFitness);
        const Genome* mateA = genomes[idA];
        const i32 speciesA = mateA->species;

        // no crossover
        if(randf64(0.0, 1.0) < 0.25) {
            memmove(nextGenomes[i], mateA, sizeof(Genome));
            continue;
        }

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
            const Genome* mateB = potentialMates[idB];

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

    for(i32 i = 0; i < popCount; ++i) {
        sortGenesByHistoricalMarker(nextGenomes[i]);
    }

    if(verbose) LOG("NEAT> noMatesFoundCount=%d", noMatesFoundCount);
#endif

#if 1
    // Mutation

    // save this evolution pass structural changes and use it to check if
    // a new structural change has already been assigned an innovation number
    resetStructuralChanges();

    i32 mutConnections = 0;
    i32 mutNodes = 0;
    i32 mutGenesDisabled = 0;
    i32 mutGenesRemoved = 0;

    for(i32 i = 0; i < popCountMinusChamps; ++i) {
        Genome& g = *nextGenomes[i];
        //Genome& g = *genomes[i];

        // disable gene
        if(randf64(0.0, 1.0) < params.mutateDisableGene) {
            const i32 gid = randi64(0, g.geneCount-1);
            g.geneDisabled[gid] = true;
            mutGenesDisabled++;
        }

        // remove gene
        if(randf64(0.0, 1.0) < params.mutateRemoveGene) {
            assert(g.geneCount > 1);
            const i32 gid = randi64(0, g.geneCount-1);
            g.genes[gid] = g.genes[g.geneCount-1];
            g.geneDisabled[gid] = g.geneDisabled[g.geneCount-1];
            g.geneCount--;
            mutGenesRemoved++;
        }

        // change weight
        if(randf64(0.0, 1.0) < params.mutateWeight) {
            i32 gid = randi64(0, g.geneCount-1);

            // add to weight or reset weight
            if(randf64(0.0, 1.0) < params.mutateResetWeight) {
                g.genes[gid].weight = randf64(-1.0, 1.0);
            }
            else {
                f64 step = params.mutateWeightStep;
                g.genes[gid].weight += randf64(-step, step);
            }
        }

        // add connection
        if(randf64(0.0, 1.0) < params.mutateAddConn) {
            constexpr auto isOutput = [](i32 id, const Genome& g) {
                return id >= g.inputNodeCount && id < g.inputNodeCount + g.outputNodeCount;
            };

            i16 nodeIn = randi64(0, g.totalNodeCount-1); // input or hidden
            i16 nodeOut = randi64(g.inputNodeCount, g.totalNodeCount-1); // hidden or output
            while(nodeOut == nodeIn || isOutput(nodeIn, g)) {
                nodeIn = randi64(0, g.totalNodeCount-1);
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
                gene.historicalMarker = newInnovationNumber(nodeIn, nodeOut, g.nodeOriginMarker);
                mutConnections++;
            }
        }

        // split connection -> 2 new connections (new node)
        if(randf64(0.0, 1.0) < params.mutateAddNode) {
            i32 splitId = randi64(0, g.geneCount-1);
            g.geneDisabled[splitId] = true;
            const i16 splitNodeIn = g.genes[splitId].nodeIn;
            const i16 splitNodeOut = g.genes[splitId].nodeOut;

            i16 newNodeId = g.totalNodeCount++;
            assert(newNodeId < NEAT_MAX_NODES);
            g.nodeOriginMarker[newNodeId] = splitId;

            i32 con1 = g.geneCount++;
            assert(con1 < NEAT_MAX_GENES);
            g.genes[con1] = { -1, splitNodeIn,
                              newNodeId, 1.0 };
            g.genes[con1].historicalMarker = newInnovationNumber(g.genes[con1].nodeIn,
                                                                 g.genes[con1].nodeOut,
                                                                 g.nodeOriginMarker);

            i32 con2 = g.geneCount++;
            assert(con2 < NEAT_MAX_GENES);
            g.genes[con2] = { -1, newNodeId,
                              splitNodeOut, g.genes[splitId].weight };
            g.genes[con2].historicalMarker = newInnovationNumber(g.genes[con2].nodeIn,
                                                                 g.genes[con2].nodeOut,
                                                                 g.nodeOriginMarker);
            mutNodes++;
        }
    }

    for(i32 i = 0; i < popCount; ++i) {
        sortGenesByHistoricalMarker(nextGenomes[i]);
    }

    if(verbose) {
        LOG("NEAT> mutations - connections=%d nodes=%d disabled=%d removed=%d",
            mutConnections, mutNodes,
            mutGenesDisabled, mutGenesRemoved);
        LOG("NEAT> structural matches=%d", g_structMatchesFound);
    }
#endif

    memmove(genomes[0], nextGenomes[0], sizeof(Genome) * popCount);

    // speciation
    u8 speciesPrevExisted[NEAT_MAX_SPECIES] = {0};
    for(i32 s = 0; s < speciesCount; ++s) {
        speciesPrevExisted[s] = (speciesPopCount[s] != 0);
    }

    Genome* speciesRep = neatSpec->speciesRep;
    mem_zero(neatSpec->speciesPopCount); // reset species population count
    f64 biggestDist = 0.0;

    const f64 c1 = params.compC1;
    const f64 c2 = params.compC2;
    const f64 c3 = params.compC3;
    const f64 compatibilityThreshold = params.compT;

    for(i32 i = 0; i < popCount; ++i) {
        Genome& g = *genomes[i];

        bool found = false;
        for(i32 s = 0; s < speciesCount; ++s) {
            if(!speciesPrevExisted[s] && speciesPopCount[s] == 0) continue;

            f64 dist = compatibilityDistance(g.genes, speciesRep[s].genes, g.geneCount,
                                             speciesRep[s].geneCount, c1, c2, c3);
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

            // find a species slot
            i32 sid = -1;
            for(i32 s = 0; s < NEAT_MAX_SPECIES; ++s) {
                if(!speciesPrevExisted[s] && speciesPopCount[s] == 0) {
                    sid = s;
                    break;
                }
            }
            assert(sid != -1);
            speciesCount = max(speciesCount, sid+1);

            speciesRep[sid] = g;
            speciesPopCount[sid] = 1;
            g.species = sid;
        }
    }

    if(verbose) LOG("NEAT> species count: %d (biggestDist=%g)", speciesCount, biggestDist);

    if(verbose) LOG("NEAT> evolution took %.3fs", timeToMicrosec(timeGet() - t0)/1000000.0);
}

void neatGenomeComputeNodePos(Genome** genomes, const i32 popCount)
{
    for(i32 i = 0; i < popCount; ++i) {
        Genome& genome = *genomes[i];
        Gene* genes = genome.genes;
        const i32 geneCount = genome.geneCount;
        NodePos* nodePos = genome.nodePos;
        i32 nodeProcessedLast = genome.inputNodeCount + genome.outputNodeCount - 1;
        i32 lastHiddenLayer = 0;
        i32 layerNodeCount[NEAT_MAX_LAYERS] = {0};

        for(i32 j = 0; j < genome.inputNodeCount; ++j) {
            nodePos[j].layer = 0;
        }

        for(i32 j = 0; j < geneCount; ++j) {
            Gene& g = genes[j];

            if(g.nodeOut > nodeProcessedLast) {
                nodePos[g.nodeOut].layer = nodePos[g.nodeIn].layer + 1;
                lastHiddenLayer = max(lastHiddenLayer, nodePos[g.nodeOut].layer);
                nodeProcessedLast = g.nodeOut;
            }
        }

        for(i32 j = 0; j < genome.outputNodeCount; ++j) {
            nodePos[j + genome.inputNodeCount].layer = lastHiddenLayer + 1;
        }

        const i32 totalNodeCount = genome.totalNodeCount;
        for(i32 j = 0; j < totalNodeCount; ++j) {
            nodePos[j].vpos = layerNodeCount[nodePos[j].layer]++;
        }
    }
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

f64 neatTestCompability(const Genome* ga, const Genome* gb, const NeatEvolutionParams& params)
{
   return compatibilityDistance(ga->genes, gb->genes, ga->geneCount, gb->geneCount,
                                params.compC1, params.compC2, params.compC3);
}

void neatTestPropagate()
{
    constexpr i32 inputCount = 3;
    constexpr i32 outputCount = 1;

    f64 input[3] = { randf64(0, 1), randf64(0, 1), 1.0 };
    f64 hiddenVal;
    f64 weightItoH[3] = { randf64(-1.0, 1.0), randf64(-1.0, 1.0), randf64(-1.0, 1.0) };
    f64 weightO = randf64(-1.0, 1.0);

    f64 val = 0.0;
    for(i32 i = 0; i < 3; i++) {
        val += input[i] * weightItoH[i];
    }
    hiddenVal = activation(val);

    f64 output = activation(weightO * hiddenVal);

    NeatEvolutionParams params;
    NeatSpeciation neatSpec;
    Genome* g;
    neatGenomeAlloc(&g, 1);
    neatGenomeInit(&g, 1, inputCount, outputCount, params, &neatSpec);

    i32 geneCount = 0;
    for(i16 in = 0; in < inputCount; ++in) {
        for(i16 out = 0; out < outputCount; ++out) {
            const i32 gid = geneCount++;
            g->genes[gid].weight = weightItoH[gid];
            g->genes[gid].nodeOut = 4;
        }
    }
    const i32 gid = g->geneCount++;
    g->genes[gid] = { gid, inputCount+outputCount, inputCount, weightO };
    g->totalNodeCount = inputCount+outputCount+1;

    NeatNN* nn;
    neatGenomeMakeNN(&g, 1, &nn);
    nn->setInputs(input, inputCount);

    for(i32 i = 0; i < 3; ++i) {
        assert(nn->nodeValues[i] == input[i]);
    }

    for(i32 i = 0; i < 3; ++i) {
        assert(nn->computations[i].weight == weightItoH[i]);
    }
    assert(nn->computations[inputCount].weight == weightO);

    neatNnPropagate(&nn, 1);

    assert(output == nn->nodeValues[inputCount]);
}
