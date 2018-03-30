#include "base.h"
#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <gl3w.h>
#include <stdlib.h>
#include <float.h>
#include <assert.h>

#include "neat.h"
#include "neat_imgui.h"
#include "window.h"

#define XOR_COUNT 1024
#define STATS_HISTORY_COUNT 30

struct App {

AppWindow window;

Genome* xorCurGen[XOR_COUNT];
Genome* xorNextGen[XOR_COUNT];
NeatNN* xorNN[XOR_COUNT];
f64 xorFitness[XOR_COUNT];
NeatEvolutionParams evolParam;
NeatSpeciation neatSpec;

struct GenerationStats {
    i32 number = 0;
    f64 maxFitness = 0.0;
    f64 avgFitness = 0.0;
    f64 avgNodeCount = 0.0;
};

i32 generationNumber = 0;
GenerationStats curGenStats;
GenerationStats lastGenStats;
GenerationStats pastGenStats[STATS_HISTORY_COUNT];

f64 outMin = 1.0;
f64 outMax = 0.0;

i32 dbgViewerId = 0;
bool dbgAutoRun = true;
bool dbgAutoSelectBest = true;

bool doneFinding = false;
i32 evolutionTries = 0;

Genome testGen;
NeatNN* testNN;

bool init()
{
    if(!window.init("XOR test", "xor_imgui.ini")) {
        return false;
    }

    neatGenomeAlloc(xorCurGen, XOR_COUNT);
    neatGenomeAlloc(xorNextGen, XOR_COUNT);
    resetSimulation();

    //evolParam.compC2 = 2.0;
    //evolParam.compC3 = 3.0;
    //evolParam.compT = 2.0;
    //evolParam.mutateAddNode = 0.01;

#if 0
    testGen.inputNodeCount = 2;
    testGen.outputNodeCount = 1;
    mem_zero(testGen.layerNodeCount);
    mem_zero(testGen.nodePos);
    mem_zero(testGen.geneDisabled);

    testGen.genes[testGen.geneCount++] = Gene{0, 0, 2, -1.992};
    testGen.geneDisabled[0] = true;
    testGen.genes[testGen.geneCount++] = Gene{1, 1, 2, -0.592};
    testGen.genes[testGen.geneCount++] = Gene{3, 1, 3, -1.187};
    testGen.genes[testGen.geneCount++] = Gene{4, 3, 2, -0.973};
    testGen.genes[testGen.geneCount++] = Gene{48, 1, 4, 0.560};
    testGen.genes[testGen.geneCount++] = Gene{49, 4, 2, 0.311};
    testGen.genes[testGen.geneCount++] = Gene{142, 1, 5, 0.572};
    testGen.genes[testGen.geneCount++] = Gene{143, 5, 2, -1.018};
    testGen.genes[testGen.geneCount++] = Gene{361, 3, 5, -1.191};

    testGen.totalNodeCount = 6;
    testGen.species = 0xdeadbeef;

    Genome* ptr = &testGen;
    neatGenomeMakeNN(&ptr, 1, &testNN, true);
#endif

    return true;
}

void cleanup()
{
    neatGenomeDealloc(xorCurGen);
    neatGenomeDealloc(xorNextGen);
}

void run()
{
    while(window.running) {
        SDL_Event event;
        while(SDL_PollEvent(&event)) {
            window.uiHandleEvent(&event);
            handleEvent(&event);
        }

        newFrame();
        render();
        window.swap();
    }
}

void handleEvent(SDL_Event* event)
{
    if(event->type == SDL_QUIT) {
        window.running = false;
        return;
    }

    if(event->type == SDL_KEYDOWN) {
        if(event->key.keysym.sym == SDLK_ESCAPE) {
            window.running = false;
            return;
        }

        if(event->key.keysym.sym == SDLK_n) {
           resetSimulation();
           return;
        }

        if(event->key.keysym.sym == SDLK_e) {
           nextGeneration();
           return;
        }
    }
}

void resetSimulation()
{
    doneFinding = false;
    dbgAutoSelectBest = true;

    generationNumber = 0;
    curGenStats = {};
    lastGenStats = {};
    memset(pastGenStats, 0, sizeof(pastGenStats));

    neatSpec = {};
    neatGenomeInit(xorCurGen, XOR_COUNT, 2, 1, evolParam, &neatSpec);
    neatGenomeMakeNN(xorCurGen, XOR_COUNT, xorNN);
    neatGenomeComputeNodePos(xorCurGen, XOR_COUNT);
}

void nextGeneration()
{
    if(doneFinding) {
        return;
    }

    if(generationNumber > 200) {
        doneFinding = true;
        return;
    }

    curGenStats = {};
    curGenStats.number = generationNumber++;

    mem_zero(xorFitness);

    u8 correctCount[XOR_COUNT];
    mem_zero(correctCount);

#if 0
    for(i32 i = 0; i < XOR_COUNT; i++) {
        xorFitness[i] = randf64(1.0, 100.0);
    }

#else
    f64 input[4][2] = { {0, 0}, {1, 0}, {0, 1}, {1, 1} };
    i32 expected[4] = { 0, 1, 1, 0};

    const i32 inputCount = xorCurGen[0]->inputNodeCount;

    for(i32 t = 0; t < 4; t++) {
        for(i32 i = 0; i < XOR_COUNT; i++) {
            xorNN[i]->setInputs(input[t], 2);
        }

        neatNnPropagate(xorNN, XOR_COUNT);

        f64 expect = expected[t];
        assert(expect == 0.0 || expect == 1.0);

        // calculate fitness
        for(i32 i = 0; i < XOR_COUNT; i++) {
            f64 nodeValOut = xorNN[i]->nodeValues[inputCount];
            f64 output = (nodeValOut + 1.0) * 0.5;
            //f64 output = nodeValOut;
            assert(output >= 0.0 && output <= 1.0);
            i32 bOut = output < 0.5 ? 0 : 1;
            xorFitness[i] += fabs(expect - output);

            if(expect == bOut) {
                correctCount[i]++;
            }
        }
    }

    for(i32 i = 0; i < XOR_COUNT; i++) {
        xorFitness[i] = 4.0 - xorFitness[i];
        xorFitness[i] = xorFitness[i] * xorFitness[i];
    }
#endif

    f64 totalFitness = 0.0;
    i64 totalNodeCount = 0;
    f64 maxFitness = 0.0;
    for(i32 i = 0; i < XOR_COUNT; i++) {
        totalFitness += xorFitness[i];
        maxFitness = max(maxFitness, xorFitness[i]);
        totalNodeCount += xorCurGen[i]->totalNodeCount;
    }

    curGenStats.avgFitness = totalFitness / XOR_COUNT;
    curGenStats.maxFitness = maxFitness;
    curGenStats.avgNodeCount = totalNodeCount / (f64)XOR_COUNT;

    lastGenStats = curGenStats;
    memmove(pastGenStats, pastGenStats+1, sizeof(pastGenStats) - sizeof(pastGenStats[0]));
    pastGenStats[STATS_HISTORY_COUNT-1] = lastGenStats;

    for(i32 i = 0; i < XOR_COUNT; i++) {
        if(correctCount[i] == 4) {
            doneFinding = true;
            LOG("DONE FINDING (%d is a valid xor gate)", i);
            dbgViewerId = i;
            dbgAutoSelectBest = false;
            return;
        }
    }

    LOG("evolution %d ----------", generationNumber);
    neatEvolve(xorCurGen, xorNextGen, xorFitness, XOR_COUNT, &neatSpec, evolParam, true);
    neatGenomeMakeNN(xorCurGen, XOR_COUNT, xorNN);
    neatGenomeComputeNodePos(xorCurGen, XOR_COUNT);
}

void ui_xorViewer()
{
    ImGui::Begin("XOR viewer");

    ImGui::Checkbox("Auto run simulation", &dbgAutoRun);

    ImGui::Checkbox("Auto-select best", &dbgAutoSelectBest);

    if(dbgAutoSelectBest) {
        f64 bestFitness = 0;
        for(i32 i = 0; i < XOR_COUNT; ++i) {
            if(xorFitness[i] > bestFitness) {
                dbgViewerId = i;
                bestFitness = xorFitness[i];
            }
        }
    }
    else {
        ImGui::PushItemWidth(-1);
        ImGui::SliderInt("##xor_id", &dbgViewerId, 0, XOR_COUNT-1);
        ImGui::InputInt("##xor_id_input", &dbgViewerId);
        ImGui::PopItemWidth();

        if(ImGui::Button("Next of same species")) {
            const i32 species = xorCurGen[dbgViewerId]->species;
            for(i32 i = 1; i < XOR_COUNT; ++i) {
                i32 id = (dbgViewerId + i) % XOR_COUNT;
                if(xorCurGen[id]->species == species) {
                    dbgViewerId = id;
                    break;
                }
            }
        }
    }

    ImGui::Separator();

    ImGui::Text("XOR_%d", dbgViewerId);
    ImGui::TextColored(ImVec4(1, 0, 1, 1), "Species: %X", xorCurGen[dbgViewerId]->species);
    ImGui::TextColored(ImVec4(0, 1, 0, 1), "Fitness: %g", xorFitness[dbgViewerId]);

    ImGui::Separator();

    ImGui_NeatGeneList(xorCurGen[dbgViewerId]);

    ImGui::Separator();

    ImGui_NeatNN(xorCurGen[dbgViewerId]);

    if(ImGui::CollapsingHeader("Propgate")) {
        static i32 input1 = 0;
        static i32 input2 = 1;

        ImGui::PushItemWidth(200);
        ImGui::InputInt("##xor_input1", &input1); ImGui::SameLine();
        ImGui::InputInt("##xor_input2", &input2); ImGui::SameLine();

        if(ImGui::Button("Propagate")) {
            f64 inputs[2] = {(f64)input1, (f64)input2};
            xorNN[dbgViewerId]->setInputs(inputs, 2);
            neatNnPropagate(&xorNN[dbgViewerId], 1);
        }
        ImGui::PopItemWidth();

        f64 out = (xorNN[dbgViewerId]->nodeValues[2] + 1.0) * 0.5;
        ImGui::Text("output: %g (%d)", out, out < 0.5 ? 0 : 1);
    }

    static i32 versusId1 = 0;
    static i32 versusId2 = 1;
    const i32 outId = XOR_COUNT-1;
    if(ImGui::CollapsingHeader("Test")) {
        ImGui::PushItemWidth(200);
        ImGui::InputInt("##xor_vs_1", &versusId1); ImGui::SameLine();
        ImGui::InputInt("##xor_vs_2", &versusId2);
        ImGui::PopItemWidth();

        if(ImGui::Button("Reproduce")) {
            neatTestTryReproduce(*xorCurGen[versusId1], *xorCurGen[versusId2]);
        }

        ImGui::SameLine();

        if(ImGui::Button("Crossover")) {
            neatTestCrossover(xorCurGen[versusId1], xorCurGen[versusId2], xorCurGen[outId]);
        }

        ImGui_NeatGeneList(xorCurGen[outId]);
        ImGui_NeatNN(xorCurGen[outId]);
    }

    if(ImGui::CollapsingHeader("Compatibility")) {
        ImGui::PushItemWidth(200);
        ImGui::InputInt("##comp_vs_1", &versusId1); ImGui::SameLine();
        ImGui::InputInt("##comp_vs_2", &versusId2);
        ImGui::PopItemWidth();

        ImGui::TextColored(ImVec4(0, 1, 1, 1), "Compatibility: %g",
                           neatTestCompability(xorCurGen[versusId1], xorCurGen[versusId2], evolParam));
    }

    ImGui::End();
}

#if 0
void ui_textXorViewer()
{
    ImGui::Begin("Test XOR viewer");

    ImGui_GeneList(&testGen);

    ImGui::Separator();

    ImGui_NeatNN(&testGen);

    if(ImGui::CollapsingHeader("Propgate")) {
        static i32 input1 = 0;
        static i32 input2 = 1;

        ImGui::PushItemWidth(200);
        ImGui::InputInt("##xor_input1", &input1); ImGui::SameLine();
        ImGui::InputInt("##xor_input2", &input2); ImGui::SameLine();

        if(ImGui::Button("Propagate")) {
            testNN->nodeValues[0] = input1;
            testNN->nodeValues[1] = input2;
            neatNnPropagate(&testNN, 1);
        }
        ImGui::PopItemWidth();

        f64 out = (testNN->nodeValues[2] + 1.0) * 0.5;
        ImGui::Text("output: %g (%d)", out, out < 0.5 ? 0 : 1);
    }

    ImGui::End();
}
#endif

void ui_subPopulations()
{

}

void ui_lastGeneration()
{
    // last generation statistics window
    ImGui::Begin("Last generation");

    ImGui::Text("Generation %d", lastGenStats.number);
    ImGui::Separator();

    ImGui::TextColored(ImVec4(0, 1, 0, 1), "Fitness max: %g", lastGenStats.maxFitness);
    ImGui::Text("Fitness avg: %g", lastGenStats.avgFitness);
    ImGui::TextColored(ImVec4(1, 1, 0, 1), "Node count avg: %g", lastGenStats.avgNodeCount);

    ImGui::Separator();

    f32 pastFitness[STATS_HISTORY_COUNT];
    f32 maxPastFitness = 0;
    for(i32 i = 0; i < STATS_HISTORY_COUNT; ++i) {
        pastFitness[i] = pastGenStats[i].avgFitness;
        maxPastFitness = max(maxPastFitness, pastFitness[i]);
    }

    ImGui::PushItemWidth(-1);
    ImGui::PlotLines("##max_fitness", pastFitness, IM_ARRAYSIZE(pastFitness),
                         0, NULL, 0.0f, maxPastFitness, ImVec2(0,50));

    ImGui::End();
}

void doUI()
{
    ImGui::StyleColorsDark();

    ui_xorViewer();
    ui_subPopulations();
    ui_lastGeneration();

    //ImGui::ShowDemoWindow();
}

void updateNNs()
{
    if(dbgAutoRun) {
      nextGeneration();
    }
}

void newFrame()
{
    window.uiNewFrame();
    doUI();

    updateNNs();
}

void render()
{
    glClear(GL_COLOR_BUFFER_BIT);

    window.uiRender();
}

};

#ifdef _WIN32
int CALLBACK WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
#else
i32 main()
#endif
{
    LOG("XOR test\n");

    timeInit();
    randSetSeed(time(NULL));

    SDL_SetMainReady();
    i32 sdl = SDL_Init(SDL_INIT_VIDEO);
    if(sdl != 0) {
        LOG("ERROR: could not init SDL2 (%s)", SDL_GetError());
        return 1;
    }

    App app;

    if(!app.init()) {
        return 1;
    }

    app.run();

    app.cleanup();
    SDL_Quit();
    return 0;
}
