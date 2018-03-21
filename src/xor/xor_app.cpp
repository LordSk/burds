#include "base.h"
#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <gl3w.h>
#include <stdlib.h>
#include <float.h>
#include <assert.h>

#include "neat.h"
#include "window.h"

#define XOR_COUNT 1024
#define STATS_HISTORY_COUNT 30

struct App {

AppWindow window;

Genome* xorCurgen[XOR_COUNT];
Genome* xorNextGen[XOR_COUNT];
NeatNN* xorNN[XOR_COUNT];
f64 xorFitness[XOR_COUNT];
NeatEvolutionParams evolParam;

struct GenerationStats {
    i32 number = 0;
    f64 maxFitness = 0.0;
    f64 avgFitness = 0.0;
};

i32 generationNumber = 0;
GenerationStats curGenStats;
GenerationStats lastGenStats;
GenerationStats pastGenStats[STATS_HISTORY_COUNT];

f64 outMin = 1.0;
f64 outMax = 0.0;

i32 dbgViewerId = 0;

bool init()
{
    if(!window.init("XOR test", "xor_imgui.ini")) {
        return false;
    }

    neatGenomeAlloc(xorCurgen, XOR_COUNT);
    neatGenomeAlloc(xorNextGen, XOR_COUNT);
    resetSimulation();

    for(i32 i = 0; i < XOR_COUNT; ++i) {
        xorFitness[i] = randf64(1.0, 10.0);
    }

    evolParam.compT = 0.3;

    return true;
}

void cleanup()
{
    neatGenomeDealloc(xorCurgen[0]);
    neatGenomeDealloc(xorNextGen[0]);
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
    generationNumber = 0;
    curGenStats = {};
    memset(pastGenStats, 0, sizeof(pastGenStats));

    neatGenomeInit(xorCurgen, XOR_COUNT, 3, 1);
    neatGenomeMakeNN(xorCurgen, XOR_COUNT, xorNN);
}

void nextGeneration()
{
    // calculate fitness
    for(i32 i = 0; i < XOR_COUNT; i++) {
        xorNN[i]->nodeValues[0] = randi64(0, 1);
        xorNN[i]->nodeValues[1] = randi64(0, 1);
        xorNN[i]->nodeValues[2] = 1; // bias
    }

    neatNnPropagate(xorNN, XOR_COUNT);

    for(i32 i = 0; i < XOR_COUNT; i++) {
        f64 expected = (i32)xorNN[i]->nodeValues[0] ^ (i32)xorNN[i]->nodeValues[1];
        f64 output = (xorNN[i]->nodeValues[3] + 1.0) * 0.5;
        assert(expected == 0.0 || expected == 1.0);
        assert(output >= 0.0 && output <= 1.0);
        xorFitness[i] = 1000.0 * (1.0 - fabs(expected - output));
    }

    neatEvolve(xorCurgen, xorNextGen, xorFitness, XOR_COUNT, evolParam);
    neatGenomeMakeNN(xorCurgen, XOR_COUNT, xorNN);
}

void ImGui_ColoredRect(const ImVec2& size, const ImVec4& color)
{
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    if (window->SkipItems)
        return;

    ImVec2 pos = window->DC.CursorPos;
    const ImRect bb(pos, pos + size);
    ImGui::ItemSize(bb);

    ImGui::RenderFrame(bb.Min, bb.Max, ImGui::ColorConvertFloat4ToU32(color), false, 0);
}

void ImGui_Gene(const Gene& gene, bool disabled)
{
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    ImVec2 size(50, 45);
    ImVec2 pos = window->DC.CursorPos;
    const ImRect bb(pos, pos + size);
    ImGui::ItemSize(bb);

    f64 blend = (clamp(gene.weight, -1.0, 1.0) + 1.0) * 0.5;
    ImVec4 color(1.0 - blend, blend, 0.2, 1.0);
    if(disabled) {
        color = ImVec4(0.4, 0.4, 0.4, 1.0);
    }

    u32 bgColor = ImGui::ColorConvertFloat4ToU32(color);
    ImGui::RenderFrame(pos, pos + size, bgColor, false, 0.0);
    ImGui::RenderFrame(pos, pos + ImVec2(50, 15), 0x80000000, false, 0.0);

    char inNumStr[10];
    char linkStr[32];
    char weightStr[32];

    sprintf(inNumStr, "#%d", gene.historicalMarker);
    sprintf(linkStr, "%d > %d", gene.nodeIn, gene.nodeOut);
    sprintf(weightStr, "%.3f", gene.weight);

    pos.x += 2;
    ImGui::RenderText(pos, inNumStr);

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0,0,0,1));
    ImGui::RenderText(pos + ImVec2(0, 15), linkStr);
    ImGui::RenderText(pos + ImVec2(0, 30), weightStr);
    ImGui::PopStyleColor(1);
}

void ImGui_GeneList(const Genome* genome)
{
    ImGui::BeginChild((ImGuiID)(intptr_t)genome, ImVec2(300, 140));

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(1,1));
    const i32 geneCount = genome->geneCount;
    const i32 perLine = 5;
    for(i32 i = 0; i < geneCount; ++i) {
        ImGui_Gene(genome->genes[i], genome->geneDisabled[i]);
        if(((i+1) % perLine) != 0 && i != geneCount-1) {
            ImGui::SameLine();
        }
    }
    ImGui::PopStyleVar(1);

    ImGui::EndChild();
}

void ImGui_NeatNN(const Genome* genome)
{
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const Genome& g = *genome;
    const ImVec2 nodeSpace(30, 30);
    const f32 linkSpaceWidth = 60.f;
    const f32 nodeRadius = 12.f;

    i32 maxVPos = 0;
    for(i32 i = 0; i < g.totalNodeCount; ++i) {
        maxVPos = max(maxVPos, g.nodePos[i].vpos);
    }
    maxVPos++;

    ImVec2 frameSize(g.layerCount * (linkSpaceWidth + nodeSpace.x), nodeSpace.y * maxVPos);
    ImVec2 pos = window->DC.CursorPos;
    ImRect bb(pos, pos + frameSize);
    ImGui::ItemSize(bb);

    // compute each node position
    ImVec2 nodePos[128];
    assert(g.totalNodeCount <= 128);

    for(i32 i = 0; i < g.totalNodeCount; ++i) {
        nodePos[i] = pos + ImVec2(nodeSpace.x * 0.5 + (linkSpaceWidth + nodeSpace.x) * g.nodePos[i].layer,
                                  nodeSpace.y * g.nodePos[i].vpos + nodeSpace.y * 0.5f);
    }

    // draw lines
    for(i32 i = 0; i < g.geneCount; ++i) {
        if(g.geneDisabled[i]) continue;
        f64 blend = (clamp(g.genes[i].weight, -1.0, 1.0) + 1.0) * 0.5;
        ImVec4 color4(1.0 - blend, blend, 0.2, 1.0);
        u32 lineCol = ImGui::ColorConvertFloat4ToU32(color4);
        draw_list->AddLine(nodePos[g.genes[i].nodeIn], nodePos[g.genes[i].nodeOut], lineCol, 2.0);
    }

    // draw nodes
    char numStr[5];
    for(i32 i = 0; i < g.totalNodeCount; ++i) {
        draw_list->AddCircleFilled(nodePos[i], nodeRadius, 0xffffffffff, 32);
        sprintf(numStr, "%d", i);
        ImVec2 labelSize = ImGui::CalcTextSize(numStr);
        draw_list->AddText(nodePos[i] + ImVec2(-labelSize.x * 0.5, -labelSize.y * 0.5), 0xFF000000, numStr);
    }
}

void ui_xorViewer()
{
    ImGui::Begin("XOR viewer");

    static bool autoSelectBest = true;
    ImGui::Checkbox("Auto-select best", &autoSelectBest);

    if(autoSelectBest) {
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
            const i32 species = xorCurgen[dbgViewerId]->species;
            for(i32 i = 1; i < XOR_COUNT; ++i) {
                i32 id = (dbgViewerId + i) % XOR_COUNT;
                if(xorCurgen[id]->species == species) {
                    dbgViewerId = id;
                    break;
                }
            }
        }
    }

    ImGui::Separator();

    ImGui::Text("XOR_%d", dbgViewerId);
    ImGui::TextColored(ImVec4(1, 0, 1, 1), "Species: %X", xorCurgen[dbgViewerId]->species);
    ImGui::TextColored(ImVec4(0, 1, 0, 1), "Fitness: %g", xorFitness[dbgViewerId]);

    ImGui::Separator();

    ImGui_GeneList(xorCurgen[dbgViewerId]);

    ImGui::Separator();

    ImGui_NeatNN(xorCurgen[dbgViewerId]);


    static i32 versusId1 = 0;
    static i32 versusId2 = 1;
    const i32 outId = XOR_COUNT-1;
    if(ImGui::CollapsingHeader("Test")) {
        ImGui::PushItemWidth(200);
        ImGui::InputInt("##xor_vs_1", &versusId1); ImGui::SameLine();
        ImGui::InputInt("##xor_vs_2", &versusId2);
        ImGui::PopItemWidth();

        if(ImGui::Button("Reproduce")) {
            neatTestTryReproduce(*xorCurgen[versusId1], *xorCurgen[versusId2]);
        }

        ImGui::SameLine();

        if(ImGui::Button("Crossover")) {
            neatTestCrossover(xorCurgen[versusId1], xorCurgen[versusId2], xorCurgen[outId]);
        }

        ImGui_GeneList(xorCurgen[outId]);
        ImGui_NeatNN(xorCurgen[outId]);
    }

    ImGui::End();
}

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

    ImGui::Separator();

    f32 pastFitness[STATS_HISTORY_COUNT];
    f32 maxPastFitness = 0;
    for(i32 i = 0; i < STATS_HISTORY_COUNT; ++i) {
        pastFitness[i] = pastGenStats[i].avgFitness;
        maxPastFitness = max(maxPastFitness, pastFitness[i]);
    }

    ImGui::PushItemWidth(180.0f);
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

}

void newGeneration()
{
#if 0
    LOG("outMin=%g outMax=%g", outMin, outMax);
    outMin = 1.0;
    outMax = 0.0;

    lastGenStats = curGenStats;
    memmove(pastGenStats, pastGenStats+1, sizeof(pastGenStats) - sizeof(pastGenStats[0]));
    pastGenStats[STATS_HISTORY_COUNT-1] = lastGenStats;

    curGenStats = {};
    curGenStats.number = generationNumber++;

    LOG("#%d maxFitness=%.5f avg=%.5f", lastGenStats.number, lastGenStats.maxFitness,
        lastGenStats.avgFitness);

    resetMap();
    resetFrogs();
#endif
}

void newFrame()
{
    window.uiUpdate();
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
