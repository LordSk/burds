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

Genome* xorGenome[XOR_COUNT];
f64 xorFitness[XOR_COUNT];

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
    if(!window.init("XOR test")) {
        return false;
    }

    neatGenomeAlloc(xorGenome, XOR_COUNT);
    resetSimulation();
    return true;
}

void cleanup()
{
    neatGenomeDealloc(xorGenome[0]);
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
        }
    }
}

void resetSimulation()
{
    generationNumber = 0;
    curGenStats = {};
    memset(pastGenStats, 0, sizeof(pastGenStats));

    neatGenomeInit(xorGenome, XOR_COUNT, 2, 1);
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

void ImGui_NeuralNet(NeuralNet* nn, NeuralNetDef* def)
{
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    if (window->SkipItems)
        return;

    constexpr i32 cellsPerLine = 10;
    const ImVec2 cellSize(10, 10);
    i32 lines = def->neuronCount / cellsPerLine + 1;
    ImVec2 size(cellsPerLine * cellSize.x, lines * cellSize.y);

    ImVec2 pos = window->DC.CursorPos;
    const ImRect bb(pos, pos + size);
    ImGui::ItemSize(bb);

    for(i32 i = 0; i < def->neuronCount; ++i) {
        f32 w = clamp(nn->values[i] * 0.5, 0.0, 1.0);
        u32 color = 0xff000000 | ((u8)(0xff*w) << 16)| ((u8)(0xff*w) << 8)| ((u8)(0xff*w));
        i32 column = i % cellsPerLine;
        i32 line = i / cellsPerLine;
        ImVec2 offset(column * cellSize.x, line * cellSize.y);
        ImGui::RenderFrame(pos + offset, pos + offset + cellSize, color, false, 0);
    }
}

void ImGui_Gene(const Gene& gene)
{
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    ImVec2 size(50, 45);
    ImVec2 pos = window->DC.CursorPos;
    const ImRect bb(pos, pos + size);
    ImGui::ItemSize(bb);

    f64 blend = (clamp(gene.weight, -1.0, 1.0) + 1.0) * 0.5;
    ImVec4 color(1.0 - blend, blend, 0.2, 1.0);
    if(gene.disabled) {
        color = ImVec4(0.4, 0.4, 0.4, 1.0);
    }

    u32 bgColor = ImGui::ColorConvertFloat4ToU32(color);
    ImGui::RenderFrame(pos, pos + size, bgColor, false, 0.0);
    ImGui::RenderFrame(pos, pos + ImVec2(50, 15), 0x80000000, false, 0.0);

    char inNumStr[10];
    char linkStr[32];
    char weightStr[32];

    sprintf(inNumStr, "#%d", gene.innovationNumber);
    sprintf(linkStr, "%d > %d", gene.nodeIn, gene.nodeOut);
    sprintf(weightStr, "%.3f", gene.weight);

    pos.x += 2;
    ImGui::RenderText(pos, inNumStr);

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0,0,0,1));
    ImGui::RenderText(pos + ImVec2(0, 15), linkStr);
    ImGui::RenderText(pos + ImVec2(0, 30), weightStr);
    ImGui::PopStyleColor(1);
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
        ImGui::PopItemWidth();
    }

    ImGui::Separator();

    ImGui::Text("XOR_%d", dbgViewerId);
    ImGui::TextColored(ImVec4(0, 1, 0, 1), "fitness: %g", xorFitness[dbgViewerId]);

    ImGui::Separator();

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(1,1));

    const i32 geneCount = xorGenome[dbgViewerId]->geneCount;
    const i32 perLine = 6;
    for(i32 i = 0; i < geneCount; ++i) {
        ImGui_Gene(xorGenome[dbgViewerId]->genes[i]);
        if(i % perLine == 0) {
            ImGui::SameLine();
        }
    }

    ImGui::PopStyleVar(1);

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
