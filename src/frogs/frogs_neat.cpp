#include "base.h"
#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <gl3w.h>
#include <stdlib.h>
#include <float.h>
#include <assert.h>

#include "window.h"
#include "sprite.h"
#include "neat.h"
#include "neat_imgui.h"
#include "imgui/imgui.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui/imgui_internal.h"
#include "imgui/imgui_sdl2_setup.h"

#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 900
#define FRAMES_PER_SEC 60.0
#define FRAME_DT ((f64)(1.0/FRAMES_PER_SEC))

#define MAP_WIDTH 256
#define MAP_HEIGHT 128
#define MAP_WATER_AVG_GRID_SIZE 8
static_assert(MAP_WIDTH % MAP_WATER_AVG_GRID_SIZE == 0,"");
static_assert(MAP_HEIGHT % MAP_WATER_AVG_GRID_SIZE == 0,"");
constexpr i32 MAP_SIZE = MAP_WIDTH*MAP_HEIGHT;
constexpr i32 MAP_WATER_AVG_WIDTH = MAP_WIDTH/MAP_WATER_AVG_GRID_SIZE;
constexpr i32 MAP_WATER_AVG_HEIGHT = MAP_HEIGHT/MAP_WATER_AVG_GRID_SIZE;
constexpr i32 MAP_WATER_AVG_SIZE = MAP_WATER_AVG_WIDTH*MAP_WATER_AVG_HEIGHT;
#define TILE_SIZE 32
#define MAP_POND_MAX_COUNT 24
#define MAP_POND_MIN_RADIUS 12
#define MAP_POND_MAX_RADIUS 40

#define FROG_COUNT 512

#define FROG_SIZE 60.0f
#define FROG_SPEED 400.0f
#define FROG_ANIM_JUMP 300
#define FROG_ANIM_TONGUE 200
#define FROG_JUMP_CD 450
#define FROG_TONGUE_CD 300
#define FROG_WATER_SENSOR_OFFSET (FROG_SIZE + 100.f)

#define ENERGY_TOTAL 100.0f
#define ENERGY_JUMP_DRAIN 2.0f
#define ENERGY_EAT_DRAIN 5.0f

#define HYDRATION_TOTAL 100.0f
#define HYDRATION_DRAIN_PER_SEC 4.0f
#define HYDRATION_GAIN_PER_SEC 20.0f

#define STATS_HISTORY_COUNT 30

#define SIMULATION_MAX_TIME 30.0

//#define VISION_WIDTH 32 // is a square

enum {
    MAP_TILE_GRASS=0,
    MAP_TILE_WATER,
    MAP_TILE_DEATH,
};

enum {
    INPUT_ACTION_STILL=0,
    INPUT_ACTION_JUMP,
    INPUT_ACTION_EAT,
};

enum {
    FROG_TEX_STILL=0,
    FROG_TEX_JUMP,
    FROG_TEX_TONGUE,
    FROG_TEX_DEAD,
    FROG_TEX_COUNT,
};

struct FrogInput
{
    f32 angle;
    i32 action; // 0: nothing, 1: jump, 2: tongue
};

struct WaterSensors
{
    f32 center;
    f32 sens[4];
};

struct App {

AppWindow window;
i32 timeScale = 1;

f32 viewZoom = 5.0f;
i32 viewX = 0;
i32 viewY = 0;
u8 mouseRightButDown;

u8 mapData[MAP_SIZE];
f32 mapAvgWater[MAP_WATER_AVG_SIZE];
u32 mapTextureData[MAP_SIZE];
u32 mapAvgWaterTextureData[MAP_WATER_AVG_SIZE];
u32 tex_map;
u32 tex_mapAvgWater;

i32 pondPos[MAP_POND_MAX_COUNT];
i32 pondRadius[MAP_POND_MAX_COUNT];
i32 pondCount = 4;

i32 tex_frog[FROG_TEX_COUNT];

Vec2 frogPos[FROG_COUNT];
f32 frogAngle[FROG_COUNT];
Transform frogTf[FROG_COUNT];
Color3 speciesColor[FROG_COUNT];
u8 frogTexId[FROG_COUNT]; // still, jump, tongue
FrogInput frogInput[FROG_COUNT];
f64 frogJumpTime[FROG_COUNT];
f64 frogTongueTime[FROG_COUNT];
WaterSensors frogWaterSensors[FROG_COUNT];
f64 frogClosestPondFactor[FROG_COUNT];
i8 frogClosestPondFactorOffsetSign[FROG_COUNT];
f32 frogClosestPondAngleDiff[FROG_COUNT];
f64 frogClosestDeathBondFactor[FROG_COUNT];

f32 frogEnergy[FROG_COUNT];
f32 frogHydration[FROG_COUNT];
u8 frogDead[FROG_COUNT];

f64 frogFitness[FROG_COUNT];
NeatEvolutionParams evolParam;
NeatSpeciation neatSpec;

Genome* frogCurGen[FROG_COUNT];
Genome* frogNextGen[FROG_COUNT];
NeatNN* frogNN[FROG_COUNT];
i32 generationNumber = 0;

f32 simulationTime = 0;

bool dbgShowBars = false;
bool dbgShowDirections = false;
bool dbgOverlayAvgWaterMap = false;
i32 dbgViewerFrogId = 0;
Vec2 dbgLastMouseLeftClick = {0, 0};

struct GenerationStats {
    i32 number = 0;
    f64 maxFitness = 0.0;
    f64 avgFitness = 0.0;
    i32 deathsByBorder = 0;
    i32 deathsByDehydratation = 0;
};

GenerationStats curGenStats;
GenerationStats lastGenStats;
GenerationStats pastGenStats[STATS_HISTORY_COUNT];

bool init()
{
    if(!window.init("Frogs [NEAT]", "frogs_neat_imgui.ini")) {
        return false;
    }

    glClearColor(0.2, 0.2, 0.2, 1.0f);

    tex_frog[FROG_TEX_STILL]  = loadTexture("../frog_still.png");
    tex_frog[FROG_TEX_JUMP]   = loadTexture("../frog_jump.png");
    tex_frog[FROG_TEX_TONGUE] = loadTexture("../frog_tongue.png");
    tex_frog[FROG_TEX_DEAD]   = loadTexture("../frog_dead.png");

    for(i32 i = 0; i < FROG_TEX_COUNT; ++i) {
        if(tex_frog[i] == -1) {
            return false;
        }
    }

    glGenTextures(1, &tex_map);
    glGenTextures(1, &tex_mapAvgWater);

    resetMap();
    resetFrogColors();

    evolParam.compC1 = 2.0;
    evolParam.compC2 = 2.0;
    evolParam.compC3 = 3.0;
    evolParam.compT = 2.0;
    //evolParam.mutateAddConn = 0.05;
    evolParam.mutateAddNode = 0.1;
    evolParam.mutateWeightStep = 0.5;

    neatGenomeAlloc(frogCurGen, FROG_COUNT);
    neatGenomeAlloc(frogNextGen, FROG_COUNT);
    resetSimulation();

    return true;
}

void cleanup()
{
    neatGenomeDealloc(frogCurGen);
    neatGenomeDealloc(frogNextGen);
    neatNnDealloc(frogNN);
}

void run()
{
    while(window.running) {
        timept t0 = timeGet();

        SDL_Event event;
        while(SDL_PollEvent(&event)) {
            window.uiHandleEvent(&event);
            handleEvent(&event);
        }

        updateCamera();

        newFrame();
        render();
        window.swap();

        const i64 frameDtMicro = FRAME_DT/timeScale * 1000000;
        while(((frameDtMicro - timeToMicrosec(timeGet() - t0)) / 1000) > 1) {
            _mm_pause();
        }
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

        if(event->key.keysym.sym == SDLK_r) {
           resetMap();
           return;
        }

        if(event->key.keysym.sym == SDLK_n) {
           resetSimulation();
           return;
        }
    }

    if(event->type == SDL_MOUSEBUTTONDOWN) {
        if(event->button.button == SDL_BUTTON_RIGHT) {
            mouseRightButDown = true;
            return;
        }
        if(event->button.button == SDL_BUTTON_LEFT) {
            mouseSelectFrog(event->button.x, event->button.y);
            return;
        }
    }

    if(event->type == SDL_MOUSEBUTTONUP) {
        if(event->button.button == SDL_BUTTON_RIGHT) {
            mouseRightButDown = false;
        }
    }

    if(event->type == SDL_MOUSEMOTION) {
        if(mouseRightButDown) {
            viewX -= event->motion.xrel * max(viewZoom, 1.f);
            viewY -= event->motion.yrel * max(viewZoom, 1.f);
        }
    }

    if(event->type == SDL_MOUSEWHEEL) {
        if(event->wheel.y > 0) {
            viewZoom *= 0.90f * event->wheel.y;
        }
        else if(event->wheel.y < 0) {
            viewZoom *= 1.10f * -event->wheel.y;
        }
    }
}

void mouseSelectFrog(f32 mx, f32 my)
{
    f32 x = viewX + mx * viewZoom;
    f32 y = viewY + my * viewZoom;
    Vec2 mWorldPos = {x, y};

    f32 closestDist = FLT_MAX;
    i32 closestId = 0;

    for(i32 i = 0; i < FROG_COUNT; ++i) {
        f32 d = vec2Distance(&frogPos[i], &mWorldPos);
        if(d < closestDist) {
            closestDist = d;
            closestId = i;
        }
    }

    if(closestDist < 50) {
        dbgViewerFrogId = closestId;
    }

    dbgLastMouseLeftClick.x = x;
    dbgLastMouseLeftClick.y = y;
}

void updateCamera()
{
    if(mouseRightButDown) {
        SDL_SetRelativeMouseMode((SDL_bool)TRUE);
    }
    else {
        SDL_SetRelativeMouseMode((SDL_bool)FALSE);
    }

    setView(viewX, viewY, WINDOW_WIDTH*viewZoom, WINDOW_HEIGHT*viewZoom);
}

inline u32 noiseColor(Color3 baseColor, u8 variance)
{
    u8 r = clamp((i32)baseColor.r + (i32)randi64(-variance, variance), 0, 0xff);
    u8 g = clamp((i32)baseColor.g + (i32)randi64(-variance, variance), 0, 0xff);
    u8 b = clamp((i32)baseColor.b + (i32)randi64(-variance, variance), 0, 0xff);
    return (0xff000000 | (b << 16) | (g << 8) | r);
}

void resetMap()
{
    memset(mapData, MAP_TILE_GRASS, sizeof(mapData));

    for(i32 i = 0; i < pondCount; ++i) {
        pondPos[i] = xorshift64star() % MAP_SIZE;
        pondRadius[i] = randi64(MAP_POND_MIN_RADIUS, MAP_POND_MAX_RADIUS);
        //LOG("pond#%d pos=%d radius=%d", i, pondPos[i], pondRadius[i]);
    }

    for(i32 i = 0; i < MAP_SIZE; ++i) {
        i32 x = i % MAP_WIDTH;
        i32 y = i / MAP_WIDTH;

        for(i32 p = 0; p < pondCount; ++p) {
            i32 px = pondPos[p] % MAP_WIDTH;
            i32 py = pondPos[p] / MAP_WIDTH;
            i32 dist = sqrtf((px - x) * (px - x) + (py - y) * (py - y));

            if(dist <= pondRadius[p]) {
                mapData[i] = MAP_TILE_WATER;
            }
        }
    }

    for(i32 i = 0; i < MAP_SIZE; ++i) {
        i32 x = i % MAP_WIDTH;
        i32 y = i / MAP_WIDTH;

        if(x == 0) {
            mapData[i] = MAP_TILE_DEATH;
        }
        else if(x == MAP_WIDTH-1) {
            mapData[i] = MAP_TILE_DEATH;
        }
        else if(y == 0) {
            mapData[i] = MAP_TILE_DEATH;
        }
        else if(y == MAP_HEIGHT-1) {
            mapData[i] = MAP_TILE_DEATH;
        }
    }

    memset(mapAvgWater, 0, sizeof(mapAvgWater));
    for(i32 i = 0; i < MAP_SIZE; ++i) {
        i32 maX = (i % MAP_WIDTH) / MAP_WATER_AVG_GRID_SIZE;
        i32 maY = (i / MAP_WIDTH) / MAP_WATER_AVG_GRID_SIZE;
        mapAvgWater[maY * MAP_WATER_AVG_WIDTH + maX] += mapData[i] == MAP_TILE_WATER ? 1.0f : 0.0f;
    }

    for(i32 i = 0; i < MAP_WATER_AVG_SIZE; ++i) {
        mapAvgWater[i] /= (f32)MAP_WATER_AVG_GRID_SIZE * MAP_WATER_AVG_GRID_SIZE;
    }

    const Color3 grassColor = {10, 50, 0};
    const Color3 waterColor = {0, 157, 255};
    const Color3 deathColor = {0, 0, 0};
    const i32 grassVariance = 4;
    const i32 waterVariance = 10;
    const i32 deathVariance = 10;
    for(i32 i = 0; i < MAP_SIZE; ++i) {
        switch(mapData[i]) {
            case MAP_TILE_GRASS: mapTextureData[i] = noiseColor(grassColor, grassVariance); break;
            case MAP_TILE_WATER: mapTextureData[i] = noiseColor(waterColor, waterVariance); break;
            case MAP_TILE_DEATH: mapTextureData[i] = noiseColor(deathColor, deathVariance); break;
        }
    }

    for(i32 i = 0; i < MAP_WATER_AVG_SIZE; ++i) {
        mapAvgWaterTextureData[i] = 0xffff0000 | ((u32)(0xff * (1.0f - mapAvgWater[i])) << 8) |
                ((u32)(0xff * (1.0f - mapAvgWater[i])));
    }

    glBindTexture(GL_TEXTURE_2D, tex_map);

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA8,
                 MAP_WIDTH, MAP_HEIGHT,
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 mapTextureData);

    glBindTexture(GL_TEXTURE_2D, tex_mapAvgWater);

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA8,
                 MAP_WATER_AVG_WIDTH, MAP_WATER_AVG_HEIGHT,
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 mapAvgWaterTextureData);
}

void resetFrogColors()
{
    const u32 colorMax = 0xFF;
    const u32 colorMin = 0x0;
    const u32 colorDelta = colorMax - colorMin;

    for(i32 i = 0; i < FROG_COUNT; ++i) {
        Color3 c;
        c.r = (xorshift64star() % colorDelta) + colorMin;
        c.g = (xorshift64star() % colorDelta) + colorMin;
        c.b = (xorshift64star() % colorDelta) + colorMin;
        speciesColor[i] = c;
    }
}

void resetFrogs()
{
    simulationTime = 0;

    memset(frogAngle, 0, sizeof(frogAngle));
    memset(frogJumpTime, 0, sizeof(frogJumpTime));
    memset(frogTongueTime, 0, sizeof(frogTongueTime));
    memset(frogDead, 0, sizeof(frogDead));
    memset(frogFitness, 0, sizeof(frogFitness));

    for(i32 i = 0; i < FROG_COUNT; ++i) {
        bool onWater = true;
        while(onWater) {
            frogPos[i] = vec2Make(randf64(TILE_SIZE + 1.0, (MAP_WIDTH-1) * TILE_SIZE - 1),
                                  randf64(TILE_SIZE + 1.0, (MAP_HEIGHT-1) * TILE_SIZE - 1));
            i32 mx = frogPos[i].x / TILE_SIZE;
            i32 my = frogPos[i].y / TILE_SIZE;
            onWater = mapData[my * MAP_WIDTH + mx] == MAP_TILE_WATER;
            assert(mapData[my * MAP_WIDTH + mx] != MAP_TILE_DEATH);
        }
    }

    for(i32 i = 0; i < FROG_COUNT; ++i) {
        frogInput[i].angle = randf64(0, TAU);
        frogInput[i].action = INPUT_ACTION_STILL;
    }

    for(i32 i = 0; i < FROG_COUNT; ++i) {
        frogEnergy[i] = ENERGY_TOTAL;
        frogHydration[i] = HYDRATION_TOTAL;
    }
}

void resetSimulation()
{
    resetFrogs();

    generationNumber = 0;
    curGenStats = {};
    lastGenStats = {};
    memset(pastGenStats, 0, sizeof(pastGenStats));

    neatSpec = {};
    neatGenomeInit(frogCurGen, FROG_COUNT, 6, 2, evolParam, &neatSpec);
    neatNnDealloc(frogNN);
    neatGenomeAllocMakeNN(frogCurGen, FROG_COUNT, frogNN);
    neatGenomeComputeNodePos(frogCurGen, FROG_COUNT);
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

void ui_frogViewer()
{
    ImGui::Begin("Champion viewer");

    static bool autoSelectBest = true;
    ImGui::Checkbox("Auto-select best", &autoSelectBest);

    if(autoSelectBest) {
        f64 bestFitness = 0;
        for(i32 i = 0; i < FROG_COUNT; ++i) {
            if(!frogDead[i] && frogFitness[i] > bestFitness) {
                dbgViewerFrogId = i;
                bestFitness = frogFitness[i];
            }
        }
    }
    else {
        ImGui::PushItemWidth(100);
        ImGui::SliderInt("##frog_id", &dbgViewerFrogId, 0, FROG_COUNT-1); ImGui::SameLine();
        if(ImGui::Button("Next alive")) {
            for(i32 i = 1; i < FROG_COUNT; ++i) {
                i32 id = (dbgViewerFrogId + i) % FROG_COUNT;
                if(!frogDead[id]) {
                    dbgViewerFrogId = id;
                    break;
                }
            }
        }
    }

    ImVec4 frogCol(speciesColor[dbgViewerFrogId].r/255.f, speciesColor[dbgViewerFrogId].g/255.f,
                   speciesColor[dbgViewerFrogId].b/255.f, 1);

    ImGui::Text("Frog_%d", dbgViewerFrogId);
    ImGui::TextColored(ImVec4(0, 1, 0, 1), "fitness: %g", frogFitness[dbgViewerFrogId]);

    ImGui::Separator();

    const ImVec2 ss(20, 20); // sensor size
    ImGui::BeginGroup();
    ImGui_ColoredRect(ss, ImVec4(frogWaterSensors[dbgViewerFrogId].sens[0], 0, 0, 1));
    ImGui::SameLine();
    ImGui_ColoredRect(ss, ImVec4(0, 0, 0, 1));
    ImGui::SameLine();
    ImGui_ColoredRect(ss, ImVec4(frogWaterSensors[dbgViewerFrogId].sens[1], 0, 0, 1));

    ImGui_ColoredRect(ss, ImVec4(0, 0, 0, 1));
    ImGui::SameLine();
    ImGui_ColoredRect(ss, ImVec4(frogWaterSensors[dbgViewerFrogId].center, 0, 0, 1));
    ImGui::SameLine();
    ImGui_ColoredRect(ss, ImVec4(0, 0, 0, 1));

    ImGui_ColoredRect(ss, ImVec4(frogWaterSensors[dbgViewerFrogId].sens[2], 0, 0, 1));
    ImGui::SameLine();
    ImGui_ColoredRect(ss, ImVec4(0, 0, 0, 1));
    ImGui::SameLine();
    ImGui_ColoredRect(ss, ImVec4(frogWaterSensors[dbgViewerFrogId].sens[3], 0, 0, 1));
    ImGui::EndGroup();

    ImGui::BeginGroup();
    /*(ImVec2(64, 32),
                      ImVec4((frogClosestPondFactorOffsetSign[champId] + 1.0) * 0.5, 0, 0, 1));*/
    f32 red = (frogClosestPondAngleDiff[dbgViewerFrogId] + 1.0) * 0.5;
    ImGui::Text("frogClosestPondAngleDiff: %g", frogClosestPondAngleDiff[dbgViewerFrogId]);
    /*f32 f = frogClosestDeathBondFactor[dbgViewerFrogId];
    ImGui_ColoredRect(ImVec2(64, 32), ImVec4(f, 0, f, 1));*/
    ImGui::EndGroup();

    //ImGui::Text("pondAngle: %g", frogClosestPondAngleDiff[dbgViewerFrogId]);

    ImGui_NeatNN(frogCurGen[dbgViewerFrogId]);

    ImGui::Separator();

    ImGui::Image((ImTextureID)(intptr_t)tex_frog[!frogDead[dbgViewerFrogId] ? FROG_TEX_STILL:FROG_TEX_DEAD],
                 ImVec2(180, 180),
                 ImVec2(0, 0), ImVec2(1, 1), frogCol);


    ImGui::Separator();

    switch(frogInput[dbgViewerFrogId].action) {
        case INPUT_ACTION_STILL: ImGui::TextUnformatted("STILL"); break;
        case INPUT_ACTION_JUMP:  ImGui::TextUnformatted("JUMPING"); break;
        case INPUT_ACTION_EAT:   ImGui::TextUnformatted("EAT"); break;
    }

    ImGui::Separator();

    // energy bar
    ImGui::ProgressBar(frogEnergy[dbgViewerFrogId]/ENERGY_TOTAL);

    // hydration bar
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, 0xffff9d00);
    ImGui::ProgressBar(frogHydration[dbgViewerFrogId]/ENERGY_TOTAL);
    ImGui::PopStyleColor(1);

    ImGui::End();
}

void ui_debug()
{
    ImGui::Begin("Debug");

        ImGui::Checkbox("Show bars", &dbgShowBars);
        ImGui::Checkbox("Show directions", &dbgShowDirections);
        ImGui::Checkbox("Overlay water average", &dbgOverlayAvgWaterMap);

    ImGui::End();
}

void ui_speciation()
{
    ImVec4 subPopColors[1024];
    const i32 speciesCount = neatSpec.speciesCount;
    for(i32 i = 0; i < speciesCount; ++i) {
        Color3 sc = speciesColor[i];
        subPopColors[i] = ImVec4(sc.r/255.f, sc.g/255.f, sc.b/255.f, 1.f);
    }

    i32 maxPopCount = 0;
    for(i32 i = 0; i < speciesCount; ++i) {
        maxPopCount = max(neatSpec.speciesPopCount[i], maxPopCount);
    }

    ImGui::Begin("Speciation");

    for(i32 i = 0; i < speciesCount; ++i) {
        if(neatSpec.speciesPopCount[i] == 0) continue;

        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, subPopColors[i]);

        char buff[64];
        sprintf(buff, "%d", neatSpec.speciesPopCount[i]);
        ImGui::ProgressBar(neatSpec.speciesPopCount[i]/(f32)maxPopCount, ImVec2(100,0), buff);
        ImGui::SameLine();
        ImGui::TextColored(subPopColors[i], "%g | %d", neatSpec.maxFitness[i], neatSpec.stagnation[i]);

        ImGui::PopStyleColor(1);
    }

    ImGui::End();
}

void ui_lastGeneration()
{
    // last generation statistics window
    ImGui::Begin("Last generation");

    ImGui::Text("Generation %d", lastGenStats.number);
    ImGui::Separator();

    ImGui::TextColored(ImVec4(0, 1, 0, 1), "Fitness max: %g", lastGenStats.maxFitness);
    ImGui::Text("Fitness avg: %g", lastGenStats.avgFitness);

    ImGui::TextColored(ImVec4(0.8, 0.8, 0.8, 1), "Deaths by border: %d", lastGenStats.deathsByBorder);
    ImGui::TextColored(ImVec4(0, 0.8, 1, 1), "Deaths by dehydration: %d",
                       lastGenStats.deathsByDehydratation);

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

    ImGui::Begin("Options");

        ImGui::TextUnformatted("Map");
        ImGui::BeginGroup();
            static i32 newPondCount = pondCount;
            ImGui::SliderInt("Pond count", &newPondCount, 1, MAP_POND_MAX_COUNT);

            if(ImGui::Button("Reset map")) {
                pondCount = newPondCount;
                resetMap();
            }
        ImGui::EndGroup();

        ImGui::Separator();

        ImGui::TextUnformatted("Game");
        ImGui::BeginGroup();
            ImGui::SliderInt("Time scale", &timeScale, 1, 20);

            if(ImGui::Button("Reset game")) {
                resetFrogs();
            }
        ImGui::EndGroup();

    ImGui::End();

    ui_debug();
    ui_frogViewer();
    ui_speciation();
    ui_lastGeneration();

    //ImGui::ShowDemoWindow();
}

void updateNNs()
{
    NeatNN* nnets[FROG_COUNT];
    i32 nnetsCount = 0;
    //const f32 waterSmellSquareCount = VISION_WIDTH * VISION_WIDTH * 0.25;

    constexpr f32 sensorOffsetPos[4][2] {
        {  -FROG_WATER_SENSOR_OFFSET,  -FROG_WATER_SENSOR_OFFSET },
        {   FROG_WATER_SENSOR_OFFSET,  -FROG_WATER_SENSOR_OFFSET },
        {  -FROG_WATER_SENSOR_OFFSET,  FROG_WATER_SENSOR_OFFSET },
        {   FROG_WATER_SENSOR_OFFSET,  FROG_WATER_SENSOR_OFFSET }
    };

    for(i32 i = 0; i < FROG_COUNT; ++i) {
        if(frogDead[i]) continue;

        // center sensor
        i32 centerX = frogPos[i].x / TILE_SIZE;
        i32 centerY = frogPos[i].y / TILE_SIZE;
        frogWaterSensors[i].center = mapData[centerX + centerY * MAP_WIDTH] == MAP_TILE_WATER ? 1.0:0.0;

        for(i32 s = 0; s < 4; ++s) {
            i32 sensorX = (frogPos[i].x + sensorOffsetPos[s][0]) / (TILE_SIZE * MAP_WATER_AVG_GRID_SIZE);
            i32 sensorY = (frogPos[i].y + sensorOffsetPos[s][1]) / (TILE_SIZE * MAP_WATER_AVG_GRID_SIZE);
            if(sensorX < 0 || sensorX >= MAP_WATER_AVG_WIDTH || sensorY < 0 ||
               sensorY >= MAP_WATER_AVG_HEIGHT) {
                frogWaterSensors[i].sens[s] = 0.0;
            }
            else {
                assert(sensorX + sensorY * MAP_WATER_AVG_WIDTH < MAP_WATER_AVG_SIZE);
                frogWaterSensors[i].sens[s] = mapAvgWater[sensorX + sensorY * MAP_WATER_AVG_WIDTH];
            }
        }

#if 0
        const i32 frogX = frogPos[i].x / TILE_SIZE;
        const i32 frogY = frogPos[i].y / TILE_SIZE;
        frogWaterSensors[i] = {};
        for(i32 y = 0; y < VISION_WIDTH; ++y) {
            for(i32 x = 0; x < VISION_WIDTH; ++x) {
                i32 mx = frogX + x - VISION_WIDTH/2;
                i32 my = frogY + y - VISION_WIDTH/2;
                i32 wi = (x / (VISION_WIDTH/2)) + (y / (VISION_WIDTH/2)) * 2;
                assert(wi < 4);

                if(my < 0 || my >= MAP_HEIGHT || mx < 0 || mx >= MAP_WIDTH) {
                }
                else {
                    frogWaterSensors[i].sens[wi] += mapData[my * MAP_WIDTH + mx];
                }
            }
        }
        frogWaterSensors[i].sens[0] /= waterSmellSquareCount;
        frogWaterSensors[i].sens[1] /= waterSmellSquareCount;
        frogWaterSensors[i].sens[2] /= waterSmellSquareCount;
        frogWaterSensors[i].sens[3] /= waterSmellSquareCount;
#endif
    }

    for(i32 i = 0; i < FROG_COUNT; ++i) {
        //if(frogDead[i]) continue;
        const i32 frogTileX = frogPos[i].x / TILE_SIZE;
        const i32 frogTileY = frogPos[i].y / TILE_SIZE;
        const Vec2 frogTilePos = vec2Make(frogTileX, frogTileY);

        f32 minDist = FLT_MAX;
        Vec2 chosenPondPos;
        for(i32 p = 0; p < pondCount; ++p) {
            const Vec2 pondTilePos = vec2Make(pondPos[p] % MAP_WIDTH, pondPos[p] / MAP_WIDTH);
            f32 dist = vec2Distance(&frogTilePos, &pondTilePos) - pondRadius[p];

            if(dist < minDist) {
                minDist = dist;
                chosenPondPos = {pondTilePos.x * TILE_SIZE, pondTilePos.y * TILE_SIZE};
            }
        }

        Vec2 pondV = vec2Minus(&chosenPondPos, &frogPos[i]);
        //pondV = vec2Normalize(&pondV);
        Vec2 vx = {1.0f, 0.0f};
        Vec2 dir = {cosf(frogAngle[i]), sinf(frogAngle[i])};
        //frogClosestPondAngleDiff[i] = vec2Dot(&pondV, &dir);
        f32 diff = vec2AngleBetween(&dir, &pondV);

        frogClosestPondAngleDiff[i] = diff / PI;
        assert(frogClosestPondAngleDiff[i] >= -1.0 && frogClosestPondAngleDiff[i] <= 1.0);
        /*f32 angle = vec2Angle(&pondV);
        f32 diff = angle - frogAngle[i];
        if(diff < 0.0f) {
            diff += TAU;
        }
        else if(diff > TAU) {
            diff -= TAU;
        }
        frogClosestPondAngleDiff[i] = diff / TAU;
        assert(frogClosestPondAngleDiff[i] >= 0.0f && frogClosestPondAngleDiff[i] <= 1.0f);*/
    }

    f64 oldClosestPondFactor[FROG_COUNT];
    memmove(oldClosestPondFactor, frogClosestPondFactor, sizeof(frogClosestPondFactor));

    for(i32 i = 0; i < FROG_COUNT; ++i) {
        if(frogDead[i]) continue;
        const i32 frogTileX = frogPos[i].x / TILE_SIZE;
        const i32 frogTileY = frogPos[i].y / TILE_SIZE;
        const Vec2 frogTilePos = vec2Make(frogTileX, frogTileY);

        f32 minDist = FLT_MAX;
        f32 chosenPondRadius = 0;
        for(i32 p = 0; p < pondCount; ++p) {
            const Vec2 pondTilePos = vec2Make(pondPos[p] % MAP_WIDTH, pondPos[p] / MAP_WIDTH);
            f32 dist = vec2Distance(&frogTilePos, &pondTilePos) - pondRadius[p];

            if(dist < minDist) {
                minDist = dist;
                chosenPondRadius = pondRadius[p];
            }
        }

        if(minDist < 1.0) {
            frogClosestPondFactor[i] = 1.0;
        }
        else {
            frogClosestPondFactor[i] = 1.0 / minDist;
        }
    }

    for(i32 i = 0; i < FROG_COUNT; ++i) {
        f64 offset = frogClosestPondFactor[i] - oldClosestPondFactor[i];
        if(offset > 0) {
            frogClosestPondFactorOffsetSign[i] = 1;
        }
        else if(offset < 0) {
            frogClosestPondFactorOffsetSign[i] = -1;
        }
        else {
            frogClosestPondFactorOffsetSign[i] = 0;
        }
    }

    u8 frogIsInWater[FROG_COUNT];
    for(i32 i = 0; i < FROG_COUNT; ++i) {
       const i32 frogTileX = frogPos[i].x / TILE_SIZE;
       const i32 frogTileY = frogPos[i].y / TILE_SIZE;
       frogIsInWater[i] = mapData[frogTileX + frogTileY * MAP_WIDTH] == MAP_TILE_WATER;
    }

    // how close are we to map bounds
    for(i32 i = 0; i < FROG_COUNT; ++i) {
        if(frogDead[i]) continue;
        const f32 halfMapWidth  = MAP_WIDTH  * 0.5f * TILE_SIZE;
        const f32 halfMapHeight = MAP_HEIGHT * 0.5f * TILE_SIZE;
        f32 db = max(fabs(frogPos[i].x - halfMapWidth) / halfMapWidth,
                     fabs(frogPos[i].y - halfMapHeight) / halfMapHeight);
        frogClosestDeathBondFactor[i] = db;
    }

    for(i32 i = 0; i < FROG_COUNT; ++i) {
        if(frogDead[i]) continue;

        f64 input[6] = {
            frogWaterSensors[i].sens[0],
            frogWaterSensors[i].sens[1],
            frogWaterSensors[i].sens[2],
            frogWaterSensors[i].sens[3],
            frogClosestPondAngleDiff[i],
            frogHydration[i] / HYDRATION_TOTAL
        };

        frogNN[i]->setInputs(input, arr_count(input));

        //input[1] = frogIsInWater[i];
        //input[1] = frogClosestDeathBondFactor[i];
        //input[2] = frogAngle[i] / TAU;
        //input[2] = frogEnergy[i];
        //input[3] = frogHydration[i];

        nnets[nnetsCount++] = frogNN[i];
    }

    neatNnPropagate(nnets, nnetsCount);

    const i32 firstOutputId = frogCurGen[0]->inputNodeCount;
    const i32 outputCount = frogCurGen[0]->outputNodeCount;

    for(i32 i = 0; i < FROG_COUNT; ++i) {
        if(frogDead[i]) continue;

        f64 out[2];
        memmove(out, &frogNN[i]->nodeValues[firstOutputId], sizeof(out[0]) * outputCount);

        out[0] = (out[0] + 1.0) * 0.5;
        out[1] = (out[1] + 1.0) * 0.5;
        assert(out[0] >= 0.0 && out[0] <= 1.0);
        assert(out[1] >= 0.0 && out[1] <= 1.0);

        frogInput[i].angle = fmod(out[0] * 10.0 * TAU, TAU);

        frogInput[i].action = out[1] * 3;
        if(frogInput[i].action > INPUT_ACTION_EAT) {
            frogInput[i].action = INPUT_ACTION_EAT;
        }
    }
}

void updatePhysics()
{
    for(i32 i = 0; i < FROG_COUNT; ++i) {
        if(frogDead[i]) continue;
        frogAngle[i] = frogInput[i].angle;

        if(frogInput[i].action == INPUT_ACTION_JUMP) {
            i64 jumpDeltaMs = frogJumpTime[i] * 1000;
            bool doJump = jumpDeltaMs < FROG_ANIM_JUMP;

            if(doJump) {
                f32 angle = frogAngle[i];
                Vec2 moveVec = vec2Make(cosf(angle) * FROG_SPEED * FRAME_DT,
                                        sinf(angle) * FROG_SPEED * FRAME_DT);
                frogPos[i] = vec2Add(&frogPos[i], &moveVec);

                frogPos[i].x = clampf64(frogPos[i].x, 0, MAP_WIDTH * TILE_SIZE - 1.0);
                frogPos[i].y = clampf64(frogPos[i].y, 0, MAP_HEIGHT * TILE_SIZE - 1.0);
            }
        }
    }
}

void updateMechanics()
{
    i32 frogRewards[FROG_COUNT] = {0};

    for(i32 i = 0; i < FROG_COUNT; ++i) {
        if(frogDead[i]) continue;

        frogHydration[i] -= HYDRATION_DRAIN_PER_SEC * FRAME_DT;
        if(frogHydration[i] <= 0) {
            frogHydration[i] = 0;
            frogDead[i] = true;
            curGenStats.deathsByDehydratation++;
        }
    }

    for(i32 i = 0; i < FROG_COUNT; ++i) {
        if(frogDead[i]) continue;

        if(frogEnergy[i] <= 0) {
            frogEnergy[i] = 0;
            //frogDead[i] = true;
        }
    }

    for(i32 i = 0; i < FROG_COUNT; ++i) {
        if(frogDead[i]) continue;

        if(frogInput[i].action != INPUT_ACTION_JUMP) {
            frogJumpTime[i] = 0;
        }
        if(frogInput[i].action != INPUT_ACTION_EAT) {
            frogTongueTime[i] = 0;
        }

        if(frogInput[i].action == INPUT_ACTION_JUMP) {
            if(frogJumpTime[i] == 0) {
                frogEnergy[i] -= ENERGY_JUMP_DRAIN;
            }
            frogJumpTime[i] += FRAME_DT;
            if(frogJumpTime[i] * 1000 > FROG_JUMP_CD) {
                frogJumpTime[i] = 0;
            }
        }
        else if(frogInput[i].action == INPUT_ACTION_EAT) {
            if(frogTongueTime[i] == 0) {
                frogEnergy[i] -= ENERGY_EAT_DRAIN;
            }
            frogTongueTime[i] += FRAME_DT;
            if(frogTongueTime[i] * 1000 > FROG_TONGUE_CD) {
                frogTongueTime[i] = 0;
            }
        }
    }

    for(i32 i = 0; i < FROG_COUNT; ++i) {
        if(frogDead[i]) continue;

        i32 mx = frogPos[i].x / TILE_SIZE;
        i32 my = frogPos[i].y / TILE_SIZE;
        assert(mx >= 0 && mx < MAP_WIDTH);
        assert(my >= 0 && my < MAP_HEIGHT);

        if(mapData[my * MAP_WIDTH + mx] == MAP_TILE_WATER) {
            frogHydration[i] += HYDRATION_GAIN_PER_SEC * FRAME_DT;
            frogHydration[i] = min(frogHydration[i], HYDRATION_TOTAL);
            frogRewards[i] += 10;
        }
        /*else if(mapData[my * MAP_WIDTH + mx] == MAP_TILE_DEATH) {
            frogDead[i] = true;
            curGenStats.deathsByBorder++;
        }*/
    }

    for(i32 i = 0; i < FROG_COUNT; ++i) {
        if(frogClosestPondFactorOffsetSign[i] > 0) {
            frogRewards[i] += 1;
        }
    }

    simulationTime += FRAME_DT;
    if(simulationTime >= SIMULATION_MAX_TIME) {
        memset(frogDead, 1, sizeof(frogDead));
    }

    for(i32 i = 0; i < FROG_COUNT; ++i) {
        if(frogDead[i]) continue;

        //frogFitness[i] += FRAME_DT;
        frogFitness[i] += frogRewards[i];
        //frogFitness[i] += frogHydration[i] / HYDRATION_TOTAL;
        //frogFitness[i] += (frogEnergy[i] / ENERGY_TOTAL) * 0.5;
        //frogFitness[i] += frogFliesEatenCount;
    }

    bool everyoneIsDead = true;
    for(i32 i = 0; i < FROG_COUNT && everyoneIsDead; ++i) {
        if(!frogDead[i]) {
            everyoneIsDead = false;
        }
    }

    f64 maxFitness = 0.0;
    f64 totalFitness = 0.0;
    for(i32 i = 0; i < FROG_COUNT; ++i) {
        totalFitness += frogFitness[i];
        if(frogFitness[i] > maxFitness) {
            maxFitness = frogFitness[i];
        }
    }

    curGenStats.avgFitness = totalFitness / FROG_COUNT;
    curGenStats.maxFitness = maxFitness;

    if(everyoneIsDead) {
        nexGeneration();
    }
}

void nexGeneration()
{
    lastGenStats = curGenStats;
    memmove(pastGenStats, pastGenStats+1, sizeof(pastGenStats) - sizeof(pastGenStats[0]));
    pastGenStats[STATS_HISTORY_COUNT-1] = lastGenStats;

    curGenStats = {};
    curGenStats.number = generationNumber++;

    LOG("#%d maxFitness=%.5f avg=%.5f", lastGenStats.number, lastGenStats.maxFitness,
        lastGenStats.avgFitness);

    neatEvolve(frogCurGen, frogNextGen, frogFitness, FROG_COUNT, &neatSpec, evolParam, true);

    neatNnDealloc(frogNN);
    neatGenomeAllocMakeNN(frogCurGen, FROG_COUNT, frogNN);

    neatGenomeComputeNodePos(frogCurGen, FROG_COUNT);

    resetMap();
    resetFrogs();
}

void newFrame()
{
    window.uiNewFrame();
    doUI();

    updateNNs();
    updatePhysics();
    updateMechanics();

    // assign frog sprite
    for(i32 i = 0; i < FROG_COUNT; ++i) {
        if(frogDead[i]) {
            frogTexId[i] = FROG_TEX_DEAD;
            continue;
        }

        if(frogInput[i].action == INPUT_ACTION_JUMP) {
            i64 jumpDeltaMs = frogJumpTime[i] * 1000;
            bool doJump = jumpDeltaMs < FROG_ANIM_JUMP;
            frogTexId[i] = doJump ? FROG_TEX_JUMP : FROG_TEX_STILL;
        }
        else if(frogInput[i].action == INPUT_ACTION_EAT) {
            i64 tongueDeltaMs = frogTongueTime[i] * 1000;
            bool doTongue = tongueDeltaMs < FROG_ANIM_TONGUE;
            frogTexId[i] = doTongue ? FROG_TEX_TONGUE : FROG_TEX_STILL;
        }
        else {
            frogTexId[i] = FROG_TEX_STILL;
        }
    }

    // frog transforms
    for(i32 i = 0; i < FROG_COUNT; ++i) {
        frogTf[i].pos = frogPos[i];
        frogTf[i].rot = frogAngle[i] + PI * 0.5;

        switch(frogTexId[i]) {
            case FROG_TEX_STILL:
            case FROG_TEX_DEAD: {
                frogTf[i].size = vec2Make(FROG_SIZE, FROG_SIZE);
                frogTf[i].center = vec2Make(FROG_SIZE*0.5f, FROG_SIZE*0.5f);
            } break;

            case FROG_TEX_JUMP: {
                frogTf[i].size = vec2Make(FROG_SIZE, FROG_SIZE * 1.5625);
                frogTf[i].center = vec2Make(frogTf[i].size.x*0.5f, frogTf[i].size.y*0.4f);
            } break;

            case FROG_TEX_TONGUE: {
                frogTf[i].size = vec2Make(FROG_SIZE, FROG_SIZE * 1.5625);
                frogTf[i].center = vec2Make(frogTf[i].size.x*0.5f, frogTf[i].size.y*0.65f);
            } break;
        }
    }
}

void render()
{
    glClear(GL_COLOR_BUFFER_BIT);

#if 0
    // draw map
    const f32 gl = 0;
    const f32 gr = TILE_SIZE * MAP_WIDTH;
    const f32 gt = 0;
    const f32 gb = TILE_SIZE * MAP_HEIGHT;
    const Color4 grassColor = {40, 89, 24, 255};
    const Color4 waterColor = {66, 138, 255, 255};

    Quad grassQuad = quadOneColor(gl, gr, gt, gb, grassColor);
    drawQuadBatch(&grassQuad, 1);

    constexpr i32 MAX_WATER_QUADS = 2048;
    Quad waterQuad[MAX_WATER_QUADS];
    i32 waterCount = 0;

    for(i32 i = 0; i < mapSize; ++i) {
        i32 x = (i % MAP_WIDTH) * TILE_SIZE;
        i32 y = (i / MAP_WIDTH) * TILE_SIZE;

        if(mapData[i] == MAP_TILE_WATER) {
            waterQuad[waterCount++] = quadOneColor(x, x+TILE_SIZE, y, y+TILE_SIZE, waterColor);
            if(waterCount == MAX_WATER_QUADS) {
                drawQuadBatch(waterQuad, MAX_WATER_QUADS);
                waterCount = 0;
            }
        }
    }

    drawQuadBatch(waterQuad, waterCount);
#endif

    Transform mapTf;
    mapTf.pos = {0, 0};
    mapTf.size = {TILE_SIZE * MAP_WIDTH, TILE_SIZE * MAP_HEIGHT};
    drawSpriteBatch(tex_map, &mapTf, 1);

    if(dbgOverlayAvgWaterMap) {
        mapTf.pos = {0, 0};
        mapTf.size = {TILE_SIZE * MAP_WIDTH, TILE_SIZE * MAP_HEIGHT};
        drawSpriteBatch(tex_mapAvgWater, &mapTf, 1);
    }

    // draw frogs
    constexpr i32 BATCH_MAX = 128;
    Transform tf[128];
    Color3 color[128];
    i32 batchCount = 0;

    for(i32 t = 0; t < FROG_TEX_COUNT; ++t) {
        batchCount = 0;

        for(i32 i = 0; i < FROG_COUNT; ++i) {
            if(frogTexId[i] == t) {
                i32 id = batchCount++;
                tf[id] = frogTf[i];
                color[id] = speciesColor[frogCurGen[i]->species];
                if(batchCount == BATCH_MAX) {
                    drawSpriteColorBatch(tex_frog[t], tf, color, BATCH_MAX);
                    batchCount = 0;
                }
            }
        }

        drawSpriteColorBatch(tex_frog[t], tf, color, batchCount);
    }

    if(dbgShowBars) {
        Quad barQuad[FROG_COUNT];
        i32 barCount = 0;
        const Color4 hydrationColor = {0, 255, 255, 200};
        const Color4 energyColor = {255, 255, 0, 200};
        constexpr f32 barWidth = 40.f;
        constexpr f32 barHeight = 4.f;

        for(i32 i = 0; i < FROG_COUNT; ++i) {
            f32 x = frogPos[i].x;
            f32 y = frogPos[i].y;
            f32 h = frogHydration[i] / HYDRATION_TOTAL;
            barQuad[barCount++] = quadOneColor(x - barWidth * 0.5, x - barWidth * 0.5 + (barWidth * h),
                                               y - FROG_SIZE * 0.5, y - FROG_SIZE * 0.5 + barHeight,
                                               hydrationColor);
        }
        drawQuadBatch(barQuad, barCount);

        barCount = 0;
        for(i32 i = 0; i < FROG_COUNT; ++i) {
            f32 x = frogPos[i].x;
            f32 y = frogPos[i].y;
            f32 h = frogEnergy[i] / ENERGY_TOTAL;
            barQuad[barCount++] = quadOneColor(x - barWidth * 0.5, x - barWidth * 0.5 + (barWidth * h),
                                               y - FROG_SIZE * 0.5 - barHeight, y - FROG_SIZE * 0.5,
                                               energyColor);
        }
        drawQuadBatch(barQuad, barCount);
    }

    if(dbgShowDirections) {
        Line lines[FROG_COUNT];
        const Color4 pink = {255, 0, 255, 255};
        const Color4 white = {255, 255, 255, 255};

        // dir angle
        for(i32 i = 0; i < FROG_COUNT; ++i) {
            f32 a = frogAngle[i];
            lines[i].c1 = white;
            lines[i].c2 = white;
            lines[i].p1 = frogPos[i];
            Vec2 dir = {cosf(a) * 100, sinf(a) * 100};
            lines[i].p2 = vec2Add(&frogPos[i], &dir);
        }
        drawLineBatch(lines, FROG_COUNT, 1.0f);

        // pond angle
        for(i32 i = 0; i < FROG_COUNT; ++i) {
            f32 a = frogAngle[i] + frogClosestPondAngleDiff[i] * PI;
            lines[i].c1 = pink;
            lines[i].c2 = pink;
            lines[i].p1 = frogPos[i];
            Vec2 dir = {cosf(a) * 100, sinf(a) * 100};
            lines[i].p2 = vec2Add(&frogPos[i], &dir);
        }
        drawLineBatch(lines, FROG_COUNT, 2.0f);
    }

    f32 champX = frogPos[dbgViewerFrogId].x;
    f32 champY = frogPos[dbgViewerFrogId].y;
    constexpr Color4 champLineColor = {255, 0, 0, 255};
    constexpr Color4 champColor = {255, 0, 0, 128};

    Line champLine;
    champLine.p1 = vec2Make(MAP_WIDTH * TILE_SIZE * 0.5, 0);
    champLine.p2 = frogPos[dbgViewerFrogId];
    champLine.c1 = champLineColor;
    champLine.c2 = champLineColor;
    drawLineBatch(&champLine, 1);

    Quad championOverlay = quadOneColor(champX - FROG_SIZE, champX + FROG_SIZE,
                                        champY - FROG_SIZE, champY + FROG_SIZE, champColor);
    drawQuadBatch(&championOverlay, 1);

    /*Quad quad = quadOneColor(dbgLastMouseLeftClick.x, dbgLastMouseLeftClick.x + 200.f,
                             dbgLastMouseLeftClick.y, dbgLastMouseLeftClick.y + 200.f, {255, 128, 0, 255});
    drawQuadBatch(&quad, 1);*/

    window.uiRender();
}

};

#ifdef _WIN32
int CALLBACK WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
#else
i32 main()
#endif
{
    LOG("=== F R O G S ===\n");
    LOG("   (o)____(o)");
    LOG(" _/          \\_ ");
    LOG("/ \\----------/ \\");
    LOG("\\   | |  | |   /");
    LOG(" ww ooo  ooo ww");
    LOG("\n");

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
