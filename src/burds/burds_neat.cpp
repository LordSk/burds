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

#define BIRD_COUNT 1024
constexpr i32 SUBPOP_MAX_COUNT = BIRD_COUNT;

#define BIRD_BODY_RATIO 0.5f
#define BIRD_WING_RATIO 1.68125f

#define BIRD_BODY_HEIGHT 50
#define BIRD_BODY_WIDTH (BIRD_BODY_HEIGHT * BIRD_BODY_RATIO)
#define BIRD_WING_WIDTH 50
#define BIRD_WING_HEIGHT (1.f / BIRD_WING_RATIO * BIRD_WING_WIDTH)

#define WING_FLAP_TIME 0.4f

#define FRICTION_AIR_ANGULAR 0.05f
#define FRICTION_AIR 0.05f

#define WING_STRENGTH 20000.f
#define WING_BRAKE 0.2
#define WING_STRENGTH_ANGULAR (PI * 10.0)

#define ANGULAR_VELOCITY_MAX (TAU * 2.0)

#define APPLE_POS_LIST_COUNT 1024
#define APPLE_RADIUS 80.0
#define HEALTH_MAX 10.0

#define GROUND_Y 1000

#define REINSERT_TRUNCATE 0x1
#define REINSERT_DUPLICATE 0x2
#define REINSERT_STRATEGY REINSERT_TRUNCATE

#define NEURAL_NET_LAYERS { 4, 12, 4 }

#define STATS_HISTORY_COUNT 30

struct BirdInput
{
    u8 left, right, flap, flapHard, brake, brakeHard;
};

enum {
    MODE_NN_TRAIN=0,
    MODE_HUMAN_PLAY
};

struct Bounds {
    f64 bmin, bmax;
};

struct App {

AppWindow window;

i32 tex_birdBody;
i32 tex_birdWing;
i32 tex_apple;

Transform birdBodyTf[BIRD_COUNT];
Transform birdLeftWingTf[BIRD_COUNT];
Transform birdRightWingTf[BIRD_COUNT];
Color3 speciesColor[SUBPOP_MAX_COUNT];

Vec2 birdPos[BIRD_COUNT];
Vec2 birdVel[BIRD_COUNT];
f32 birdRot[BIRD_COUNT];
f32 birdAngularVel[BIRD_COUNT];
f32 birdFlapLeftCd[BIRD_COUNT];
f32 birdFlapRightCd[BIRD_COUNT];
BirdInput birdInput[BIRD_COUNT];

f32 birdHealth[BIRD_COUNT];
u8 birdDead[BIRD_COUNT];
u8 birdDeadFromGround[BIRD_COUNT];
i32 birdApplePositionId[BIRD_COUNT];

Vec2 applePosList[APPLE_POS_LIST_COUNT];
Transform appleTf[BIRD_COUNT];

Line targetLine[BIRD_COUNT];

Genome* birdCurGen[BIRD_COUNT];
Genome* birdNextGen[BIRD_COUNT];
NeatNN* birdNN[BIRD_COUNT] = {0};
NeatEvolutionParams evolParam;
NeatSpeciation neatSpec;

i32 birdAppleEatenCount[BIRD_COUNT];
f32 birdDistToNextApple[BIRD_COUNT];
f32 birdShortestDistToNextApple[BIRD_COUNT];

f64 birdFitnessAcc[BIRD_COUNT];
f64 birdFitness[BIRD_COUNT];

f32 viewZoom;
i32 viewX;
i32 viewY;
u8 mouseRightButDown = 0;

i32 timeScale;

i32 dbgViewerBirdId = 0;
bool dbgShowObjLines = true;
bool dbgHightlightBird = true;
bool dbgFollowBird = false;

struct GenerationStats {
    i32 number = 0;
    f64 maxFitness = 0.0;
    f64 avgFitness = 0.0;
};

i32 generationNumber = 0;
GenerationStats curGenStats;
GenerationStats lastGenStats;
GenerationStats pastGenStats[STATS_HISTORY_COUNT];

void resetBirdColors()
{
    const u32 colorMax = 0xFF;
    const u32 colorMin = 0x0;

    for(i32 i = 0; i < SUBPOP_MAX_COUNT; ++i) {
        Color3 c = {0, 0, 0};
        while((c.r + c.g + c.b) / 3.0 < 100) {
            c.r = randi64(colorMin, colorMax);
            c.g = randi64(colorMin, colorMax);
            c.b = randi64(colorMin, colorMax);
        }
        speciesColor[i] = c;
    }
}

void resetBirds()
{
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        birdPos[i].x = 0;
        birdPos[i].y = GROUND_Y-100;
    }

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        birdRot[i] = -PI * 0.5;
    }

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        birdBodyTf[i].size.x = BIRD_BODY_WIDTH;
        birdBodyTf[i].size.y = BIRD_BODY_HEIGHT;
        birdBodyTf[i].center.x = BIRD_BODY_WIDTH * 0.5f;
        birdBodyTf[i].center.y = BIRD_BODY_HEIGHT * 0.5f;
    }

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        birdLeftWingTf[i].size.x = -BIRD_WING_WIDTH;
        birdLeftWingTf[i].size.y = BIRD_WING_HEIGHT;
        birdLeftWingTf[i].center.x = 0;
        birdLeftWingTf[i].center.y = BIRD_WING_HEIGHT * 0.6f;
    }
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        birdRightWingTf[i].size.x = BIRD_WING_WIDTH;
        birdRightWingTf[i].size.y = BIRD_WING_HEIGHT;
        birdRightWingTf[i].center.x = 0;
        birdRightWingTf[i].center.y = BIRD_WING_HEIGHT * 0.6f;
    }

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        birdHealth[i] = HEALTH_MAX; // seconds
    }

    mem_zero(birdVel);
    mem_zero(birdAngularVel);
    mem_zero(birdFlapLeftCd);
    mem_zero(birdFlapRightCd);
    mem_zero(birdApplePositionId);
    mem_zero(birdAppleEatenCount);
    mem_zero(birdDead);
    mem_zero(birdDeadFromGround);
    mem_zero(birdFitness);
    mem_zero(birdFitnessAcc);

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        birdShortestDistToNextApple[i] = 99999999.9;
    }

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        appleTf[i].size.x = 80;
        appleTf[i].size.y = 80;
        appleTf[i].center.x = 40;
        appleTf[i].center.y = 40;
    }

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        memmove(&targetLine[i].c1, &speciesColor[i], 3);
        targetLine[i].c1.a = 128;
        memmove(&targetLine[i].c2, &speciesColor[i], 3);
        targetLine[i].c2.a = 0;
    }
}

void resetApplePath()
{
    i32 spawnOriginX = 0;
    i32 spawnOriginY = 0;
    const i32 spawnRadius = 1500;

    for(i32 i = 0; i < APPLE_POS_LIST_COUNT; ++i) {
        applePosList[i].x = spawnOriginX - spawnRadius + randi64(0, spawnRadius* 2);
        applePosList[i].y = spawnOriginY - spawnRadius + randi64(0, spawnRadius* 2);
        if(applePosList[i].y >= GROUND_Y - (spawnRadius / 2)) {
            applePosList[i].y = spawnOriginY - spawnRadius;
        }
        spawnOriginX = applePosList[i].x;
        spawnOriginY = applePosList[i].y;
    }
}

void resetView()
{
    viewX = -2000;
    viewY = -1000;
    viewZoom = 3.f;
}

void resetSimulation()
{
    resetBirdColors();
    resetBirds();
    resetApplePath();

    generationNumber = 0;
    curGenStats = {};
    lastGenStats = {};
    mem_zero(pastGenStats);

    neatSpec = {};
    neatGenomeInit(birdCurGen, BIRD_COUNT, 6, 4, evolParam, &neatSpec); // INPUTS: 6, OUPUTS: 4
    neatNnDealloc(birdNN);
    neatGenomeAllocMakeNN(birdCurGen, BIRD_COUNT, birdNN);
    neatGenomeComputeNodePos(birdCurGen, BIRD_COUNT);
}

bool init()
{
    if(!window.init("Burds [NEAT]", "burds_neat_imgui.ini")) {
        return false;
    }

    // sky color
    glClearColor(179.f/255.f, 204.f/255.f, 255.f/255.f, 1.0f);

    viewX = -2000;
    viewY = -1000;
    viewZoom = 3.f;

    tex_birdBody = loadTexture("../bird_body.png");
    tex_birdWing = loadTexture("../wing.png");
    tex_apple    = loadTexture("../apple.png");
    if(tex_birdBody == -1 ||
       tex_birdWing == -1 ||
       tex_apple    == -1) {
        return false;
    }

    timeScale = 1.0f;

    evolParam.compC1 = 2.0;
    evolParam.compC2 = 2.0;
    evolParam.compC3 = 3.0;
    evolParam.compT = 2.0;
    //evolParam.mutateAddConn = 0.05;
    evolParam.mutateAddNode = 0.1;
    evolParam.mutateWeightStep = 0.5;

    neatGenomeAlloc(birdCurGen, BIRD_COUNT);
    neatGenomeAlloc(birdNextGen, BIRD_COUNT);
    resetSimulation();

    return true;
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

        newFrame();
        render();
        window.swap();

        const i64 frameDtMicro = FRAME_DT/timeScale * 1000000;
        while(((frameDtMicro - timeToMicrosec(timeGet() - t0)) / 1000) > 1) {
            _mm_pause();
        }
    }
}

void handleEvent(const SDL_Event* event)
{
    if(event->type == SDL_QUIT) {
        window.running = false;
    }

    if(event->type == SDL_KEYDOWN) {
        if(event->key.keysym.sym == SDLK_ESCAPE) {
            window.running = false;
        }

        if(event->key.keysym.sym == SDLK_r) {
           resetBirds();
        }

        if(event->key.keysym.sym == SDLK_n) {
           resetSimulation();
        }
    }

    if(event->type == SDL_MOUSEBUTTONDOWN) {
        if(event->button.button == SDL_BUTTON_RIGHT) {
            mouseRightButDown = true;
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

void ImGui_BirdImage(i32 birdId, f32 height)
{
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    if (window->SkipItems)
        return;

    ImDrawList* drawList = ImGui::GetWindowDrawList();

    const ImVec2 size(height * 2, height);
    const ImVec2 pos = window->DC.CursorPos;
    const ImRect bb(pos, pos + size);
    ImGui::ItemSize(bb);

    ImVec2 bodySize(size.y * BIRD_BODY_RATIO, size.y);
    ImVec2 bodyPos(pos.x + size.x/2 - bodySize.x/2, pos.y);

    ImVec2 wingSize(size.x / 2, (1.f / BIRD_WING_RATIO * size.x / 2));
    ImVec2 wingPos(pos.x + size.x / 2, pos.y + size.y / 2 - wingSize.y * 0.6f);

    i32 species = birdCurGen[birdId]->species;
    if(species < 0) species = 0;
    Color3 bc = speciesColor[species];
    u32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(bc.r / 255.f, bc.g / 255.f, bc.b / 255.f, 1.0f));

    drawList->AddImage((ImTextureID)(intptr_t)tex_birdWing, wingPos - ImVec2(wingSize.x,0),
                       wingPos + ImVec2(0, wingSize.y),
                       ImVec2(1, 0), ImVec2(0, 1), col);
    drawList->AddImage((ImTextureID)(intptr_t)tex_birdWing, wingPos, wingPos + wingSize,
                       ImVec2(0,0), ImVec2(1,1), col);
    drawList->AddImage((ImTextureID)(intptr_t)tex_birdBody, bodyPos, bodyPos + bodySize,
                       ImVec2(0,0), ImVec2(1,1), col);
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

void ui_birdViewer()
{
    ImGui::Begin("Bird viewer");

    static bool autoSelectBest = true;
    ImGui::Checkbox("Auto-select best", &autoSelectBest);

    if(autoSelectBest) {
        f64 bestFitness = 0;
        for(i32 i = 0; i < BIRD_COUNT; ++i) {
            if(!birdDead[i] && birdFitness[i] > bestFitness) {
                dbgViewerBirdId = i;
                bestFitness = birdFitness[i];
            }
        }
    }
    else {
        ImGui::PushItemWidth(100);
        ImGui::SliderInt("##bird_id", &dbgViewerBirdId, 0, BIRD_COUNT-1); ImGui::SameLine();
        if(ImGui::Button("Next alive")) {
            for(i32 i = 1; i < BIRD_COUNT; ++i) {
                i32 id = (dbgViewerBirdId + i) % BIRD_COUNT;
                if(!birdDead[id]) {
                    dbgViewerBirdId = id;
                    break;
                }
            }
        }
    }

    ImGui::Text("Bird_%04d [%03X]", dbgViewerBirdId, birdCurGen[dbgViewerBirdId]->species);
    ImGui::TextColored(ImVec4(0, 1, 0, 1), "fitness: %g", birdFitness[dbgViewerBirdId]);
    ImGui::TextColored(ImVec4(1, 0.5, 0, 1), "apples: %d", birdAppleEatenCount[dbgViewerBirdId]);

    ImGui::Separator();

    ImGui_NeatNN(birdCurGen[dbgViewerBirdId]);

    ImGui::Separator();

    const ImVec4 titleColor(0.4, 0.62, 1.0, 1.0);

    ImGui::BeginGroup();
    ImGui::TextColored(titleColor, "Input");

    Vec2 applePos = applePosList[birdApplePositionId[dbgViewerBirdId]];
    f64 appleOffsetX = applePos.x - birdPos[dbgViewerBirdId].x;
    f64 appleOffsetY = applePos.y - birdPos[dbgViewerBirdId].y;

    ImGui::Text("velX: %g", birdVel[dbgViewerBirdId].x / 1000.0);
    ImGui::Text("velY: %g", birdVel[dbgViewerBirdId].y / 1000.0);
    ImGui::Text("appleOffsetX: %g", appleOffsetX / 2000.0);
    ImGui::Text("appleOffsetY: %g", appleOffsetY / 2000.0);
    ImGui::Text("rot: %g", birdRot[dbgViewerBirdId] / PI);
    ImGui::EndGroup();

    ImGui::Separator();

    ImGui::TextColored(titleColor, "Fitness factors");
    f64 distFactor = 1.0 - (min(vec2Distance(&birdPos[dbgViewerBirdId],
                                             &appleTf[dbgViewerBirdId].pos), 2000) / 2000.0);
    f64 healthFactor = birdHealth[dbgViewerBirdId] / HEALTH_MAX;
    ImGui::Text("distance: %g", distFactor);
    ImGui::Text("health: %g", healthFactor);

    ImGui::Separator();

    ImGui::BeginGroup();
    ImGui::TextColored(titleColor, "Output");
    f32 left = birdInput[dbgViewerBirdId].left;
    f32 right = birdInput[dbgViewerBirdId].right;
    ImGui_ColoredRect(ImVec2(20,20), ImVec4(left/255.0, right/255.0, 0, 1));
    ImGui::SameLine();

    f64 flap = birdInput[dbgViewerBirdId].flap / 255.0;
    ImGui_ColoredRect(ImVec2(20,20), ImVec4(0, flap, flap, 1));
    ImGui::SameLine();

    f64 brake = birdInput[dbgViewerBirdId].brake / 255.0;
    ImGui_ColoredRect(ImVec2(20,20), ImVec4(brake, 0, 0, 1));
    ImGui::EndGroup();

    ImGui::Separator();

    ImGui_BirdImage(dbgViewerBirdId, 100.f);

    ImGui::Separator();

    ImGui::ProgressBar(birdHealth[dbgViewerBirdId]/HEALTH_MAX);

    ImGui::End();
}

void ui_generationViewer()
{
    // last generation statistics window
    ImGui::Begin("Last generation");

    ImGui::Text("Generation %d", lastGenStats.number);
    ImGui::Separator();

    ImGui::TextColored(ImVec4(0, 1, 0, 1), "Fitness max: %g", lastGenStats.maxFitness);
    ImGui::Text("Fitness avg: %g", lastGenStats.avgFitness);

    /*ImGui::TextColored(ImVec4(0.8, 0.8, 0.8, 1), "Deaths by border: %d", lastGenStats.deathsByBorder);
    ImGui::TextColored(ImVec4(0, 0.8, 1, 1), "Deaths by dehydration: %d",
                       lastGenStats.deathsByDehydratation);*/

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

void ui_speciation()
{
    ImVec4 subPopColors[SUBPOP_MAX_COUNT];
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

void ui_simOptions()
{
    ImGui::Begin("Simulation");

    ImGui::SliderInt("time scale", &timeScale, 1, 20);
    ImGui::Checkbox("Show objective lines", &dbgShowObjLines);
    ImGui::Checkbox("Highlight bird", &dbgHightlightBird);
    ImGui::Checkbox("Follow bird", &dbgFollowBird);

    ImGui::Separator();

    if(ImGui::Button("Reset view")) {
        resetView();
    }

    ImGui::SameLine();

    if(ImGui::Button("Reset simulation")) {
        resetSimulation();
    }

    ImGui::End();
}

void doUI()
{
    window.uiNewFrame();
    ImGui::StyleColorsDark();

    ui_simOptions();
    ui_birdViewer();
    ui_generationViewer();
    ui_speciation();
}


void updateNNs()
{
    NeatNN* aliveNN[BIRD_COUNT];
    i32 aliveCount = 0;

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(birdDead[i]) continue;
        aliveNN[aliveCount++] = birdNN[i];
    }

    // setup neural net inputs
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(birdDead[i]) continue;
        Vec2 applePos = applePosList[birdApplePositionId[i]];
        f64 appleOffsetX = applePos.x - birdPos[i].x;
        f64 appleOffsetY = applePos.y - birdPos[i].y;
        f64 velX = birdVel[i].x;
        f64 velY = birdVel[i].y;
        f64 rot = birdRot[i];
        f64 angVel = birdAngularVel[i];

        Vec2 diff = vec2Minus(&applePos, &birdPos[i]);
        Vec2 dir = {cosf(rot), sinf(rot)};
        f32 diffRot = vec2AngleBetween(&dir, &diff);

        f64 inputs[6] = {
            velX / 1000.0,
            velY / 1000.0,
            appleOffsetX / 2000.0,
            appleOffsetY / 2000.0,
            rot / PI,
            birdFlapLeftCd[i] <= 0.0f ? 1.0 : 0.0
            //diffRot / PI,
            //vec2Len(&diff) / 3000.0
        };

        birdNN[i]->setInputs(inputs, arr_count(inputs));
    }

    neatNnPropagate(aliveNN, aliveCount);

    // get neural net output
    const i32 firstOutputId = birdCurGen[0]->inputNodeCount;
    const i32 outputCount = birdCurGen[0]->outputNodeCount;

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(birdDead[i]) continue;
        f64 out[4];
        memmove(out, &birdNN[i]->nodeValues[firstOutputId], sizeof(out[0]) * outputCount);

        // tanh
        out[0] = (out[0] + 1.0) * 0.5;
        out[1] = (out[1] + 1.0) * 0.5;
        out[2] = (out[2] + 1.0) * 0.5;
        out[3] = (out[3] + 1.0) * 0.5;

        assert(out[0] >= 0 && out[0] <= 1.0);
        assert(out[1] >= 0 && out[1] <= 1.0);
        assert(out[2] >= 0 && out[2] <= 1.0);
        assert(out[3] >= 0 && out[3] <= 1.0);

        birdInput[i].left = clamp(out[0] - 0.5, 0.0, 0.5) * 2.0 * 255;
        birdInput[i].right = clamp(out[1] - 0.5, 0.0, 0.5) * 2.0 * 255;
        birdInput[i].flap = clamp(out[2] - 0.5, 0.0, 0.5) * 2.0 * 255;
        birdInput[i].brake = clamp(out[3] - 0.5, 0.0, 0.5) * 2.0 * 255;
    }
}

void updateMechanics()
{
    // wing anim time
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(birdDead[i]) continue;
        birdFlapLeftCd[i] -= FRAME_DT;
        birdFlapRightCd[i] -= FRAME_DT;
        if(birdFlapLeftCd[i] <= 0.0f) {
            birdFlapLeftCd[i] = 0.0f;
        }
        if(birdFlapRightCd[i] <= 0.0f) {
            birdFlapRightCd[i] = 0.0f;
        }
    }

    // check if we touched the apple
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(birdDead[i]) continue;
        Vec2* applePos = &applePosList[birdApplePositionId[i]];
        if(vec2Distance(applePos, &birdPos[i]) < APPLE_RADIUS) {
            birdApplePositionId[i]++;
            birdAppleEatenCount[i]++;
            birdHealth[i] = HEALTH_MAX;
            birdShortestDistToNextApple[i] = vec2Distance(&applePosList[birdApplePositionId[i]],
                    &birdPos[i]);
        }
    }

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(birdDead[i]) continue;
        f32 dist = vec2Distance(&applePosList[birdApplePositionId[i]], &birdPos[i]);
        birdShortestDistToNextApple[i] = min(birdShortestDistToNextApple[i], dist);
    }

    f32 birdDistToNextAppleOld[BIRD_COUNT];
    memmove(birdDistToNextAppleOld, birdDistToNextApple, sizeof(birdDistToNextApple));

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        birdDistToNextApple[i] = vec2Distance(&applePosList[birdApplePositionId[i]], &birdPos[i]);
    }

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(birdDead[i]) continue;

        birdHealth[i] -= FRAME_DT;
        if(birdHealth[i] <= 0.0f) { // death
            birdDead[i] = true;
        }
    }

    // calculate fitness
#if 1
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(birdDead[i]) continue;
        f64 applesFactor = birdAppleEatenCount[i];
        f64 shortDistFactor = APPLE_RADIUS / min(birdShortestDistToNextApple[i], 5000); // 0.0 -> 1.0
        //f64 distFactor = APPLE_RADIUS / min(vec2Distance(&birdPos[i], &appleTf[i].pos), 5000); // 0.0 -> 1.0
        f64 distFactor = 1.0 - (min(vec2Distance(&birdPos[i], &appleTf[i].pos), 2000) / 2000.0); // 0.0 -> 1.0
        f64 healthFactor = birdHealth[i] / HEALTH_MAX;

        //birdFitnessAcc[i] += distFactor * distFactor * distFactor * shortDistFactor * healthFactor;
        //birdFitness[i] = birdFitnessAcc[i] * (applesFactor + 1.0);
        birdFitnessAcc[i] += (healthFactor + distFactor * distFactor) * FRAME_DT;
        birdFitness[i] = birdFitnessAcc[i];
        birdFitness[i] += birdFitness[i] * 0.1 * applesFactor;
    }
#else
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(birdDead[i]) continue;

        if(birdDistToNextAppleOld[i] > birdDistToNextApple[i]) {
           birdFitnessAcc[i] += (birdDistToNextAppleOld[i] - birdDistToNextApple[i]) / 1000.0;
        }

        birdFitness[i] = 1.0 + birdFitnessAcc[i] * (birdAppleEatenCount[i] + 1.0);
    }
#endif

    f64 maxFitness = 0.0;
    f64 totalFitness = 0.0;
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        totalFitness += birdFitness[i];
        if(birdFitness[i] > maxFitness) {
            maxFitness = birdFitness[i];
        }
    }

    curGenStats.avgFitness = totalFitness / BIRD_COUNT;
    curGenStats.maxFitness = maxFitness;

    bool allDead = true;
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(!birdDead[i]) {
            allDead = false;
            break;
        }
    }

    // produce next generation
    if(allDead) {
        nextGeneration();
    }
}

void updatePhysics()
{
#if 0
    // apply bird input
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        u8 flapLeft = (birdFlapLeftCd[i] <= 0.0f) && birdInput[i].left;
        u8 flapRight = (birdFlapRightCd[i] <= 0.0f) && birdInput[i].right;

        Vec2 leftForce = {
            flapLeft * cosf(birdRot[i] +PI * 0.15f) * WING_STRENGTH,
            flapLeft * sinf(birdRot[i] +PI * 0.15f) * WING_STRENGTH,
        };
        Vec2 rightForce = {
            flapRight * cosf(birdRot[i] -PI * 0.15f) * WING_STRENGTH,
            flapRight * sinf(birdRot[i] -PI * 0.15f) * WING_STRENGTH,
        };

        Vec2 totalForce = { leftForce.x + rightForce.x, leftForce.y + rightForce.y };

        if(flapLeft && flapRight) {
            totalForce.x *= 1.5f;
            totalForce.y *= 1.5f;
            birdAngularVel[i] *= 0.5;
        }

        // easier to rectify speed
        f32 vl = vec2Len(&birdVel[i]);
        f32 fl = vec2Len(&totalForce);

        if(vec2Dot(&birdVel[i], &totalForce) < -(vl * fl * 0.5)) {
            totalForce.x += -birdVel[i].x * 0.75;
            totalForce.y += -birdVel[i].y * 0.75;
        }

        birdVel[i].x += totalForce.x;
        birdVel[i].y += totalForce.y;

        // easier to rectify angle
        f32 leftAngStr = WING_STRENGTH_ANGULAR;
        f32 rightAngStr = -WING_STRENGTH_ANGULAR;

        /*if(birdAngularVel[i] < 0.0) {
            leftAngStr -= birdAngularVel[i];
            rightAngStr += birdAngularVel[i] * 0.5;
        }

        if(birdAngularVel[i] > 0.0) {
            rightAngStr -= birdAngularVel[i];
            leftAngStr += birdAngularVel[i] * 0.5;
        }*/

        birdAngularVel[i] += flapLeft * leftAngStr +
                                 flapRight * rightAngStr;

        if(flapLeft) {
            birdFlapLeftCd[i] = WING_FLAP_TIME;
        }
        if(flapRight) {
            birdFlapRightCd[i] = WING_FLAP_TIME;
        }
    }
#endif

    constexpr f32 gravity = 200.f;

    // apply bird input
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(birdDead[i]) continue;
        f64 rotLeft = birdInput[i].left / 255.0;
        f64 rotRight = birdInput[i].right / 255.0;
        f64 flap = 0;
        f64 brake = 0;

        if(birdFlapLeftCd[i] <= 0.0f) {
            if(birdInput[i].brake) {
                brake = birdInput[i].brake / 255.0;
            }
            else if(birdInput[i].flap) {
                flap = birdInput[i].flap / 255.0;
            }
        }

        birdRot[i] += rotRight * WING_STRENGTH_ANGULAR * FRAME_DT +
                      rotLeft  * -WING_STRENGTH_ANGULAR * FRAME_DT;

        if(brake > 0.0) {
            birdRot[i] = brake * TAU;
            birdVel[i].x *= 1.0 - WING_BRAKE;
            birdVel[i].y *= 1.0 - WING_BRAKE;
            birdFlapLeftCd[i] = WING_FLAP_TIME;
            birdFlapRightCd[i] = WING_FLAP_TIME;
        }
        else if(flap > 0.0) {
            Vec2 force = {(f32)(cosf(birdRot[i]) * WING_STRENGTH * FRAME_DT * flap),
                          (f32)(sinf(birdRot[i]) * WING_STRENGTH * FRAME_DT * flap)};

            birdVel[i].x += force.x;
            birdVel[i].y += force.y;
            birdFlapLeftCd[i] = WING_FLAP_TIME;
            birdFlapRightCd[i] = WING_FLAP_TIME;
        }
    }

    // apply gravity and friction to bird velocity
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        birdVel[i].y += gravity * FRAME_DT;
        /*birdVel[i].y *= 1.f - FRICTION_AIR * FRAME_DT;
        birdVel[i].x *= 1.f - FRICTION_AIR * FRAME_DT;*/
    }
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(birdAngularVel[i] > ANGULAR_VELOCITY_MAX) birdAngularVel[i] = ANGULAR_VELOCITY_MAX;
        if(birdAngularVel[i] < -ANGULAR_VELOCITY_MAX) birdAngularVel[i] = -ANGULAR_VELOCITY_MAX;
        birdAngularVel[i] *= 1.f - FRICTION_AIR_ANGULAR * FRAME_DT;
    }

    // apply bird velocity to pos
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        birdPos[i].x += birdVel[i].x * FRAME_DT;
        birdPos[i].y += birdVel[i].y * FRAME_DT;
    }
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        //birdRot[i] += birdAngularVel[i] * FRAME_DT;
        birdRot[i] = fmod(birdRot[i], PI);
    }

    // check for ground collision
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(birdPos[i].y > GROUND_Y) {
            birdPos[i].y = GROUND_Y;
            birdVel[i].x = 0;
            birdVel[i].y = 0;

            if(!birdDead[i]) {
                birdDead[i] = true;
                birdDeadFromGround[i] = true;
                birdFitness[i] *= 0.8; // death from ground penalty
            }
        }
    }
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

struct FitnessPair
{
    i32 id;
    f64 fitness;
};

void nextGeneration()
{
    lastGenStats = curGenStats;
    memmove(pastGenStats, pastGenStats+1, sizeof(pastGenStats) - sizeof(pastGenStats[0]));
    pastGenStats[STATS_HISTORY_COUNT-1] = lastGenStats;

    curGenStats = {};
    curGenStats.number = generationNumber++;

    LOG("#%d maxFitness=%.5f avg=%.5f", generationNumber,
        lastGenStats.maxFitness, lastGenStats.avgFitness);

    neatEvolve(birdCurGen, birdNextGen, birdFitness, BIRD_COUNT, &neatSpec, evolParam, true);

    neatNnDealloc(birdNN);
    neatGenomeAllocMakeNN(birdCurGen, BIRD_COUNT, birdNN);

    neatGenomeComputeNodePos(birdCurGen, BIRD_COUNT);

    resetBirds();
}

void newFrame()
{
    doUI();

    if(dbgFollowBird) {
        viewX = birdPos[dbgViewerBirdId].x - WINDOW_WIDTH * viewZoom * 0.5;
        viewY = birdPos[dbgViewerBirdId].y - WINDOW_HEIGHT * viewZoom * 0.5;
    }

    updateCamera();
    updateNNs();

    updatePhysics();
    updateMechanics();

    // update bird body transform
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        birdBodyTf[i].pos.x = birdPos[i].x;
        birdBodyTf[i].pos.y = birdPos[i].y;
        birdBodyTf[i].rot = birdRot[i] + PI * 0.5;
    }

    const f32 wingUpAngle = -PI * 0.1f;
    const f32 wingDownAngle = PI * 0.6f;

    // update bird wing transform
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        birdLeftWingTf[i].pos.x = birdPos[i].x;
        birdLeftWingTf[i].pos.y = birdPos[i].y;
        birdLeftWingTf[i].rot = birdRot[i] + PI * 0.5 -
            (wingUpAngle + wingDownAngle) * (birdFlapLeftCd[i] / WING_FLAP_TIME);
    }
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        birdRightWingTf[i].pos.x = birdPos[i].x;
        birdRightWingTf[i].pos.y = birdPos[i].y;
        birdRightWingTf[i].rot = birdRot[i] + PI * 0.5 +
            (wingUpAngle + wingDownAngle) * (birdFlapRightCd[i] / WING_FLAP_TIME);
    }

    // update apples position
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        appleTf[i].pos = applePosList[birdApplePositionId[i]];
    }

    // update target lines
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        targetLine[i].p1 = birdPos[i];
        targetLine[i].p2 = appleTf[i].pos;
    }
}

void render()
{
    const Color3 black = {0, 0, 0};
    const Color4 black4 = {0, 0, 0, 0};
    Color3 birdColor[BIRD_COUNT];

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(birdDead[i]) {
            birdColor[i] = black;
            targetLine[i].c1 = black4;
            targetLine[i].c2 = black4;
        }
        else {
            i32 species = birdCurGen[i]->species;
            if(species < 0) species = 0;
            birdColor[i] = speciesColor[species];
            targetLine[i].c1 = {birdColor[i].r, birdColor[i].g, birdColor[i].b, 255};
            targetLine[i].c2 = {birdColor[i].r, birdColor[i].g, birdColor[i].b, 0};
        }
    }

    glClear(GL_COLOR_BUFFER_BIT);

    const f32 gl = -10000000.f;
    const f32 gr = 10000000.f;
    const f32 gt = GROUND_Y;
    const f32 gb = GROUND_Y + 10000.f;
    const Color4 groundColor = {40, 89, 24, 255};

    Quad groundQuad = quadOneColor(gl, gr, gt, gb, groundColor);

    drawQuadBatch(&groundQuad, 1);

    if(dbgShowObjLines) {
        drawLineBatch(targetLine, BIRD_COUNT);
    }

    drawSpriteBatch(tex_birdWing, birdLeftWingTf, birdColor, BIRD_COUNT);
    drawSpriteBatch(tex_birdWing, birdRightWingTf, birdColor, BIRD_COUNT);
    drawSpriteBatch(tex_birdBody, birdBodyTf, birdColor, BIRD_COUNT);

    drawSpriteBatch(tex_apple, appleTf, birdColor, BIRD_COUNT);

    /*Transform pr;
    pr.pos.x = birdPos[0].x + cosf(birdRot[0] -PI * 0.5f +PI * 0.15f) * 40.f;
    pr.pos.y = birdPos[0].y + sinf(birdRot[0] -PI * 0.5f +PI * 0.15f) * 40.f;
    pr.size.x = 10;
    pr.size.y = 10;
    pr.center.x = 5;
    pr.center.y = 5;

    Color3 red = {255, 0, 0};

    drawSpriteBatch(tex_apple, &pr, &red, 1);*/

    if(dbgHightlightBird) {
        f32 halfSize = 50;
        Vec2 hlPos = birdPos[dbgViewerBirdId];
        Quad hlQuad = quadOneColor(hlPos.x - halfSize, hlPos.x + halfSize,
                                   hlPos.y - halfSize, hlPos.y + halfSize,
                                   {255, 0, 0, 128});
        drawQuadBatch(&hlQuad, 1);
    }

    imguiRender();
}

void cleanup()
{
    window.cleanup();
    neatGenomeDealloc(birdCurGen);
    neatGenomeDealloc(birdNextGen);
    neatNnDealloc(birdNN);
}

};

#ifdef _WIN32
int CALLBACK WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
#else
i32 main()
#endif
{
    LOG("Burds [NEAT]");

    randSetSeed(time(NULL));
    timeInit();

    /*for(i32 i = 0; i < 1000; ++i) {
        LOG("%g", randf64(0.0, 2.4578));
    }*/

    //neatTestPropagate();

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
