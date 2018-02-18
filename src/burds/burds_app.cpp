#include "base.h"
#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <gl3w.h>
#include <stdlib.h>
#include <float.h>
#include <assert.h>

#include "sprite.h"
#include "neural.h"
#include "imgui/imgui.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui/imgui_internal.h"
#include "imgui/imgui_sdl2_setup.h"

#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 900
#define FRAMES_PER_SEC 60.0
#define FRAME_DT ((f64)(1.0/FRAMES_PER_SEC))

#define BIRD_COUNT 1024
#define BIRD_TAG_BITS 3
constexpr i32 SUBPOP_MAX_COUNT = 1 << BIRD_TAG_BITS;

#define BIRD_BODY_RATIO 0.5f
#define BIRD_WING_RATIO 1.68125f

#define BIRD_BODY_HEIGHT 50
#define BIRD_BODY_WIDTH (BIRD_BODY_HEIGHT * BIRD_BODY_RATIO)
#define BIRD_WING_WIDTH 50
#define BIRD_WING_HEIGHT (1.f / BIRD_WING_RATIO * BIRD_WING_WIDTH)

#define WING_FLAP_TIME 0.4f

#define FRICTION_AIR_ANGULAR 0.05f
#define FRICTION_AIR 0.05f

#define WING_STRENGTH 5000.f
#define WING_STRENGTH_ANGULAR (PI * 10.0)

#define ANGULAR_VELOCITY_MAX (TAU * 2.0)

#define APPLE_POS_LIST_COUNT 1024
#define APPLE_RADIUS 80.0
#define HEALTH_MAX 10.0

#define GROUND_Y 1000

#define REINSERT_TRUNCATE 0x1
#define REINSERT_DUPLICATE 0x2
#define REINSERT_STRATEGY REINSERT_TRUNCATE

#define NEURAL_NET_LAYERS { 4, 10, 4 }

#define STATS_HISTORY_COUNT 30

struct BirdInput
{
    u8 left, right, flapLight, flapHard, brakeLight, brakeHard;
};

enum {
    MODE_NN_TRAIN=0,
    MODE_HUMAN_PLAY
};

struct Bounds {
    f64 bmin, bmax;
};

struct App {

i32 running;
SDL_Window* window;
SDL_GLContext glContext;

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
i32 birdApplePositionId[BIRD_COUNT];

Vec2 applePosList[APPLE_POS_LIST_COUNT];
Transform appleTf[BIRD_COUNT];

Line targetLine[BIRD_COUNT];

RecurrentNeuralNetDef nnDef;
RecurrentNeuralNet* curGenNN[BIRD_COUNT];
RecurrentNeuralNet* nextGenNN[BIRD_COUNT];
u8 curSpeciesTag[BIRD_COUNT];
u8 nextSpeciesTag[BIRD_COUNT];

i32 birdAppleEatenCount[BIRD_COUNT];
f32 birdShortestDistToNextApple[BIRD_COUNT];
f32 birdLongestDistToNextApple[BIRD_COUNT];
f32 birdMaxHealthAchieved[BIRD_COUNT];
f64 birdDistToNextAppleSum[BIRD_COUNT];
i32 birdDistCheckCount[BIRD_COUNT];

f64 birdFitness[BIRD_COUNT];

Bounds genFitness;
Bounds genAvgDistFactor;
Bounds genShortDistFactor;
Bounds genLongDistFactor;
Bounds genHealthFactor;

f32 viewZoom;
i32 viewX;
i32 viewY;
u8 mouseRightButDown = 0;

i32 mode;

struct ImGuiGLSetup* ims;

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

Bounds outBounds[4];

GeneticEnvRnn genetivEnv = { BIRD_COUNT, BIRD_TAG_BITS, curSpeciesTag, nextSpeciesTag,
                         curGenNN, nextGenNN, &nnDef, birdFitness };

void resetBirdColors()
{
    const u32 colorMax = 0xFF;
    const u32 colorMin = 0x0;

    for(i32 i = 0; i < SUBPOP_MAX_COUNT; ++i) {
        Color3 c;
        c.r = randi64(colorMin, colorMax);
        c.g = randi64(colorMin, colorMax);
        c.b = randi64(colorMin, colorMax);
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

    memset(birdVel, 0, sizeof(birdVel));
    memset(birdAngularVel, 0, sizeof(birdAngularVel));
    memset(birdFlapLeftCd, 0, sizeof(birdFlapLeftCd));
    memset(birdFlapRightCd, 0, sizeof(birdFlapRightCd));
    memset(birdApplePositionId, 0, sizeof(birdApplePositionId));
    memset(birdAppleEatenCount, 0, sizeof(birdAppleEatenCount));
    memset(birdMaxHealthAchieved, 0, sizeof(birdMaxHealthAchieved));
    memset(birdDistToNextAppleSum, 0, sizeof(birdDistToNextAppleSum));
    memset(birdDistCheckCount, 0, sizeof(birdDistCheckCount));
    memset(birdDead, 0, sizeof(birdDead));
    memset(birdFitness, 0, sizeof(birdFitness));

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        //birdRot[i] = PI;
        birdShortestDistToNextApple[i] = 99999.f;
        birdLongestDistToNextApple[i] = 0.f;
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

    Bounds minb = {DBL_MAX, -DBL_MAX};
    genFitness = minb;
    genAvgDistFactor = minb;
    genShortDistFactor = minb;
    genLongDistFactor = minb;
    genHealthFactor = minb;

    outBounds[0] = minb;
    outBounds[1] = minb;
    outBounds[2] = minb;
    outBounds[3] = minb;
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
    generateSpeciesTags(curSpeciesTag, BIRD_COUNT, BIRD_TAG_BITS);
    resetBirdColors();
    resetBirds();
    resetApplePath();

    generationNumber = 0;
    curGenStats = {};
    memset(pastGenStats, 0, sizeof(pastGenStats));
    rnnInitRandom(curGenNN, BIRD_COUNT, &nnDef);

    /*const i32 weightTotalCount = nnDef.weightTotalCount;
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        f64 sum = 0.0;
        for(i32 w = 0; w < weightTotalCount; ++w) {
            sum += curGenNN[i]->weights[w];
        }
        sum *= sum;
        sum *= 1000.0;
        curSpeciesTag[i] = ((u32)sum) & ((1 << BIRD_TAG_BITS) - 1);
    }*/
}

i32 init()
{
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);

    window = SDL_CreateWindow("Burds",
                              SDL_WINDOWPOS_CENTERED,
                              SDL_WINDOWPOS_CENTERED,
                              WINDOW_WIDTH, WINDOW_HEIGHT,
                              SDL_WINDOW_OPENGL);
    running = TRUE;

    if(!window) {
        LOG("ERROR: can't create SDL2 window (%s)",  SDL_GetError());
        return FALSE;
    }

    glContext = SDL_GL_CreateContext(window);
    if(!glContext) {
        LOG("ERROR: can't create OpenGL 3.3 context (%s)",  SDL_GetError());
        return FALSE;
    }

    SDL_GL_SetSwapInterval(0);

    if(gl3w_init()) {
        LOG("ERROR: can't init gl3w");
        return FALSE;
    }

    if(!gl3w_is_supported(3, 3)) {
        LOG("ERROR: OpenGL 3.3 isn't available on this system");
        return FALSE;
    }

    if(!initSpriteState(WINDOW_WIDTH, WINDOW_HEIGHT)) {
        return FALSE;
    }

    ims = imguiInit(WINDOW_WIDTH, WINDOW_HEIGHT, "burds_imgui.ini");
    if(!ims) {
        LOG("ERROR: could not init imgui");
    }

    viewX = -2000;
    viewY = -1000;
    viewZoom = 3.f;

    glClearColor(179.f/255.f, 204.f/255.f, 255.f/255.f, 1.0f);

    tex_birdBody = loadTexture("../bird_body.png");
    tex_birdWing = loadTexture("../wing.png");
    tex_apple = loadTexture("../apple.png");
    if(tex_birdBody == -1 || tex_birdWing == -1 ||
       tex_apple == -1) {
        return FALSE;
    }

    timeScale = 1.0f;


    const i32 layers[] = NEURAL_NET_LAYERS;
    rnnMakeDef(&nnDef, sizeof(layers) / sizeof(layers[0]), layers, 1.f);
    rnnAlloc(curGenNN, BIRD_COUNT, &nnDef);
    rnnAlloc(nextGenNN, BIRD_COUNT, &nnDef);

    mode = MODE_NN_TRAIN;

    resetSimulation();

    return TRUE;
}

void run()
{
    while(running) {
        timept t0 = timeGet();

        SDL_Event event;
        while(SDL_PollEvent(&event)) {
            handleEvent(&event);
        }

        newFrame();
        render();
        SDL_GL_SwapWindow(window);

        const i64 frameDtMicro = FRAME_DT/timeScale * 1000000;
        while(((frameDtMicro - timeToMicrosec(timeGet() - t0)) / 1000) > 1) {
            _mm_pause();
        }
        //LOG("ft=%lld elapsed=%lld remainingMs=%d", frameDtMicro, elapsedMicro, remainingMs);
    }
}

void handleEvent(const SDL_Event* event)
{
    imguiHandleInput(ims, *event);

    if(event->type == SDL_QUIT) {
        running = FALSE;
    }

    if(event->type == SDL_KEYDOWN) {
        if(event->key.keysym.sym == SDLK_ESCAPE) {
            running = FALSE;
        }

        if(event->key.keysym.sym == SDLK_q) {
            birdInput[0].left = 1;
        }
        if(event->key.keysym.sym == SDLK_d) {
            birdInput[0].right = 1;
        }

        if(event->key.keysym.sym == SDLK_r) {
           resetBirds();
        }

        if(event->key.keysym.sym == SDLK_n) {
           resetSimulation();
        }
    }

    if(event->type == SDL_KEYUP) {
        if(event->key.keysym.sym == SDLK_q) {
            birdInput[0].left = 0;
        }
        if(event->key.keysym.sym == SDLK_d) {
            birdInput[0].right = 0;
        }
    }

    if(event->type == SDL_MOUSEBUTTONDOWN) {
        if(event->button.button == SDL_BUTTON_RIGHT) {
            mouseRightButDown = TRUE;
        }
    }

    if(event->type == SDL_MOUSEBUTTONUP) {
        if(event->button.button == SDL_BUTTON_RIGHT) {
            mouseRightButDown = FALSE;
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

    Color3 bc = speciesColor[curSpeciesTag[birdId]];
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

    ImGui::Text("Bird_%04d [%03X]", dbgViewerBirdId, curSpeciesTag[dbgViewerBirdId]);
    ImGui::TextColored(ImVec4(0, 1, 0, 1), "fitness: %g", birdFitness[dbgViewerBirdId]);
    ImGui::TextColored(ImVec4(1, 0.5, 0, 1), "apples: %d", birdAppleEatenCount[dbgViewerBirdId]);

    ImGui::Separator();

    ImGui_RecurrentNeuralNet(curGenNN[dbgViewerBirdId], &nnDef);

    ImGui::Separator();

    ImGui::TextUnformatted("Input");
    f32 left = birdInput[dbgViewerBirdId].left;
    f32 right = birdInput[dbgViewerBirdId].right;
    ImGui_ColoredRect(ImVec2(20,20), ImVec4(left, right, 0, 1));
    ImGui::SameLine();

    f32 flap = (birdInput[dbgViewerBirdId].flapHard ? 2:birdInput[dbgViewerBirdId].flapLight) * 0.5;
    ImGui_ColoredRect(ImVec2(20,20), ImVec4(0, flap, flap, 1));
    ImGui::SameLine();

    f32 brake = (birdInput[dbgViewerBirdId].brakeHard ? 2:birdInput[dbgViewerBirdId].brakeLight) * 0.5;
    ImGui_ColoredRect(ImVec2(20,20), ImVec4(brake, 0, 0, 1));

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

void ui_subPopulations()
{
    ImVec4 subPopColors[SUBPOP_MAX_COUNT];
    for(i32 i = 0; i < SUBPOP_MAX_COUNT; ++i) {
        Color3 sc = speciesColor[i];
        subPopColors[i] = ImVec4(sc.r/255.f, sc.g/255.f, sc.b/255.f, 1.f);
    }

    ImGui_SubPopWindow(&env, subPopColors);
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
    ImGui::StyleColorsDark();

    ui_simOptions();
    ui_birdViewer();
    ui_generationViewer();
    ui_subPopulations();
}


void updateNNs()
{
    RecurrentNeuralNet* aliveNN[BIRD_COUNT];
    i32 aliveCount = 0;

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(birdDead[i]) continue;
        aliveNN[aliveCount++] = curGenNN[i];
    }

    // setup neural net inputs
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        Vec2 applePos = applePosList[birdApplePositionId[i]];
        f64 appleOffsetX = applePos.x - birdPos[i].x;
        f64 appleOffsetY = applePos.y - birdPos[i].y;
        f64 velX = birdVel[i].x;
        f64 velY = birdVel[i].y;
        f64 rot = birdRot[i];
        f64 angVel = birdAngularVel[i];

#if 0
        curGenNN[i]->values[0] = appleOffsetX / 10000.0;
        curGenNN[i]->values[1] = appleOffsetY / 10000.0;
        curGenNN[i]->values[2] = velX / 1000.0;
        curGenNN[i]->values[3] = velY / 1000.0;
        curGenNN[i]->values[4] = rot / TAU;
        // 5 rotations per second is propably high enough
        curGenNN[i]->values[5] = angVel / (TAU * 5.0);
#endif

        Vec2 diff = vec2Minus(&applePos, &birdPos[i]);
        Vec2 dir = {cosf(rot), sinf(rot)};
        f32 diffRot = vec2AngleBetween(&dir, &diff);
        curGenNN[i]->values[0] = diffRot / PI;
        curGenNN[i]->values[1] = velX / 1000.0;
        curGenNN[i]->values[2] = velY / 1000.0;
        curGenNN[i]->values[3] = vec2Len(&diff) / 3000.0;
    }

    rnnPropagate(aliveNN, aliveCount, &nnDef);

    // get neural net output
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(birdDead[i]) continue;
        f64* output = curGenNN[i]->output;
        birdInput[i].left = output[0] > 0.5;
        birdInput[i].right = output[1] > 0.5;

        birdInput[i].flapLight = output[2] > 0.33 && output[2] <= 0.66;
        birdInput[i].flapHard = output[2] > 0.66;

        birdInput[i].brakeLight = output[3] > 0.33 && output[3] <= 0.66;
        birdInput[i].brakeHard = output[3] > 0.66;

        for(i32 o = 0; o < 4; ++o) {
            outBounds[o].bmin = min(outBounds[o].bmin, output[o]);
            outBounds[o].bmax = max(outBounds[o].bmax, output[o]);
        }
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
            birdLongestDistToNextApple[i] = birdShortestDistToNextApple[i];
            birdDistToNextAppleSum[i] = 0;
            birdDistCheckCount[i] = 0;
        }
    }

    // update shortest/longest distance to next apple
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        f32 dist = vec2Distance(&applePosList[birdApplePositionId[i]], &birdPos[i]);
        birdShortestDistToNextApple[i] = min(birdShortestDistToNextApple[i], dist);
        birdLongestDistToNextApple[i] = max(birdShortestDistToNextApple[i], dist);
        birdDistToNextAppleSum[i] += dist;
        birdDistCheckCount[i]++;
    }

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(birdDead[i]) continue;

        birdHealth[i] -= FRAME_DT;
        if(birdHealth[i] <= 0.0f) { // death
            birdDead[i] = true;
        }
    }

    // update maximum health achieved
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        birdMaxHealthAchieved[i] = max(birdMaxHealthAchieved[i], birdHealth[i]);
    }

    // calculate fitness
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(birdDead[i]) continue;
        f64 applesFactor = birdAppleEatenCount[i];
        f64 shortDistFactor = APPLE_RADIUS / min(birdShortestDistToNextApple[i], 5000); // 0.0 -> 1.0
        f64 longDistFactor = birdLongestDistToNextApple[i] / 5000.0; // 0.0 -> inf
        f64 deathDistFactor = APPLE_RADIUS * 1000.0 / vec2Distance(&birdPos[i],
                              &appleTf[i].pos); // 0.0 -> 1.0
        f64 distFactor = APPLE_RADIUS / min(vec2Distance(&birdPos[i], &appleTf[i].pos), 5000); // 0.0 -> 1.0
        f64 avgDistFactor = (birdDistToNextAppleSum[i] / birdDistCheckCount[i]) /
                            5000.0;
        Vec2 nVel = vec2Normalize(&birdVel[i]);
        Vec2 diff = vec2Minus(&appleTf[i].pos, &birdPos[i]);
        Vec2 nDiff = vec2Normalize(&diff);

        f64 distFactor3 = (distFactor * distFactor * distFactor);
        f64 dot = (vec2Dot(&nVel, &nDiff) + 1.0) * 0.5;

        //birdFitness[i] += applesFactor * 2.0 + (distFactor * distFactor * distFactor);
        //birdFitness[i] += applesFactor * 2.0 + dot + shortDistFactor;
        f64 healthFactor = birdHealth[i]/HEALTH_MAX * (applesFactor > 0.0);
        birdFitness[i] += applesFactor * 2.0 + (shortDistFactor * distFactor);
        /*birdFitness[i] += applesFactor * 2.0 + distFactor * 0.2 +
                (1.0 - birdHealth[i]/HEALTH_MAX) * distFactor;*/

        genFitness.bmin = min(genFitness.bmin, birdFitness[i]);
        genFitness.bmax = max(genFitness.bmax, birdFitness[i]);
        genAvgDistFactor.bmin = min(genAvgDistFactor.bmin, avgDistFactor);
        genAvgDistFactor.bmax = max(genAvgDistFactor.bmax, avgDistFactor);
        genShortDistFactor.bmin = min(genShortDistFactor.bmin, shortDistFactor);
        genShortDistFactor.bmax = max(genShortDistFactor.bmax, shortDistFactor);
        genLongDistFactor.bmin = min(genLongDistFactor.bmin, longDistFactor);
        genLongDistFactor.bmax = max(genLongDistFactor.bmax, longDistFactor);
        genHealthFactor.bmin = min(genHealthFactor.bmin, healthFactor);
        genHealthFactor.bmax = max(genHealthFactor.bmax, healthFactor);
    }

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
        u8 rotLeft = birdInput[i].left;
        u8 rotRight = birdInput[i].right;
        u8 flap = 0;
        if(birdFlapLeftCd[i] <= 0.0f) {
            if(birdInput[i].flapHard) {
                flap = 2;
            }
            else if(birdInput[i].flapLight) {
                flap = 1;
            }
        }
        u8 brake = 0;
        if(birdInput[i].brakeHard) {
            brake = 2;
        }
        else if(birdInput[i].brakeLight) {
            brake = 1;
        }

        birdRot[i] += rotRight * WING_STRENGTH_ANGULAR * FRAME_DT +
                      rotLeft  * -WING_STRENGTH_ANGULAR * FRAME_DT;

        if(brake) {
            birdVel[i].x *= 1.0 - 5.0 * brake * FRAME_DT;
            birdVel[i].y *= 1.0 - 5.0 * brake * FRAME_DT;
        }
        else {
            if(flap) {
                Vec2 force = {(f32)(cosf(birdRot[i]) * WING_STRENGTH * FRAME_DT * flap),
                              (f32)(sinf(birdRot[i]) * WING_STRENGTH * FRAME_DT * flap)};

                birdVel[i].x += force.x;
                birdVel[i].y += force.y;
                birdFlapLeftCd[i] = WING_FLAP_TIME;
                birdFlapRightCd[i] = WING_FLAP_TIME;
            }
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
        if(birdRot[i] > TAU) birdRot[i] -= TAU;
        else if(birdRot[i] < TAU) birdRot[i] += TAU;
    }

    // check for ground collision
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(birdPos[i].y > GROUND_Y) {
            birdPos[i].y = GROUND_Y;
            birdVel[i].x = 0;
            birdVel[i].y = 0;
            birdDead[i] = 1;
            //birdAngularVel[i] = 0;
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
    for(i32 o = 0; o < 4; ++o) {
        LOG("output[%d] min=%g max=%g", o, outBounds[o].bmin, outBounds[o].bmax);
    }

    lastGenStats = curGenStats;
    memmove(pastGenStats, pastGenStats+1, sizeof(pastGenStats) - sizeof(pastGenStats[0]));
    pastGenStats[STATS_HISTORY_COUNT-1] = lastGenStats;

    curGenStats = {};
    curGenStats.number = generationNumber++;

    LOG("#%d maxFitness=%.5f avg=%.5f", generationNumber,
        lastGenStats.maxFitness, lastGenStats.avgFitness);
    //LOG("out1[%.3f, %.3f] out1[%.3f, %.3f]", outMin1, outMax1, outMin2, outMax2);
    LOG("fitness=[%.5f, %.5f] shortDist=[%.5f, %.5f] avgDist=[%.5f, %.5f] longDist=[%.5f, %.5f]"
        " health=[%.5f, %.5f]",
        genFitness.bmin, genFitness.bmax,
        genShortDistFactor.bmin, genShortDistFactor.bmax,
        genAvgDistFactor.bmin, genAvgDistFactor.bmax,
        genLongDistFactor.bmin, genLongDistFactor.bmax,
        genHealthFactor.bmin, genHealthFactor.bmax
        );


    evolutionSSS1(&genetivEnv);

    resetBirds();
}

void newFrame()
{
    imguiUpdate(ims);
    doUI();

    if(dbgFollowBird) {
        viewX = birdPos[dbgViewerBirdId].x - WINDOW_WIDTH * viewZoom * 0.5;
        viewY = birdPos[dbgViewerBirdId].y - WINDOW_HEIGHT * viewZoom * 0.5;
    }

    updateCamera();

    if(mode == MODE_NN_TRAIN) {
        updateNNs();
    }
    else {
        birdInput[0].left = 0;
        birdInput[0].right = 0;
    }

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
            birdColor[i] = speciesColor[curSpeciesTag[i]];
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
    free(ims);
    rnnDealloc(curGenNN[0]);
    rnnDealloc(nextGenNN[0]);
}

};

#ifdef _WIN32
int CALLBACK WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
#else
i32 main()
#endif
{
    LOG("Burds");

    randSetSeed(time(NULL));
    timeInit();

    testPropagateRNN();

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
