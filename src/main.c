#include "base.h"
#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <gl3w.h>
#include <stdlib.h>
#include <float.h>
#include <assert.h>

#include "sprite.h"
#include "neural.h"
#include "imgui/imgui_sdl2_setup.h"

#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 900
#define UPDATE_DT (1.0/60.0)

#define BIRD_COUNT 1024

#define BIRD_BODY_RATIO 0.5f
#define BIRD_WING_RATIO 1.68125f

#define BIRD_BODY_HEIGHT 50
#define BIRD_BODY_WIDTH (BIRD_BODY_HEIGHT * BIRD_BODY_RATIO)
#define BIRD_WING_WIDTH 50
#define BIRD_WING_HEIGHT (1.f / BIRD_WING_RATIO * BIRD_WING_WIDTH)

#define WING_FLAP_TIME 0.25f

#define FRICTION_AIR_ANGULAR 0.1f
#define FRICTION_AIR 0.05f

#define WING_STRENGTH 200.f
#define WING_STRENGTH_ANGULAR (PI * 0.15f)

#define APPLE_POS_LIST_COUNT 1024
#define APPLE_RADIUS 80.f

#define GROUND_Y 1000

#define CROSS_STRAT_RANDOM 0x1
#define CROSS_STRAT_FAVOR_BEST 0x2
#define CROSSOVER_STRATEGY CROSS_STRAT_FAVOR_BEST

#define NEURAL_NET_LAYERS { 6, 6, 2 }

static f64 outMin1 = 1000.0;
static f64 outMax1 = -1000.0;
static f64 outMin2 = 1000.0;
static f64 outMax2 = -1000.0;

typedef struct
{
    u8 left, right;
} BirdInput;

enum {
    MODE_NN_TRAIN=0,
    MODE_HUMAN_PLAY
};

struct App
{
    i32 running;
    SDL_Window* window;
    SDL_GLContext glContext;

    i32 tex_birdBody;
    i32 tex_birdWing;
    i32 tex_apple;

    Transform birdBodyTf[BIRD_COUNT];
    Transform birdLeftWingTf[BIRD_COUNT];
    Transform birdRightWingTf[BIRD_COUNT];
    Color3 birdColor[BIRD_COUNT];

    Vec2 birdPos[BIRD_COUNT];
    Vec2 birdVel[BIRD_COUNT];
    f32 birdRot[BIRD_COUNT];
    f32 birdAngularVel[BIRD_COUNT];
    f32 birdFlapLeftAnimTime[BIRD_COUNT];
    f32 birdFlapRightAnimTime[BIRD_COUNT];
    BirdInput birdInput[BIRD_COUNT];

    f32 birdHealth[BIRD_COUNT];
    i32 birdApplePositionId[BIRD_COUNT];

    Vec2 applePosList[APPLE_POS_LIST_COUNT];
    Transform appleTf[BIRD_COUNT];

    Line targetLine[BIRD_COUNT];

    NeuralNetDef nnDef;
    NeuralNet* birdNN[BIRD_COUNT];
    NeuralNet* newGenNN[BIRD_COUNT];
    u8* birdNNData;
    u8* newGenNNData;

    i32 birdAppleEatenCount[BIRD_COUNT];
    f32 birdShortestDistToNextApple[BIRD_COUNT];
    f32 birdShortestDistToNextAppleX[BIRD_COUNT];
    f32 birdShortestDistToNextAppleY[BIRD_COUNT];
    f32 birdFitness[BIRD_COUNT];
    f64 genTotalFitness;

    f32 bestFitness[BIRD_COUNT];
    f64 bestTotalFitness;
    Color3 newGenColor[BIRD_COUNT];

    f32 viewZoom;
    i32 viewX;
    i32 viewY;
    u8 mouseRightButDown;

    i32 mode;
    i32 genNumber;

    struct ImGuiGLSetup* ims;

    f32 timeScale;
} app;

typedef i64 timept;
timept startCounter;
i64 PERFORMANCE_FREQUENCY;

void timeInit()
{
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    startCounter = (i64)li.QuadPart;
    QueryPerformanceFrequency(&li);
    PERFORMANCE_FREQUENCY = (i64)li.QuadPart;
    LOG("performanceFrequency=%llu", (i64)li.QuadPart);
}

inline timept timeGet()
{
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return ((timept)li.QuadPart - startCounter);
}

inline f64 timeToSeconds(i64 delta)
{
    return delta / (f64)PERFORMANCE_FREQUENCY;
}

void resetBirds()
{
    const u32 colorMax = 0xFF;
    const u32 colorMin = 0x2;
    const u32 colorDelta = colorMax - colorMin;

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        Color3 c;
        c.r = (rand() % colorDelta) + colorMin;
        c.g = (rand() % colorDelta) + colorMin;
        c.b = (rand() % colorDelta) + colorMin;
        app.birdColor[i] = c;
    }

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdPos[i].x = 0;
        app.birdPos[i].y = GROUND_Y-100;
    }

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdBodyTf[i].size.x = BIRD_BODY_WIDTH;
        app.birdBodyTf[i].size.y = BIRD_BODY_HEIGHT;
        app.birdBodyTf[i].center.x = BIRD_BODY_WIDTH * 0.5f;
        app.birdBodyTf[i].center.y = BIRD_BODY_HEIGHT * 0.5f;
    }

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdLeftWingTf[i].size.x = -BIRD_WING_WIDTH;
        app.birdLeftWingTf[i].size.y = BIRD_WING_HEIGHT;
        app.birdLeftWingTf[i].center.x = 0;
        app.birdLeftWingTf[i].center.y = BIRD_WING_HEIGHT * 0.6f;
    }
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdRightWingTf[i].size.x = BIRD_WING_WIDTH;
        app.birdRightWingTf[i].size.y = BIRD_WING_HEIGHT;
        app.birdRightWingTf[i].center.x = 0;
        app.birdRightWingTf[i].center.y = BIRD_WING_HEIGHT * 0.6f;
    }

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdHealth[i] = 6.f; // seconds
    }

    memset(app.birdVel, 0, sizeof(app.birdVel));
    memset(app.birdRot, 0, sizeof(app.birdRot));
    memset(app.birdAngularVel, 0, sizeof(app.birdAngularVel));
    memset(app.birdFlapLeftAnimTime, 0, sizeof(app.birdFlapLeftAnimTime));
    memset(app.birdFlapRightAnimTime, 0, sizeof(app.birdFlapRightAnimTime));
    memset(app.birdApplePositionId, 0, sizeof(app.birdApplePositionId));
    memset(app.birdAppleEatenCount, 0, sizeof(app.birdAppleEatenCount));

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        //app.birdRot[i] = PI;
        app.birdShortestDistToNextApple[i] = 99999.f;
        app.birdShortestDistToNextAppleX[i] = 99999.f;
        app.birdShortestDistToNextAppleY[i] = 99999.f;
    }

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.appleTf[i].size.x = 80;
        app.appleTf[i].size.y = 80;
        app.appleTf[i].center.x = 40;
        app.appleTf[i].center.y = 40;
    }

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        memmove(&app.targetLine[i].c1, &app.birdColor[i], 3);
        app.targetLine[i].c1.a = 128;
        memmove(&app.targetLine[i].c2, &app.birdColor[i], 3);
        app.targetLine[i].c2.a = 0;
    }

    outMin1 = 1000.0;
    outMax1 = -1000.0;
    outMin2 = 1000.0;
    outMax2 = -1000.0;

    app.genTotalFitness = 0.0;
}

void resetApplePath()
{
    i32 spawnOriginX = 0;
    i32 spawnOriginY = 0;
    const i32 spawnRadius = 2000;

    for(i32 i = 0; i < APPLE_POS_LIST_COUNT; ++i) {
        app.applePosList[i].x = spawnOriginX - spawnRadius + (rand() % spawnRadius) * 2;
        app.applePosList[i].y = spawnOriginY - spawnRadius + (rand() % spawnRadius) * 2;
        if(app.applePosList[i].y >= GROUND_Y - (spawnRadius / 2)) {
            app.applePosList[i].y = spawnOriginY - spawnRadius;
        }
        spawnOriginX = app.applePosList[i].x;
        spawnOriginY = app.applePosList[i].y;
    }
}

i32 init()
{
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);

    app.window = SDL_CreateWindow("Burds",
                                  SDL_WINDOWPOS_CENTERED,
                                  SDL_WINDOWPOS_CENTERED,
                                  WINDOW_WIDTH, WINDOW_HEIGHT,
                                  SDL_WINDOW_OPENGL);
    app.running = TRUE;

    if(!app.window) {
        LOG("ERROR: can't create SDL2 window (%s)",  SDL_GetError());
        return FALSE;
    }

    app.glContext = SDL_GL_CreateContext(app.window);
    if(!app.glContext) {
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

    app.ims = imguiInit(WINDOW_WIDTH, WINDOW_HEIGHT);
    if(!app.ims) {
        LOG("ERROR: could not init imgui");
    }

    app.viewX = -2000;
    app.viewY = -1000;
    app.viewZoom = 3.f;

    glClearColor(179.f/255.f, 204.f/255.f, 255.f/255.f, 1.0f);

    app.tex_birdBody = loadTexture("../bird_body.png");
    app.tex_birdWing = loadTexture("../wing.png");
    app.tex_apple = loadTexture("../apple.png");
    if(app.tex_birdBody == -1 || app.tex_birdWing == -1 ||
       app.tex_apple == -1) {
        return FALSE;
    }

    app.timeScale = 1.0f;


    const i32 layers[] = NEURAL_NET_LAYERS;
    makeNeuralNetDef(&app.nnDef, sizeof(layers) / sizeof(i32), layers);

    app.birdNNData = allocNeuralNets(app.birdNN, BIRD_COUNT, &app.nnDef);
    app.newGenNNData = allocNeuralNets(app.newGenNN, BIRD_COUNT, &app.nnDef);

    resetBirds();
    resetApplePath();

    app.mode = MODE_NN_TRAIN;
    initNeuralNets(app.birdNN, BIRD_COUNT, &app.nnDef);
    app.genNumber = 0;

    return TRUE;
}

void handleEvent(const SDL_Event* event)
{
    imguiHandleInput(app.ims, *event);

    if(event->type == SDL_QUIT) {
        app.running = FALSE;
    }

    if(event->type == SDL_KEYDOWN) {
        if(event->key.keysym.sym == SDLK_ESCAPE) {
            app.running = FALSE;
        }

        if(event->key.keysym.sym == SDLK_q) {
            app.birdInput[0].left = 1;
        }
        if(event->key.keysym.sym == SDLK_d) {
            app.birdInput[0].right = 1;
        }

        if(event->key.keysym.sym == SDLK_r) {
           resetBirds();
        }

        if(event->key.keysym.sym == SDLK_n) {
           resetBirds();
           resetApplePath();
           initNeuralNets(app.birdNN, BIRD_COUNT, &app.nnDef);
           app.genNumber = 0;
        }
    }

    if(event->type == SDL_KEYUP) {
        if(event->key.keysym.sym == SDLK_q) {
            app.birdInput[0].left = 0;
        }
        if(event->key.keysym.sym == SDLK_d) {
            app.birdInput[0].right = 0;
        }
    }

    if(event->type == SDL_MOUSEBUTTONDOWN) {
        if(event->button.button == SDL_BUTTON_RIGHT) {
            app.mouseRightButDown = TRUE;
        }
    }

    if(event->type == SDL_MOUSEBUTTONUP) {
        if(event->button.button == SDL_BUTTON_RIGHT) {
            app.mouseRightButDown = FALSE;
        }
    }

    if(event->type == SDL_MOUSEMOTION) {
        if(app.mouseRightButDown) {
            app.viewX -= event->motion.xrel * max(app.viewZoom, 1.f);
            app.viewY -= event->motion.yrel * max(app.viewZoom, 1.f);
        }
    }

    if(event->type == SDL_MOUSEWHEEL) {
        if(event->wheel.y > 0) {
            app.viewZoom *= 0.90f * event->wheel.y;
        }
        else if(event->wheel.y < 0) {
            app.viewZoom *= 1.10f * -event->wheel.y;
        }
    }
}

void updateBirdInputNN(f64 delta)
{
    // setup neural net inputs
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        Vec2 applePos = app.applePosList[app.birdApplePositionId[i]];
        f64 appleOffsetX = applePos.x - app.birdPos[i].x;
        f64 appleOffsetY = applePos.y - app.birdPos[i].y;
        f64 velX = app.birdVel[i].x;
        f64 velY = app.birdVel[i].y;
        f64 rot = app.birdRot[i];
        f64 angVel = app.birdAngularVel[i];

#if 0
        app.birdNN[i]->neurons[0].value = appleOffsetX;
        app.birdNN[i]->neurons[1].value = appleOffsetY;
        app.birdNN[i]->neurons[2].value = velX;
        app.birdNN[i]->neurons[3].value = velY;
        app.birdNN[i]->neurons[4].value = rot;
        app.birdNN[i]->neurons[4].value = angVel;
#else // "normalized"
        app.birdNN[i]->neurons[0].value = appleOffsetX / 5000.0;
        app.birdNN[i]->neurons[1].value = appleOffsetY / 5000.0;
        app.birdNN[i]->neurons[2].value = velX / 1000.0;
        app.birdNN[i]->neurons[3].value = velY / 1000.0;
        app.birdNN[i]->neurons[4].value = (rot-PI*0.25f) / TAU;
        app.birdNN[i]->neurons[5].value = angVel / 1000.0;

        /*f32 leftEyeDir = rot -PI * 0.25f -PI * 0.15f;
        f32 rightEyeDir = rot -PI * 0.25f +PI * 0.15f;
        const f32 eyeDist = 40.f;
        Vec2 leftEye = { app.birdPos[i].x + cosf(leftEyeDir) * eyeDist,
                         app.birdPos[i].y + sinf(leftEyeDir) * eyeDist};
        Vec2 rightEye = { app.birdPos[i].x + cosf(rightEyeDir) * eyeDist,
                          app.birdPos[i].y + sinf(rightEyeDir) * eyeDist};
        app.birdNN[i]->neurons[0].value = vec2Distance(&leftEye, &app.birdPos[i]) / 5000.0;
        app.birdNN[i]->neurons[1].value = vec2Distance(&rightEye, &app.birdPos[i]) / 5000.0;
        app.birdNN[i]->neurons[2].value = velX / 1000.0;
        app.birdNN[i]->neurons[3].value = velY / 1000.0;
        app.birdNN[i]->neurons[4].value = angVel / 1000.0;*/
#endif
    }

    propagateNeuralNets(app.birdNN, BIRD_COUNT, &app.nnDef);

    // get neural net output
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        f64 out1 = app.birdNN[i]->neurons[app.nnDef.neuronCount-2].value;
        f64 out2 = app.birdNN[i]->neurons[app.nnDef.neuronCount-1].value;
        outMin1 = min(out1, outMin1);
        outMin2 = min(out2, outMin2);
        outMax1 = max(out1, outMax1);
        outMax2 = max(out2, outMax2);
    }


    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdInput[i].left = app.birdNN[i]->neurons[app.nnDef.neuronCount-2].value > 0.0;
        app.birdInput[i].right = app.birdNN[i]->neurons[app.nnDef.neuronCount-1].value > 0.0;
    }

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(app.birdHealth[i] <= 0.0f) {
            app.birdInput[i].left = 0;
            app.birdInput[i].right = 0;
        }
    }
}

void updateBirdPhysics(f64 delta)
{
    // apply bird input
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        u8 flapLeft = (app.birdFlapLeftAnimTime[i] <= 0.0f) && app.birdInput[i].left;
        u8 flapRight = (app.birdFlapRightAnimTime[i] <= 0.0f) && app.birdInput[i].right;

        Vec2 leftForce = {
            flapLeft * cosf(app.birdRot[i] -PI * 0.5f +PI * 0.15f) * WING_STRENGTH,
            flapLeft * sinf(app.birdRot[i] -PI * 0.5f +PI * 0.15f) * WING_STRENGTH,
        };
        Vec2 rightForce = {
            flapRight * cosf(app.birdRot[i] -PI * 0.5f -PI * 0.15f) * WING_STRENGTH,
            flapRight * sinf(app.birdRot[i] -PI * 0.5f -PI * 0.15f) * WING_STRENGTH,
        };
        app.birdVel[i].x += leftForce.x + rightForce.x;
        app.birdVel[i].y += leftForce.y + rightForce.y;
        app.birdAngularVel[i] += flapLeft * WING_STRENGTH_ANGULAR +
                                 flapRight * -WING_STRENGTH_ANGULAR;

        if(flapLeft) {
            app.birdFlapLeftAnimTime[i] = WING_FLAP_TIME;
        }
        if(flapRight) {
            app.birdFlapRightAnimTime[i] = WING_FLAP_TIME;
        }
    }

    // wing anim time
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(app.birdHealth[i] <= 0.0f) continue;
        app.birdFlapLeftAnimTime[i] -= delta;
        app.birdFlapRightAnimTime[i] -= delta;
        if(app.birdFlapLeftAnimTime[i] <= 0.0f) {
            app.birdFlapLeftAnimTime[i] = 0.0f;
        }
        if(app.birdFlapRightAnimTime[i] <= 0.0f) {
            app.birdFlapRightAnimTime[i] = 0.0f;
        }
    }

    const f32 gravity = 200.f;

    // apply gravity and friction to bird velocity
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdVel[i].y += gravity * delta;
        app.birdVel[i].y *= 1.f - FRICTION_AIR * delta;
        app.birdVel[i].x *= 1.f - FRICTION_AIR * delta;
    }
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdAngularVel[i] *= 1.f - FRICTION_AIR_ANGULAR * delta;
    }

    // apply bird velocity to pos
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdPos[i].x += app.birdVel[i].x * delta;
        app.birdPos[i].y += app.birdVel[i].y * delta;
    }
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdRot[i] += app.birdAngularVel[i] * delta;
        if(app.birdRot[i] > TAU) app.birdRot[i] -= TAU;
        if(app.birdRot[i] < TAU) app.birdRot[i] += TAU;
    }

    // check for ground collision
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(app.birdPos[i].y > GROUND_Y) {
            app.birdPos[i].y = GROUND_Y;
            app.birdVel[i].x = 0;
            app.birdVel[i].y = 0;
            //app.birdAngularVel[i] = 0;
        }
    }

    // update bird body transform
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdBodyTf[i].pos.x = app.birdPos[i].x;
        app.birdBodyTf[i].pos.y = app.birdPos[i].y;
        app.birdBodyTf[i].rot = app.birdRot[i];
    }

    const f32 wingUpAngle = -PI * 0.1f;
    const f32 wingDownAngle = PI * 0.4f;

    // update bird wing transform
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdLeftWingTf[i].pos.x = app.birdPos[i].x;
        app.birdLeftWingTf[i].pos.y = app.birdPos[i].y;
        app.birdLeftWingTf[i].rot = app.birdRot[i] -
            (wingUpAngle + wingDownAngle) * (app.birdFlapLeftAnimTime[i] / WING_FLAP_TIME);
    }
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdRightWingTf[i].pos.x = app.birdPos[i].x;
        app.birdRightWingTf[i].pos.y = app.birdPos[i].y;
        app.birdRightWingTf[i].rot = app.birdRot[i] +
            (wingUpAngle + wingDownAngle) * (app.birdFlapRightAnimTime[i] / WING_FLAP_TIME);
    }

    // check if we touched the apple
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        Vec2* applePos = &app.applePosList[app.birdApplePositionId[i]];
        if(app.birdHealth[i] > 0.0f && vec2Distance(applePos, &app.birdPos[i]) < APPLE_RADIUS) {
            app.birdApplePositionId[i]++;
            app.birdAppleEatenCount[i]++;
            app.birdHealth[i] += 5.f; // seconds
            app.birdShortestDistToNextApple[i] = vec2Distance(&app.applePosList[app.birdApplePositionId[i]],
                    &app.birdPos[i]);
            app.birdShortestDistToNextAppleX[i] = fabs(app.applePosList[app.birdApplePositionId[i]].x -
                    app.birdPos[i].x);
            app.birdShortestDistToNextAppleY[i] = fabs(app.applePosList[app.birdApplePositionId[i]].y -
                    app.birdPos[i].y);
        }
    }

    // update shortest distance to next apple
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdShortestDistToNextApple[i] = min(app.birdShortestDistToNextApple[i],
            vec2Distance(&app.applePosList[app.birdApplePositionId[i]], &app.birdPos[i]));
        app.birdShortestDistToNextAppleX[i] = min(app.birdShortestDistToNextAppleX[i],
            fabs(app.applePosList[app.birdApplePositionId[i]].x - app.birdPos[i].x));
        app.birdShortestDistToNextAppleY[i] = min(app.birdShortestDistToNextAppleY[i],
            fabs(app.applePosList[app.birdApplePositionId[i]].y - app.birdPos[i].y));
    }

    // update apples position
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.appleTf[i].pos = app.applePosList[app.birdApplePositionId[i]];
    }

    // update target lines
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.targetLine[i].p1 = app.birdPos[i];
        app.targetLine[i].p2 = app.appleTf[i].pos;
    }
}

void updateCamera(f64 delta)
{
    if(app.mouseRightButDown) {
        SDL_SetRelativeMouseMode(TRUE);
    }
    else {
        SDL_SetRelativeMouseMode(FALSE);
    }
    setView(app.viewX, app.viewY, WINDOW_WIDTH*app.viewZoom, WINDOW_HEIGHT*app.viewZoom);
}

i32 selectOneParent(const i32 maxCount)
{
    return rand()%maxCount;
    i32 r = ((u32)rand() << 16) + (u32)rand();
    i32 b = r % (i32)(app.bestTotalFitness * 1000.0);

    f64 spinTotal = 0;
    for(i32 i = 0; i < maxCount; ++i) {
        spinTotal += app.bestFitness[i] * 1000.0;
        if(spinTotal > b) {
            return i;
        }
    }

    return maxCount-1;
}

void crossover(i32 id, i32 parentA, i32 parentB)
{
#if 1
    u8 synapseCross[512];
    for(i32 s = 0; s < app.nnDef.synapseTotalCount; ++s) {
        synapseCross[s] = rand () % 3; // [0,1,2]
    }

    i32 curLayerNeuronIdOff = 0;
    i32 synapseId = 0;
    for(i32 l = 1; l < app.nnDef.layerCount; ++l) {
        const i32 prevLayerNeuronCount = app.nnDef.layerNeuronCount[l-1];
        curLayerNeuronIdOff += prevLayerNeuronCount;

        for(i32 n = 0; n < app.nnDef.layerNeuronCount[l]; ++n) {
            for(i32 s = 0; s < prevLayerNeuronCount; ++s) {
                // get weight from parent A
                if(synapseCross[synapseId] == 0) {
                    app.newGenNN[id]->neurons[curLayerNeuronIdOff + n].synapseWeight[s] =
                        app.newGenNN[parentA]->neurons[curLayerNeuronIdOff + n].synapseWeight[s];
                }
                // get weight from parent B
                else if(synapseCross[synapseId] == 1) {
                    app.newGenNN[id]->neurons[curLayerNeuronIdOff + n].synapseWeight[s] =
                        app.newGenNN[parentB]->neurons[curLayerNeuronIdOff + n].synapseWeight[s];
                }
                // both
                else {
                    app.newGenNN[id]->neurons[curLayerNeuronIdOff + n].synapseWeight[s] =
                    (app.newGenNN[parentA]->neurons[curLayerNeuronIdOff + n].synapseWeight[s] +
                     app.newGenNN[parentB]->neurons[curLayerNeuronIdOff + n].synapseWeight[s])
                     * 0.5;
                }
                synapseId++;
            }
        }
    }
#else
    f64 synapseCross[512];
    for(i32 s = 0; s < app.nnDef.synapseTotalCount; ++s) {
        synapseCross[s] = randf64(0.0, 1.0);
    }

    i32 curLayerNeuronIdOff = 0;
    i32 synapseId = 0;
    for(i32 l = 1; l < app.nnDef.layerCount; ++l) {
        const i32 prevLayerNeuronCount = app.nnDef.layerNeuronCount[l-1];
        curLayerNeuronIdOff += prevLayerNeuronCount;

        for(i32 n = 0; n < app.nnDef.layerNeuronCount[l]; ++n) {
            for(i32 s = 0; s < prevLayerNeuronCount; ++s) {
                f64 paw = app.newGenNN[parentA]->neurons[curLayerNeuronIdOff + n].synapseWeight[s];
                f64 pbw = app.newGenNN[parentB]->neurons[curLayerNeuronIdOff + n].synapseWeight[s];
                app.newGenNN[id]->neurons[curLayerNeuronIdOff + n].synapseWeight[s] =
                    paw * synapseCross[s] + pbw * (1.0 - synapseCross[s]);
                synapseId++;
            }
        }
    }
#endif
}

void crossoverRandom(f32 percent)
{
    /*
    f64 lowestFit = 9999999;
    f64 highestFit = -9999999;
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        lowestFit = min(lowestFit, app.birdFitness[i]);
        highestFit = max(highestFit, app.birdFitness[i]);
    }

    f64 median = percent * (lowestFit + highestFit);

    LOG("lowestFit=%.5f highestFit=%.5f median=%.5f", lowestFit, highestFit, median);

    const i32 maxBest = BIRD_COUNT * percent;
    i32 bestCount = 0;
    for(i32 i = 0; i < BIRD_COUNT && bestCount < maxBest; ++i) {
        if(app.birdFitness[i] > median) {
            memmove(app.birdNNCopy[bestCount++], app.birdNN[i], neuralNetSize);
        }
    }

    LOG("best performing birds copied over: %d", bestCount);
    return bestCount;*/

    /*f32 lowestFit = 9999999;
    f32 highestFit = -9999999;
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        lowestFit = min(lowestFit, app.birdFitness[i]);
        highestFit = max(highestFit, app.birdFitness[i]);
    }

    f32 median = 0.5 * (lowestFit + highestFit);*/

    const i32 maxBest = BIRD_COUNT * percent;
    i32 bestCount = 0;
    u8 copied[BIRD_COUNT];
    memset(copied, 0, sizeof(copied));

    while(bestCount < maxBest) {
        i32 highestId = -1;
        f32 highestFit = -9999999;

        for(i32 i = 0; i < BIRD_COUNT; ++i) {
            if(!copied[i] && app.birdFitness[i] > highestFit) {
                highestId = i;
                highestFit = app.birdFitness[i];
            }
        }

        if(highestFit <= 0) {
            break;
        }

        app.newGenColor[bestCount] = app.birdColor[highestId];
        app.bestFitness[bestCount] = highestFit;
        app.bestTotalFitness += highestFit;
        memmove(app.newGenNN[bestCount], app.birdNN[highestId], app.nnDef.neuralNetSize);
        bestCount++;
        copied[highestId] = TRUE;
    }

    LOG("best birds copied: %d", bestCount);

    for(i32 i = bestCount; i < BIRD_COUNT; ++i) {
        i32 parentA = rand() % bestCount;
        i32 parentB = rand() % bestCount;
        crossover(i, parentA, parentB);
    }
}

void crossoverFavorBest(f32 percent, f32 startDuplicatePerc)
{
    const i32 maxBest = BIRD_COUNT * percent;
    i32 bestCount = 0;
    u8 copied[BIRD_COUNT];
    memset(copied, 0, sizeof(copied));

    i32 duplicateCount = BIRD_COUNT * startDuplicatePerc;

    while(bestCount < maxBest) {
        i32 highestId = -1;
        f32 highestFit = -9999999;

        for(i32 i = 0; i < BIRD_COUNT; ++i) {
            if(!copied[i] && app.birdFitness[i] > highestFit) {
                highestId = i;
                highestFit = app.birdFitness[i];
            }
        }

        if(highestFit <= 0) {
            break;
        }

        for(i32 d = 0; d < duplicateCount; ++d) {
            app.newGenColor[bestCount] = app.birdColor[highestId];
            app.bestFitness[bestCount] = highestFit;
            app.bestTotalFitness += highestFit;
            memmove(app.newGenNN[bestCount], app.birdNN[highestId], app.nnDef.neuralNetSize);
            bestCount++;
        }

        duplicateCount = max(1, duplicateCount / 2);
        copied[highestId] = TRUE;
    }

    LOG("best birds copied: %d", bestCount);

    duplicateCount = (BIRD_COUNT-bestCount) * startDuplicatePerc;
    i32 bestParentId = 0;
    i32 birdChildIdOff = bestCount;

    for(i32 d = 0; d < duplicateCount; ++d) {
        for(i32 i = birdChildIdOff; i < BIRD_COUNT; ++i) {
            i32 parentA = bestParentId;
            i32 parentB = bestParentId+1;

            crossover(i, parentA, parentB);
        }

        birdChildIdOff += duplicateCount;
        bestParentId++;

        if(duplicateCount <= 1) {
            duplicateCount = 0;
        }
        else {
            duplicateCount /= 2;
        }
    }

    LOG("children based off top best: %d", birdChildIdOff - bestCount);

    for(i32 i = birdChildIdOff; i < BIRD_COUNT; ++i) {
        i32 parentA = rand() % bestCount;
        i32 parentB = rand() % bestCount;
        crossover(i, parentA, parentB);
    }

    LOG("children based off random best: %d", BIRD_COUNT - birdChildIdOff);
}

void nextGeneration()
{
#if CROSSOVER_STRATEGY == CROSS_STRAT_RANDOM
    crossoverRandom(0.3f);
#elif CROSSOVER_STRATEGY == CROSS_STRAT_FAVOR_BEST
    crossoverFavorBest(0.4f, 0.2f);
#endif

    i32 mutationCount = 0;
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        //f64 mutationFactor = (f64)i/BIRD_COUNT;
        //mutationFactor = mutationFactor * mutationFactor * mutationFactor;

        i32 curLayerNeuronIdOff = 0;
        for(i32 l = 1; l < app.nnDef.layerCount; ++l) {
            const i32 prevLayerNeuronCount = app.nnDef.layerNeuronCount[l-1];
            curLayerNeuronIdOff += prevLayerNeuronCount;

            for(i32 n = 0; n < app.nnDef.layerNeuronCount[l]; ++n) {
                for(i32 s = 0; s < prevLayerNeuronCount; ++s) {
                    // mutate
                    if(rand()/(f32)RAND_MAX <= 0.005f) {
                        mutationCount++;
                        app.newGenNN[i]->neurons[curLayerNeuronIdOff + n].synapseWeight[s] =
                                randf64(-1.0, 1.0);
                    }
                }
            }
        }
    }

    LOG("mutationCount=%d", mutationCount);

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        memmove(app.birdNN[i], app.newGenNN[i], app.nnDef.neuralNetSize);
    }
}

void doUI()
{
    imguiTestWindow();

    imguiBegin("Timescale");

    imguiSliderFloat("scale", &app.timeScale, 0.1f, 20.f);

    if(imguiButton("1.0")) app.timeScale = 1.0f;
    imguiSameLine();
    if(imguiButton("2.0")) app.timeScale = 2.0f;
    imguiSameLine();
    if(imguiButton("10.0")) app.timeScale = 10.0f;
    imguiSameLine();
    if(imguiButton("10.0")) app.timeScale = 20.0f;

    imguiEnd();
}

void update(f64 delta)
{
    updateCamera(delta);

    // check if dead
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(app.birdHealth[i] <= 0.0f) continue;

        app.birdHealth[i] -= delta;
        if(app.birdHealth[i] <= 0.0f) { // death
            const Color3 black = {0, 0, 0};
            app.birdColor[i] = black;
            const Color4 c1 = {0, 0, 0, 0};
            const Color4 c2 = {0, 0, 0, 0};
            app.targetLine[i].c1 = c1;
            app.targetLine[i].c2 = c2;

            app.birdFitness[i] = app.birdAppleEatenCount[i] * 2 +
                                 APPLE_RADIUS / app.birdShortestDistToNextApple[i];
            app.genTotalFitness += app.birdFitness[i];
        }
    }

    if(app.mode == MODE_NN_TRAIN) {
        updateBirdInputNN(delta);
    }

    updateBirdPhysics(delta);

    if(app.mode == MODE_HUMAN_PLAY) {
        app.birdInput[0].left = 0;
        app.birdInput[0].right = 0;
    }

    i32 aliveCount = 0;
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(app.birdHealth[i] > 0.f) {
            aliveCount++;
        }
    }

    // produce next generation
    if(aliveCount == 0) {
        LOG("#%d genTotalFitness=%.5f avg=%.5f", app.genNumber++,
            app.genTotalFitness, app.genTotalFitness/BIRD_COUNT);
        //LOG("out1[%.3f, %.3f] out1[%.3f, %.3f]", outMin1, outMax1, outMin2, outMax2);
        nextGeneration();
        resetBirds();
    }
}

void render(f64 delta)
{
    imguiUpdate(app.ims, delta);
    doUI();

    glClear(GL_COLOR_BUFFER_BIT);

    drawLineBatch(app.targetLine, BIRD_COUNT);

    drawSpriteBatch(app.tex_birdWing, app.birdLeftWingTf, app.birdColor, BIRD_COUNT);
    drawSpriteBatch(app.tex_birdWing, app.birdRightWingTf, app.birdColor, BIRD_COUNT);
    drawSpriteBatch(app.tex_birdBody, app.birdBodyTf, app.birdColor, BIRD_COUNT);

    drawSpriteBatch(app.tex_apple, app.appleTf, app.birdColor, BIRD_COUNT);

    /*Transform pr;
    pr.pos.x = app.birdPos[0].x + cosf(app.birdRot[0] -PI * 0.5f +PI * 0.15f) * 40.f;
    pr.pos.y = app.birdPos[0].y + sinf(app.birdRot[0] -PI * 0.5f +PI * 0.15f) * 40.f;
    pr.size.x = 10;
    pr.size.y = 10;
    pr.center.x = 5;
    pr.center.y = 5;

    Color3 red = {255, 0, 0};

    drawSpriteBatch(app.tex_apple, &pr, &red, 1);*/

    imguiRender();
}

void cleanup()
{
    free(app.ims);
    free(app.birdNNData);
    free(app.newGenNNData);
}

i32 main()
{
    LOG("Burds");

    srand(time(NULL));
    timeInit();

    i32 sdl = SDL_Init(SDL_INIT_VIDEO);
    if(sdl != 0) {
        LOG("ERROR: could not init SDL2 (%s)", SDL_GetError());
        return 1;
    }

    if(!init()) {
        return 1;
    }

    timept t0 = timeGet();
    f64 accumulator = 0.0;

    while(app.running) {
        SDL_Event event;
        while(SDL_PollEvent(&event)) {
            handleEvent(&event);
        }

        timept t1 = timeGet();

        f64 elapsed = timeToSeconds(t1 - t0);
        t0 = t1;

        accumulator += elapsed;

        while(accumulator > UPDATE_DT) {
            update(UPDATE_DT * app.timeScale);
            accumulator -= UPDATE_DT;
        }

        render(elapsed);

        SDL_GL_SwapWindow(app.window);
    }

    cleanup();

    SDL_Quit();
    return 0;
}
