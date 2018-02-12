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
#define FRAMES_PER_SEC 60.0
#define FRAME_DT ((f64)(1.0/FRAMES_PER_SEC))

#define BIRD_COUNT 512

#define BIRD_BODY_RATIO 0.5f
#define BIRD_WING_RATIO 1.68125f

#define BIRD_BODY_HEIGHT 50
#define BIRD_BODY_WIDTH (BIRD_BODY_HEIGHT * BIRD_BODY_RATIO)
#define BIRD_WING_WIDTH 50
#define BIRD_WING_HEIGHT (1.f / BIRD_WING_RATIO * BIRD_WING_WIDTH)

#define WING_FLAP_TIME 0.4f

#define FRICTION_AIR_ANGULAR 0.05f
#define FRICTION_AIR 0.05f

#define WING_STRENGTH 160.f
#define WING_STRENGTH_ANGULAR (PI * 0.4f)

#define ANGULAR_VELOCITY_MAX (TAU * 2.0)

#define APPLE_POS_LIST_COUNT 1024
#define APPLE_RADIUS 80.0
#define APPLE_HEALTH_BONUS 5.0

#define GROUND_Y 1000

#define REINSERT_TRUNCATE 0x1
#define REINSERT_DUPLICATE 0x2
#define REINSERT_STRATEGY REINSERT_TRUNCATE

#define INPUT_STACK_SIZE 20

#define NEURAL_NET_LAYERS { 6, INPUT_STACK_SIZE*3, INPUT_STACK_SIZE*2 }

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

typedef struct Bounds {
    f64 worst, best;
} Bounds;

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
    f32 birdFlapLeftCd[BIRD_COUNT];
    f32 birdFlapRightCd[BIRD_COUNT];
    BirdInput birdInput[BIRD_COUNT];

    f32 birdHealth[BIRD_COUNT];
    i32 birdApplePositionId[BIRD_COUNT];

    Vec2 applePosList[APPLE_POS_LIST_COUNT];
    Transform appleTf[BIRD_COUNT];

    Line targetLine[BIRD_COUNT];

    NeuralNetDef nnDef;
    NeuralNet* curGenNN[BIRD_COUNT];
    NeuralNet* newGenNN[BIRD_COUNT];
    u8* birdNNData;
    u8* newGenNNData;

    i32 birdAppleEatenCount[BIRD_COUNT];
    f32 birdShortestDistToNextApple[BIRD_COUNT];
    f32 birdLongestDistToNextApple[BIRD_COUNT];
    f32 birdMaxHealthAchieved[BIRD_COUNT];
    f64 birdDistToNextAppleSum[BIRD_COUNT];
    i32 birdDistCheckCount[BIRD_COUNT];

    f64 birdFitness[BIRD_COUNT];
    f64 birdLivingFitness[BIRD_COUNT];
    BirdInput birdInputStack[BIRD_COUNT][INPUT_STACK_SIZE];
    f64 genTotalFitness;

    Bounds genFitness;
    Bounds genAvgDistFactor;
    Bounds genShortDistFactor;
    Bounds genLongDistFactor;
    Bounds genHealthFactor;

    f32 viewZoom;
    i32 viewX;
    i32 viewY;
    u8 mouseRightButDown;

    i32 mode;
    i32 genNumber;

    struct ImGuiGLSetup* ims;

    i32 timeScale;
    i32 inputStackTop;
} app;

typedef i64 timept;
timept startCounter;
i64 PERFORMANCE_FREQUENCY;

void timeInit()
{
    LARGE_INTEGER li;
    QueryPerformanceFrequency(&li);
    PERFORMANCE_FREQUENCY = (i64)li.QuadPart;
    QueryPerformanceCounter(&li);
    startCounter = (i64)li.QuadPart;
    LOG("performanceFrequency=%lld", PERFORMANCE_FREQUENCY);
}

inline timept timeGet()
{
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return ((timept)li.QuadPart - startCounter);
}

inline i64 timeToMicrosec(i64 delta)
{
    return (delta * 1000000) / PERFORMANCE_FREQUENCY;
}

void resetBirdColors()
{
    const u32 colorMax = 0xFF;
    const u32 colorMin = 0x0;
    const u32 colorDelta = colorMax - colorMin;

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        Color3 c;
        c.r = (rand() % colorDelta) + colorMin;
        c.g = (rand() % colorDelta) + colorMin;
        c.b = (rand() % colorDelta) + colorMin;
        app.birdColor[i] = c;
    }
}

void resetBirds()
{
    resetBirdColors();

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
        app.birdHealth[i] = APPLE_HEALTH_BONUS; // seconds
    }

    memset(app.birdVel, 0, sizeof(app.birdVel));
    memset(app.birdRot, 0, sizeof(app.birdRot));
    memset(app.birdAngularVel, 0, sizeof(app.birdAngularVel));
    memset(app.birdFlapLeftCd, 0, sizeof(app.birdFlapLeftCd));
    memset(app.birdFlapRightCd, 0, sizeof(app.birdFlapRightCd));
    memset(app.birdApplePositionId, 0, sizeof(app.birdApplePositionId));
    memset(app.birdAppleEatenCount, 0, sizeof(app.birdAppleEatenCount));
    memset(app.birdMaxHealthAchieved, 0, sizeof(app.birdMaxHealthAchieved));
    memset(app.birdDistToNextAppleSum, 0, sizeof(app.birdDistToNextAppleSum));
    memset(app.birdDistCheckCount, 0, sizeof(app.birdDistCheckCount));

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        //app.birdRot[i] = PI;
        app.birdShortestDistToNextApple[i] = 99999.f;
        app.birdLongestDistToNextApple[i] = 0.f;
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

    Bounds minb = {DBL_MAX, -DBL_MAX};
    app.genFitness = minb;
    app.genAvgDistFactor = minb;
    app.genShortDistFactor = minb;
    app.genLongDistFactor = minb;
    app.genHealthFactor = minb;

    app.inputStackTop = 0;
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

void resetTraining()
{
    resetBirds();
    resetApplePath();

    neuralNetInitRandom(app.curGenNN, BIRD_COUNT, &app.nnDef);
    app.genNumber = 0;
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
    makeNeuralNetDef(&app.nnDef, sizeof(layers) / sizeof(layers[0]), layers, 1.f);

    app.birdNNData = neuralNetAlloc(app.curGenNN, BIRD_COUNT, &app.nnDef);
    app.newGenNNData = neuralNetAlloc(app.newGenNN, BIRD_COUNT, &app.nnDef);

    app.mode = MODE_NN_TRAIN;

    resetTraining();

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
           resetTraining();
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

void updateBirdInputNN()
{
    i32 inputStackCur = app.inputStackTop--;
    if(app.inputStackTop < 0) {
        app.inputStackTop = INPUT_STACK_SIZE-1;
        inputStackCur = app.inputStackTop;

        // setup neural net inputs
        for(i32 i = 0; i < BIRD_COUNT; ++i) {
            Vec2 applePos = app.applePosList[app.birdApplePositionId[i]];
            f64 appleOffsetX = applePos.x - app.birdPos[i].x;
            f64 appleOffsetY = applePos.y - app.birdPos[i].y;
            f64 velX = app.birdVel[i].x;
            f64 velY = app.birdVel[i].y;
            f64 rot = app.birdRot[i];
            f64 angVel = app.birdAngularVel[i];

            app.curGenNN[i]->values[0] = appleOffsetX / 10000.0;
            app.curGenNN[i]->values[1] = appleOffsetY / 10000.0;
            app.curGenNN[i]->values[2] = velX / 1000.0;
            app.curGenNN[i]->values[3] = velY / 1000.0;
            app.curGenNN[i]->values[4] = (rot -PI * 0.25f) / TAU;
            // 5 rotations per second is propably high enough
            app.curGenNN[i]->values[5] = angVel / (TAU * 5.0);
            //app.birdNN[i]->values[6] = app.birdFlapLeftCd[i] / WING_FLAP_TIME;
            //app.birdNN[i]->values[7] = app.birdFlapRightCd[i] / WING_FLAP_TIME;

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
        }

        neuralNetPropagate(app.curGenNN, BIRD_COUNT, &app.nnDef);

        // get neural net output
        for(i32 i = 0; i < BIRD_COUNT; ++i) {
            f64* output = &app.curGenNN[i]->values[app.nnDef.neuronCount-(INPUT_STACK_SIZE*2)];
            for(i32 s = 0; s < INPUT_STACK_SIZE; ++s) {
                app.birdInputStack[i][s].left = output[s*2] > 0.8;
                app.birdInputStack[i][s].right = output[s*2+1] > 0.8;
            }
        }
    }


    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(app.birdHealth[i] <= 0.0f) {
            app.birdInput[i].left = 0;
            app.birdInput[i].right = 0;
        }
        else {
            app.birdInput[i] = app.birdInputStack[i][inputStackCur];
        }
    }
}

void updateBirdPhysics()
{
    // apply bird input
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        u8 flapLeft = (app.birdFlapLeftCd[i] <= 0.0f) && app.birdInput[i].left;
        u8 flapRight = (app.birdFlapRightCd[i] <= 0.0f) && app.birdInput[i].right;

        Vec2 leftForce = {
            flapLeft * cosf(app.birdRot[i] -PI * 0.5f +PI * 0.15f) * WING_STRENGTH,
            flapLeft * sinf(app.birdRot[i] -PI * 0.5f +PI * 0.15f) * WING_STRENGTH,
        };
        Vec2 rightForce = {
            flapRight * cosf(app.birdRot[i] -PI * 0.5f -PI * 0.15f) * WING_STRENGTH,
            flapRight * sinf(app.birdRot[i] -PI * 0.5f -PI * 0.15f) * WING_STRENGTH,
        };

        Vec2 totalForce = { leftForce.x + rightForce.x, leftForce.y + rightForce.y };

        if(flapLeft && flapRight) {
            totalForce.x *= 1.5f;
            totalForce.y *= 1.5f;
            app.birdAngularVel[i] *= 0.5;
        }

        // easier to rectify speed
        f32 vl = vec2Len(&app.birdVel[i]);
        f32 fl = vec2Len(&totalForce);

        if(vec2Dot(&app.birdVel[i], &totalForce) < -(vl * fl * 0.5)) {
            totalForce.x += -app.birdVel[i].x * 0.75;
            totalForce.y += -app.birdVel[i].y * 0.75;
        }

        app.birdVel[i].x += totalForce.x;
        app.birdVel[i].y += totalForce.y;

        // easier to rectify angle
        f32 leftAngStr = WING_STRENGTH_ANGULAR;
        f32 rightAngStr = -WING_STRENGTH_ANGULAR;

        /*if(app.birdAngularVel[i] < 0.0) {
            leftAngStr -= app.birdAngularVel[i];
            rightAngStr += app.birdAngularVel[i] * 0.5;
        }

        if(app.birdAngularVel[i] > 0.0) {
            rightAngStr -= app.birdAngularVel[i];
            leftAngStr += app.birdAngularVel[i] * 0.5;
        }*/

        app.birdAngularVel[i] += flapLeft * leftAngStr +
                                 flapRight * rightAngStr;

        if(flapLeft) {
            app.birdFlapLeftCd[i] = WING_FLAP_TIME;
        }
        if(flapRight) {
            app.birdFlapRightCd[i] = WING_FLAP_TIME;
        }
    }

    // wing anim time
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(app.birdHealth[i] <= 0.0f) continue;
        app.birdFlapLeftCd[i] -= FRAME_DT;
        app.birdFlapRightCd[i] -= FRAME_DT;
        if(app.birdFlapLeftCd[i] <= 0.0f) {
            app.birdFlapLeftCd[i] = 0.0f;
        }
        if(app.birdFlapRightCd[i] <= 0.0f) {
            app.birdFlapRightCd[i] = 0.0f;
        }
    }

    const f32 gravity = 200.f;

    // apply gravity and friction to bird velocity
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdVel[i].y += gravity * FRAME_DT;
        app.birdVel[i].y *= 1.f - FRICTION_AIR * FRAME_DT;
        app.birdVel[i].x *= 1.f - FRICTION_AIR * FRAME_DT;
    }
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(app.birdAngularVel[i] > ANGULAR_VELOCITY_MAX) app.birdAngularVel[i] = ANGULAR_VELOCITY_MAX;
        if(app.birdAngularVel[i] < -ANGULAR_VELOCITY_MAX) app.birdAngularVel[i] = -ANGULAR_VELOCITY_MAX;
        app.birdAngularVel[i] *= 1.f - FRICTION_AIR_ANGULAR * FRAME_DT;
    }

    // apply bird velocity to pos
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdPos[i].x += app.birdVel[i].x * FRAME_DT;
        app.birdPos[i].y += app.birdVel[i].y * FRAME_DT;
    }
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdRot[i] += app.birdAngularVel[i] * FRAME_DT;
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
    const f32 wingDownAngle = PI * 0.6f;

    // update bird wing transform
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdLeftWingTf[i].pos.x = app.birdPos[i].x;
        app.birdLeftWingTf[i].pos.y = app.birdPos[i].y;
        app.birdLeftWingTf[i].rot = app.birdRot[i] -
            (wingUpAngle + wingDownAngle) * (app.birdFlapLeftCd[i] / WING_FLAP_TIME);
    }
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdRightWingTf[i].pos.x = app.birdPos[i].x;
        app.birdRightWingTf[i].pos.y = app.birdPos[i].y;
        app.birdRightWingTf[i].rot = app.birdRot[i] +
            (wingUpAngle + wingDownAngle) * (app.birdFlapRightCd[i] / WING_FLAP_TIME);
    }

    // check if we touched the apple
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        Vec2* applePos = &app.applePosList[app.birdApplePositionId[i]];
        if(app.birdHealth[i] > 0.0f && vec2Distance(applePos, &app.birdPos[i]) < APPLE_RADIUS) {
            app.birdApplePositionId[i]++;
            app.birdAppleEatenCount[i]++;
            app.birdHealth[i] += APPLE_HEALTH_BONUS;
            app.birdShortestDistToNextApple[i] = vec2Distance(&app.applePosList[app.birdApplePositionId[i]],
                    &app.birdPos[i]);
            app.birdLongestDistToNextApple[i] = app.birdShortestDistToNextApple[i];
            app.birdDistToNextAppleSum[i] = 0;
            app.birdDistCheckCount[i] = 0;
        }
    }

    // update shortest/longest distance to next apple
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        f32 dist = vec2Distance(&app.applePosList[app.birdApplePositionId[i]], &app.birdPos[i]);
        app.birdShortestDistToNextApple[i] = min(app.birdShortestDistToNextApple[i], dist);
        app.birdLongestDistToNextApple[i] = max(app.birdShortestDistToNextApple[i], dist);
        app.birdDistToNextAppleSum[i] += dist;
        app.birdDistCheckCount[i]++;
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

void updateCamera()
{
    if(app.mouseRightButDown) {
        SDL_SetRelativeMouseMode(TRUE);
    }
    else {
        SDL_SetRelativeMouseMode(FALSE);
    }
    setView(app.viewX, app.viewY, WINDOW_WIDTH*app.viewZoom, WINDOW_HEIGHT*app.viewZoom);
}

void crossover(i32 id, i32 parentA, i32 parentB)
{
    for(i32 s = 0; s < app.nnDef.synapseTotalCount; ++s) {
        // get weight from parent A
        if(rand() & 1) {
            app.newGenNN[id]->weights[s] = app.curGenNN[parentA]->weights[s];
        }
        // get weight from parent B
        else {
            app.newGenNN[id]->weights[s] = app.curGenNN[parentB]->weights[s];
        }
    }
}

i32 selectRandom(const i32 reinsertCount, i32 notThisId)
{
    i32 r = rand() % reinsertCount;
    while(r == notThisId) {
        r = rand() % reinsertCount;
    }
    return r;
}

i32 selectTournament(const i32 reinsertCount, const i32 tournamentSize, i32 notThisId)
{
    i32 champion = selectRandom(reinsertCount, notThisId);
    f32 championFitness = app.birdFitness[champion];

    for(i32 i = 0; i < tournamentSize; ++i) {
        i32 opponent = selectRandom(reinsertCount, notThisId);
        if(app.birdFitness[opponent] > championFitness) {
            champion = opponent;
            championFitness = app.birdFitness[opponent];
        }
    }

    return champion;
}

typedef struct {
    i32 id;
    f64 fitness;
} FitnessPair;

i32 compareFitness(const void* a, const void* b)
{
    const FitnessPair* fa = (FitnessPair*)a;
    const FitnessPair* fb = (FitnessPair*)b;
    if(fa->fitness > fb->fitness) return -1;
    if(fa->fitness < fb->fitness) return 1;
    return 0;
}

i32 reinsertTruncateNN(i32 maxBest, i32 nnCount, NeuralNet** nextGen, NeuralNet** curGen)
{
    FitnessPair list[BIRD_COUNT];
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        list[i].id = i;
        list[i].fitness = app.birdFitness[i];
    }

    qsort(list, BIRD_COUNT, sizeof(FitnessPair), compareFitness);

    for(i32 i = 0; i < maxBest; ++i) {
        memmove(app.newGenNN[i], app.curGenNN[list[i].id], app.nnDef.neuralNetSize);
    }

    return maxBest;
}

// Reinsert best percentile, duplicate best ones (does not exceed percent count)
i32 reinsertDuplicate(f32 percent, i32 startingDuplicateCount)
{
    const i32 maxBest = BIRD_COUNT * percent;
    i32 reinsertCount = 0;
    u8 copied[BIRD_COUNT];
    memset(copied, 0, sizeof(copied));

    i32 duplicateCount = startingDuplicateCount;

    while(reinsertCount < maxBest) {
        i32 highestId = -1;
        f32 highestFit = -9999999;

        for(i32 i = 0; i < BIRD_COUNT; ++i) {
            if(!copied[i] && app.birdFitness[i] > highestFit) {
                highestId = i;
                highestFit = app.birdFitness[i];
            }
        }

        for(i32 d = 0; d < duplicateCount; ++d) {
            memmove(app.newGenNN[reinsertCount], app.curGenNN[highestId], app.nnDef.neuralNetSize);
            reinsertCount++;
        }

        duplicateCount = max(1, duplicateCount / 2);
        copied[highestId] = TRUE;
    }

    return reinsertCount;
}

void nextGeneration()
{
    LOG("#%d totalFitness=%.5f avg=%.5f", app.genNumber++,
        app.genTotalFitness, app.genTotalFitness/BIRD_COUNT);
    //LOG("out1[%.3f, %.3f] out1[%.3f, %.3f]", outMin1, outMax1, outMin2, outMax2);
    LOG("fitness=[%.5f, %.5f] shortDist=[%.5f, %.5f] avgDist=[%.5f, %.5f] longDist=[%.5f, %.5f]"
        " health=[%.5f, %.5f]",
        app.genFitness.worst, app.genFitness.best,
        app.genShortDistFactor.worst, app.genShortDistFactor.best,
        app.genAvgDistFactor.worst, app.genAvgDistFactor.best,
        app.genLongDistFactor.worst, app.genLongDistFactor.best,
        app.genHealthFactor.worst, app.genHealthFactor.best
        );


#if REINSERT_STRATEGY == REINSERT_TRUNCATE
    i32 reinsertCount = reinsertTruncateNN(BIRD_COUNT*0.1);
#elif REINSERT_STRATEGY == REINSERT_DUPLICATE
    i32 reinsertCount = reinsertDuplicate(0.3f, 16);
#endif

    const i32 tournamentSize = 15;
    LOG("tournamentSize=%d", tournamentSize);

    // cross over
    for(i32 i = reinsertCount; i < BIRD_COUNT; ++i) {
        i32 parentA = selectTournament(BIRD_COUNT, tournamentSize, -1);
        i32 parentB = selectTournament(BIRD_COUNT, tournamentSize, parentA);
        crossover(i, parentA, parentB);
    }

    // mutate
    const f32 mutationRate = 0.005f;
    i32 mutationCount = 0;
    for(i32 i = 1; i < BIRD_COUNT; ++i) {
        //f64 mutationFactor = 0.1 + ((f64)i/BIRD_COUNT) * 0.9;
        //f64 mutationFactor = (f64)i/BIRD_COUNT * 0.5;
        f64 mutationFactor = 0.5;

        for(i32 s = 0; s < app.nnDef.synapseTotalCount; ++s) {
            // mutate
            if(randf64(0.0, 1.0) < mutationRate) {
                mutationCount++;
                app.newGenNN[i]->weights[s] += randf64(-mutationFactor, mutationFactor);
            }
        }
    }

    LOG("mutationRate=%g mutationCount=%d", mutationRate, mutationCount);

    memmove(app.curGenNN[0], app.newGenNN[0], app.nnDef.neuralNetSize * BIRD_COUNT);
}

void doUI()
{
    //imguiTestWindow();

    imguiBegin("Timescale");

    imguiSliderInt("scale", &app.timeScale, 1, 10);

    imguiEnd();
}

void newFrame()
{
    // update maximum health achieved
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdMaxHealthAchieved[i] = max(app.birdMaxHealthAchieved[i], app.birdHealth[i]);
    }

    // check if dead
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(app.birdHealth[i] <= 0.0f) continue;

        app.birdHealth[i] -= FRAME_DT;
        if(app.birdHealth[i] <= 0.0f) { // death
            const Color3 black = {0, 0, 0};
            app.birdColor[i] = black;
            const Color4 c1 = {0, 0, 0, 0};
            const Color4 c2 = {0, 0, 0, 0};
            app.targetLine[i].c1 = c1;
            app.targetLine[i].c2 = c2;

            f64 applesFactor = app.birdAppleEatenCount[i];
            f64 shortDistFactor = APPLE_RADIUS / app.birdShortestDistToNextApple[i]; // 0.0 -> 1.0
            f64 longDistFactor = app.birdLongestDistToNextApple[i] / 5000.0; // 0.0 -> inf
            f64 deathDistFactor = APPLE_RADIUS * 1000.0 / vec2Distance(&app.birdPos[i],
                                  &app.appleTf[i].pos); // 0.0 -> 1.0
            f64 healthFactor = app.birdMaxHealthAchieved[i] / APPLE_HEALTH_BONUS - 1.0f;
            f64 avgDistFactor = (app.birdDistToNextAppleSum[i] / app.birdDistCheckCount[i]) /
                                5000.0;

            /*if(app.birdShortestDistToNextApple[i] > 2500.0) {
                shortDistFactor = (app.birdShortestDistToNextApple[i] - 2500.0) / -2500.0;
            }*/

            /*if(applesFactor == 0) {
                deathDistFactor = 0;
            }*/

            /*app.birdFitness[i] = applesFactor * 4.0 +
                                 shortDistFactor +
                                 deathDistFactor;*/

            app.birdFitness[i] = applesFactor * 2000.0 + deathDistFactor;
            //app.birdFitness[i] = applesFactor * 2.0 + shortDistFactor;

            app.genTotalFitness += app.birdFitness[i];
            app.genFitness.worst = min(app.genFitness.worst, app.birdFitness[i]);
            app.genFitness.best = max(app.genFitness.best, app.birdFitness[i]);
            app.genAvgDistFactor.worst = min(app.genAvgDistFactor.worst, avgDistFactor);
            app.genAvgDistFactor.best = max(app.genAvgDistFactor.best, avgDistFactor);
            app.genShortDistFactor.worst = min(app.genShortDistFactor.worst, shortDistFactor);
            app.genShortDistFactor.best = max(app.genShortDistFactor.best, shortDistFactor);
            app.genLongDistFactor.worst = min(app.genLongDistFactor.worst, longDistFactor);
            app.genLongDistFactor.best = max(app.genLongDistFactor.best, longDistFactor);
            app.genHealthFactor.worst = min(app.genHealthFactor.worst, healthFactor);
            app.genHealthFactor.best = max(app.genHealthFactor.best, healthFactor);
        }
        else {
            f64 applesFactor = app.birdAppleEatenCount[i];
            f64 shortDistFactor = APPLE_RADIUS / app.birdShortestDistToNextApple[i]; // 0.0 -> 1.0
            f64 deathDistFactor = APPLE_RADIUS / vec2Distance(&app.birdPos[i],
                                                              &app.appleTf[i].pos); // 0.0 -> 1.0

            if(applesFactor == 0) {
                deathDistFactor = 0;
            }

            app.birdLivingFitness[i] = applesFactor * 2.0 +
                                         shortDistFactor +
                                         deathDistFactor;
        }
    }

    if(app.mode == MODE_NN_TRAIN) {
        updateBirdInputNN();
    }

    updateBirdPhysics();

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
        nextGeneration();
        resetBirds();
    }
}

void render()
{
    imguiUpdate(app.ims);
    doUI();

    Color3* fitColor = app.birdColor;

#if 0
    Color3 fitColor[BIRD_COUNT];
    f64 lowestFitness = 99999;
    f64 highestFitness = -99999;
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(app.birdHealth[i] <= 0.0) continue;
        lowestFitness = min(lowestFitness, app.birdLivingFitness[i]);
        highestFitness = max(highestFitness, app.birdLivingFitness[i]);
    }

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        f64 f = (app.birdLivingFitness[i] - lowestFitness) / (highestFitness - lowestFitness);
        u8 r = 255 * (1.0 - f);
        u8 g = 255 * f;
        u8 b = 0;
        fitColor[i] = color3(r, g, b);
        if(app.birdHealth[i] <= 0.0) {
            fitColor[i] = color3(0, 0, 0);
        }
    }
#endif

    glClear(GL_COLOR_BUFFER_BIT);

    const f32 gl = -10000000.f;
    const f32 gr = 10000000.f;
    const f32 gt = GROUND_Y;
    const f32 gb = GROUND_Y + 10000.f;
    const Color4 groundColor = {40, 89, 24, 255};

    Quad groundQuad = quadOneColor(gl, gr, gt, gb, groundColor);

    drawQuadBatch(&groundQuad, 1);

    drawLineBatch(app.targetLine, BIRD_COUNT);

    drawSpriteBatch(app.tex_birdWing, app.birdLeftWingTf, fitColor, BIRD_COUNT);
    drawSpriteBatch(app.tex_birdWing, app.birdRightWingTf, fitColor, BIRD_COUNT);
    drawSpriteBatch(app.tex_birdBody, app.birdBodyTf, fitColor, BIRD_COUNT);

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
    _aligned_free(app.birdNNData);
    _aligned_free(app.newGenNNData);
}

i32 main()
{
    LOG("Burds");

    testPropagateNN();

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

    while(app.running) {
        timept t0 = timeGet();
        srand(t0);

        SDL_Event event;
        while(SDL_PollEvent(&event)) {
            handleEvent(&event);
        }

        updateCamera();

        newFrame();
        render();
        SDL_GL_SwapWindow(app.window);

        const i64 frameDtMicro = FRAME_DT/app.timeScale * 1000000;
        while(((frameDtMicro - timeToMicrosec(timeGet() - t0)) / 1000) > 1) {
            _mm_pause();
        }
        //LOG("ft=%lld elapsed=%lld remainingMs=%d", frameDtMicro, elapsedMicro, remainingMs);
    }

    cleanup();

    SDL_Quit();
    return 0;
}
