#include "base.h"
#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <gl3w.h>
#include <stdlib.h>
#include <float.h>

#include "sprite.h"
#include "neural.h"

#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 900

#define BIRD_COUNT 128

#define BIRD_BODY_RATIO 0.5f
#define BIRD_WING_RATIO 1.68125f

#define BIRD_BODY_HEIGHT 50
#define BIRD_BODY_WIDTH (BIRD_BODY_HEIGHT * BIRD_BODY_RATIO)
#define BIRD_WING_WIDTH 50
#define BIRD_WING_HEIGHT (1.f / BIRD_WING_RATIO * BIRD_WING_WIDTH)

#define WING_ANIM_TIME 0.5f

#define FRICTION_AIR_ANGULAR 0.06f
#define FRICTION_AIR 0.02f

#define WING_STRENGTH 200.f
#define WING_STRENGTH_ANGULAR (PI * 0.25f)

#define APPLE_POS_LIST_COUNT 1024

#define GROUND_Y 3000

const NeuralNetDef NEURAL_NET_DEF = { 3, { 5, 10, 2 } };

static f64 outMin1 = 1000.0;
static f64 outMax1 = -1000.0;
static f64 outMin2 = 1000.0;
static f64 outMax2 = -1000.0;

typedef struct
{
    u8 left, right;
} BirdInput;

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

    NeuralNet* birdNN[BIRD_COUNT];
    u8* birdNNData;

    i32 birdAppleEatenCount[BIRD_COUNT];
    f32 birdShortestDistToNextApple[BIRD_COUNT];
    f32 birdFitness[BIRD_COUNT];
    f64 genFitness;

    f32 viewZoom;
    i32 viewX;
    i32 viewY;
    u8 mouseRightButDown;
} app;

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
        app.birdPos[i].y = 0;
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
        app.birdHealth[i] = 10.f; // seconds
    }

    memset(app.birdVel, 0, sizeof(app.birdVel));
    memset(app.birdRot, 0, sizeof(app.birdRot));
    memset(app.birdAngularVel, 0, sizeof(app.birdAngularVel));
    memset(app.birdFlapLeftAnimTime, 0, sizeof(app.birdFlapLeftAnimTime));
    memset(app.birdFlapRightAnimTime, 0, sizeof(app.birdFlapRightAnimTime));
    memset(app.birdApplePositionId, 0, sizeof(app.birdApplePositionId));
    memset(app.birdAppleEatenCount, 0, sizeof(app.birdAppleEatenCount));

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdRot[i] = PI;
        app.birdShortestDistToNextApple[i] = 5000.f;
    }

    i32 spawnOriginX = 0;
    i32 spawnOriginY = -500;
    const i32 spawnRadius = 1000;

    for(i32 i = 0; i < APPLE_POS_LIST_COUNT; ++i) {
        app.applePosList[i].x = spawnOriginX - spawnRadius + (rand() % spawnRadius) * 2;
        app.applePosList[i].y = spawnOriginY - spawnRadius + (rand() % spawnRadius) * 2;
        if(app.applePosList[i].y >= GROUND_Y) {
            app.applePosList[i].y = spawnOriginY - spawnRadius;
        }
        spawnOriginX = app.applePosList[i].x;
        spawnOriginY = app.applePosList[i].y;
    }

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.appleTf[i].size.x = 20;
        app.appleTf[i].size.y = 20;
        app.appleTf[i].center.x = 10;
        app.appleTf[i].center.y = 10;
    }

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        memmove(&app.targetLine[i].c1, &app.birdColor[i], 3);
        app.targetLine[i].c1.a = 128;
        memmove(&app.targetLine[i].c2, &app.birdColor[i], 3);
        app.targetLine[i].c2.a = 0;
    }

    initNeuralNets(app.birdNN, BIRD_COUNT, &NEURAL_NET_DEF);

    outMin1 = 1000.0;
    outMax1 = -1000.0;
    outMin2 = 1000.0;
    outMax2 = -1000.0;

    app.genFitness = 0.0;
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

    app.birdNNData = allocNeuralNets(app.birdNN, BIRD_COUNT, &NEURAL_NET_DEF);

    resetBirds();

    return TRUE;
}

void handleEvent(const SDL_Event* event)
{
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
            app.birdVel[0].x = 0;
            app.birdVel[0].y = 0;
            app.birdRot[0] = 0;
            app.birdAngularVel[0] = 0;
            app.birdPos[0].x = WINDOW_WIDTH * 0.5;
            app.birdPos[0].y = WINDOW_HEIGHT * 0.5;

            resetBirds();
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
            app.viewX -= event->motion.xrel * app.viewZoom;
            app.viewY -= event->motion.yrel * app.viewZoom;
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

void updateBirdInput(f64 delta)
{
    // setup neural net inputs
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        Vec2 applePos = app.applePosList[app.birdApplePositionId[i]];
        f64 appleOffsetX = applePos.x - app.birdPos[i].x;
        f64 appleOffsetY = applePos.y - app.birdPos[i].y;
        f64 velX = app.birdVel[i].x;
        f64 velY = app.birdVel[i].y;
        f64 rot = app.birdRot[i];

#if 0
        app.birdNN[i]->neurons[0].value = appleOffsetX;
        app.birdNN[i]->neurons[1].value = appleOffsetY;
        app.birdNN[i]->neurons[2].value = velX;
        app.birdNN[i]->neurons[3].value = velY;
        app.birdNN[i]->neurons[4].value = rot;
#else // "normalized"
        app.birdNN[i]->neurons[0].value = appleOffsetX / 1000.0;
        app.birdNN[i]->neurons[1].value = appleOffsetY / 1000.0;
        app.birdNN[i]->neurons[2].value = velX / 1000.0;
        app.birdNN[i]->neurons[3].value = velY / 1000.0;
        app.birdNN[i]->neurons[4].value = (rot-PI*0.25f) / TAU;
#endif
    }

    propagateNeuralNets(app.birdNN, BIRD_COUNT, &NEURAL_NET_DEF);

    static f64 acc = 0.0f;
    acc += delta;

    // get neural net output
    // TODO: store this somewhere and use it in neural.c
    i32 neuronCount = 0;
    for(i32 l = 0; l < NEURAL_NET_DEF.layerCount; ++l) {
        neuronCount += NEURAL_NET_DEF.layerNeuronCount[l];
    }

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        f64 out1 = app.birdNN[i]->neurons[neuronCount-2].value;
        f64 out2 = app.birdNN[i]->neurons[neuronCount-1].value;
        outMin1 = min(out1, outMin1);
        outMin2 = min(out2, outMin2);
        outMax1 = max(out1, outMax1);
        outMax2 = max(out2, outMax2);
    }

    if(acc > 0.25) {
        for(i32 i = 0; i < BIRD_COUNT; ++i) {
            app.birdInput[i].left = app.birdNN[i]->neurons[neuronCount-2].value > 0.5;
            app.birdInput[i].right = app.birdNN[i]->neurons[neuronCount-1].value > 0.5;
        }
        /*LOG("out1[min=%.3f, max=%.3f] out2[min=%.3f max=%.3f]",
            outMin1, outMax1, outMin2, outMax2);*/
        acc = 0.0;
    }
    else {
        for(i32 i = 0; i < BIRD_COUNT; ++i) {
            app.birdInput[i].left = 0;
            app.birdInput[i].right = 0;
        }
    }

#if 0
    // random input
    if(acc > 1.0) {
        for(i32 i = 0; i < BIRD_COUNT; ++i) {
            app.birdInput[i].left = rand()%2;
            app.birdInput[i].right = rand()%2;
        }
        acc = 0.0;
    }
    else {
        for(i32 i = 0; i < BIRD_COUNT; ++i) {
            app.birdInput[i].left = 0;
            app.birdInput[i].right = 0;
        }
    }
#endif

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(app.birdHealth[i] <= 0.0f) {
            app.birdInput[i].left = 0;
            app.birdInput[i].right = 0;
        }
    }

    // apply bird input
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        Vec2 leftForce = {
            app.birdInput[i].left * cosf(app.birdRot[i] -PI * 0.5f +PI * 0.15f) * WING_STRENGTH,
            app.birdInput[i].left * sinf(app.birdRot[i] -PI * 0.5f +PI * 0.15f) * WING_STRENGTH,
        };
        Vec2 rightForce = {
            app.birdInput[i].right * cosf(app.birdRot[i] -PI * 0.5f -PI * 0.15f) * WING_STRENGTH,
            app.birdInput[i].right * sinf(app.birdRot[i] -PI * 0.5f -PI * 0.15f) * WING_STRENGTH,
        };
        app.birdVel[i].x += leftForce.x + rightForce.x;
        app.birdVel[i].y += leftForce.y + rightForce.y;
        app.birdAngularVel[i] += app.birdInput[i].left * WING_STRENGTH_ANGULAR +
                                 app.birdInput[i].right * -WING_STRENGTH_ANGULAR;
        if(app.birdInput[i].left) {
            app.birdFlapLeftAnimTime[i] = WING_ANIM_TIME;
        }
        if(app.birdInput[i].right) {
            app.birdFlapRightAnimTime[i] = WING_ANIM_TIME;
        }
    }
}

void updateBirdPhysics(f64 delta)
{
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
            (wingUpAngle + wingDownAngle) * (app.birdFlapLeftAnimTime[i] / WING_ANIM_TIME);
    }
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdRightWingTf[i].pos.x = app.birdPos[i].x;
        app.birdRightWingTf[i].pos.y = app.birdPos[i].y;
        app.birdRightWingTf[i].rot = app.birdRot[i] +
            (wingUpAngle + wingDownAngle) * (app.birdFlapRightAnimTime[i] / WING_ANIM_TIME);
    }

    // check if we touched the apple
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        Vec2* applePos = &app.applePosList[app.birdApplePositionId[i]];
        if(vec2Distance(applePos, &app.birdPos[i]) < 20.f) {
            app.birdApplePositionId[i]++;
            app.birdAppleEatenCount[i]++;
            app.birdHealth[i] += 5.f; // seconds
            app.birdShortestDistToNextApple[i] = vec2Distance(&app.applePosList[app.birdApplePositionId[i]],
                    &app.birdPos[i]);
        }
    }

    // update shortest distance to next apple
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdShortestDistToNextApple[i] = min(app.birdShortestDistToNextApple[i],
            vec2Distance(&app.applePosList[app.birdApplePositionId[i]], &app.birdPos[i]));
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
            const Color4 c1 = {0, 0, 0, 50};
            const Color4 c2 = {0, 0, 0, 0};
            app.targetLine[i].c1 = c1;
            app.targetLine[i].c2 = c2;

            app.birdFitness[i] = app.birdAppleEatenCount[i] * 5000.f +
                                 (5000.f - app.birdShortestDistToNextApple[i]);
            app.genFitness += app.birdFitness[i];
        }
    }

    updateBirdInput(delta);
    updateBirdPhysics(delta);

    i32 aliveCount = 0;
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(app.birdHealth[i] > 0.f) {
            aliveCount++;
        }
    }

    if(aliveCount == 0) {
        LOG("generation total fitness: %.5f", app.genFitness);
        resetBirds();
    }

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
}

void cleanup()
{
    free(app.birdNNData);
}

i32 main()
{
    LOG("Burds");

    srand(time(NULL));

    i32 sdl = SDL_Init(SDL_INIT_VIDEO);
    if(sdl != 0) {
        LOG("ERROR: could not init SDL2 (%s)", SDL_GetError());
        return 1;
    }

    if(!init()) {
        return 1;
    }

    clock_t t0 = TIME_MILLI();

    while(app.running) {
        SDL_Event event;
        while(SDL_PollEvent(&event)) {
            handleEvent(&event);
        }

        clock_t t1 = TIME_MILLI();
        update((t1 - t0) / 1000.0);
        t0 = t1;

        SDL_GL_SwapWindow(app.window);
    }

    cleanup();

    SDL_Quit();
    return 0;
}
