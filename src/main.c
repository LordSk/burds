#include "base.h"
#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <gl3w.h>
#include <stdlib.h>

#include "sprite.h"

#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720

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
        app.birdPos[i].x = rand() % WINDOW_WIDTH;
        app.birdPos[i].y = rand() % WINDOW_HEIGHT;
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

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdRot[i] = (rand() % 10000) / 10000.f * TAU;
        app.birdAngularVel[i] = -TAU + (rand() % 10000) / 10000.f * TAU * 2.f;
    }

    for(i32 i = 0; i < APPLE_POS_LIST_COUNT; ++i) {
        app.applePosList[i].x = rand() % WINDOW_WIDTH;
        app.applePosList[i].y = rand() % WINDOW_HEIGHT;
    }

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.appleTf[i].size.x = 20;
        app.appleTf[i].size.y = 20;
        app.appleTf[i].center.x = 10;
        app.appleTf[i].center.y = 10;
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

    glClearColor(179.f/255.f, 204.f/255.f, 255.f/255.f, 1.0f);

    app.tex_birdBody = loadTexture("../bird_body.png");
    app.tex_birdWing = loadTexture("../wing.png");
    app.tex_apple = loadTexture("../apple.png");
    if(app.tex_birdBody == -1 || app.tex_birdWing == -1 ||
       app.tex_apple == -1) {
        return FALSE;
    }

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
}

void updateBirdInput(f64 delta)
{
    // random input
    static f64 acc = 0.0f;
    acc += delta;
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

void updateBirdCore(f64 delta)
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
        if(app.birdPos[i].y > WINDOW_HEIGHT) {
            app.birdPos[i].y = WINDOW_HEIGHT;
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

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        Vec2 applePos = app.applePosList[app.birdApplePositionId[i]];
        if(vec2Distance(&applePos, &app.birdPos[i]) < 20.f) {
            app.birdApplePositionId[i]++;
            app.birdHealth[i] += 5.f; // seconds
        }
    }

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.appleTf[i].pos = app.applePosList[app.birdApplePositionId[i]];
    }
}

void update(f64 delta)
{
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdHealth[i] -= delta;
        if(app.birdHealth[i] <= 0.0f) {
            const Color3 black = {0, 0, 0};
            app.birdColor[i] = black;
        }
    }

    updateBirdInput(delta);
    updateBirdCore(delta);

    glClear(GL_COLOR_BUFFER_BIT);

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

    SDL_Quit();
    return 0;
}
