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

struct App
{
    i32 running;
    SDL_Window* window;
    SDL_GLContext glContext;

    i32 tex_birdBody;
    i32 tex_birdWing; // ratio : 268/160

    Transform birdBodyTf[BIRD_COUNT];
    Transform birdLeftWingTf[BIRD_COUNT];
    Transform birdRightWingTf[BIRD_COUNT];
    Color3 birdColor[BIRD_COUNT];

    Vec2 birdPos[BIRD_COUNT];
    Vec2 birdVel[BIRD_COUNT];
    f32 birdRot[BIRD_COUNT];
    f32 birdFlapAnimTime[BIRD_COUNT];
} app;

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

    glClearColor(209.f/255.f, 234.f/255.f, 255.f/255.f, 1.0f);

    app.tex_birdBody = loadTexture("../bird_body.png");
    app.tex_birdWing = loadTexture("../wing.png");
    if(app.tex_birdBody == -1 || app.tex_birdWing == -1) {
        return FALSE;
    }

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
        app.birdBodyTf[i].scale.x = BIRD_BODY_WIDTH;
        app.birdBodyTf[i].scale.y = BIRD_BODY_HEIGHT;
    }

    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdLeftWingTf[i].scale.x = -BIRD_WING_WIDTH;
        app.birdLeftWingTf[i].scale.y = BIRD_WING_HEIGHT;
    }
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdRightWingTf[i].scale.x = BIRD_WING_WIDTH;
        app.birdRightWingTf[i].scale.y = BIRD_WING_HEIGHT;
    }

    memset(app.birdVel, 0, sizeof(app.birdVel));
    memset(app.birdRot, 0, sizeof(app.birdRot));

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
    }
}

void update(f64 delta)
{
    const f32 gravity = 200.f;

    // apply gravity to bird ve
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdVel[i].y += gravity * delta;
    }

    // apply bird velocity to pos
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdPos[i].x += app.birdVel[i].x * delta;
        app.birdPos[i].y += app.birdVel[i].y * delta;
    }

    // check for ground collision
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        if(app.birdPos[i].y > WINDOW_HEIGHT) {
            app.birdPos[i].y = WINDOW_HEIGHT;
            app.birdVel[i].x = 0;
            app.birdVel[i].y = 0;
        }
    }

    // update bird body transform
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdBodyTf[i].pos.x = app.birdPos[i].x - BIRD_BODY_WIDTH * 0.5f;
        app.birdBodyTf[i].pos.y = app.birdPos[i].y - BIRD_BODY_HEIGHT * 0.5f;
    }

    // update bird wing transform
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdLeftWingTf[i].pos.x = app.birdPos[i].x;
        app.birdLeftWingTf[i].pos.y = app.birdPos[i].y - BIRD_BODY_HEIGHT * 0.4f;
    }
    for(i32 i = 0; i < BIRD_COUNT; ++i) {
        app.birdRightWingTf[i].pos.x = app.birdPos[i].x;
        app.birdRightWingTf[i].pos.y = app.birdPos[i].y - BIRD_BODY_HEIGHT * 0.4f;
    }

    glClear(GL_COLOR_BUFFER_BIT);

    drawSpriteBatch(app.tex_birdWing, app.birdLeftWingTf, app.birdColor, BIRD_COUNT);
    drawSpriteBatch(app.tex_birdWing, app.birdRightWingTf, app.birdColor, BIRD_COUNT);
    drawSpriteBatch(app.tex_birdBody, app.birdBodyTf, app.birdColor, BIRD_COUNT);
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
