#include "base.h"
#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <gl3w.h>

#include "sprite.h"

#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720

struct App
{
    i32 running;
    SDL_Window* window;
    SDL_GLContext glContext;

    i32 tex_birdBody;
    i32 tex_birdWing; // ratio : 268/160
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

    // OpenGL base state
    glClearColor(0.15f, 0.15f, 0.15f, 1.0f);

    if(!initSpriteState(WINDOW_WIDTH, WINDOW_HEIGHT)) {
        return FALSE;
    }

    app.tex_birdBody = loadTexture("../bird_body.png");
    app.tex_birdWing = loadTexture("../wing.png");
    if(app.tex_birdBody == -1 || app.tex_birdWing == -1) {
        return FALSE;
    }

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

void update(f32 delta)
{
    glClear(GL_COLOR_BUFFER_BIT);

    Transform tr;
    tr.pos.x = 300;
    tr.pos.y = 100;
    tr.scale.x = -268;
    tr.scale.y = 160;
    tr.rot = 0;

    Color3 color;
    color.r = 255;
    color.g = 0;
    color.b = 0;

    drawSpriteBatch(app.tex_birdWing, &tr, &color, 1);
}

i32 main()
{
    LOG("Burds");

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

        update((t1 - t0) / 1000.f);
        SDL_GL_SwapWindow(app.window);
    }

    SDL_Quit();
    return 0;
}
