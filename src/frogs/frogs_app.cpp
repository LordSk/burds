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

#define MAP_WIDTH 512
#define MAP_HEIGHT 256
#define MAP_POND_COUNT 2
#define MAP_POND_MIN_RADIUS 25
#define MAP_POND_MAX_RADIUS 80

enum {
    MAP_TILE_GRASS=0,
    MAP_TILE_WATER,
};

struct App {

SDL_Window* window;
SDL_GLContext glContext;
ImGuiGLSetup* ims;
bool running = true;
i32 timeScale = 1;

f32 viewZoom = 10.0f;
i32 viewX = 0;
i32 viewY = 0;
u8 mouseRightButDown;

const i32 mapSize = MAP_WIDTH*MAP_HEIGHT;
u8 mapData[MAP_WIDTH*MAP_HEIGHT];

bool init()
{
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);

    window = SDL_CreateWindow("Frogs",
                              SDL_WINDOWPOS_CENTERED,
                              SDL_WINDOWPOS_CENTERED,
                              WINDOW_WIDTH, WINDOW_HEIGHT,
                              SDL_WINDOW_OPENGL);

    if(!window) {
        LOG("ERROR: can't create SDL2 window (%s)",  SDL_GetError());
        return false;
    }

    glContext = SDL_GL_CreateContext(window);
    if(!glContext) {
        LOG("ERROR: can't create OpenGL 3.3 context (%s)",  SDL_GetError());
        return false;
    }

    SDL_GL_SetSwapInterval(0);

    if(gl3w_init()) {
        LOG("ERROR: can't init gl3w");
        return false;
    }

    if(!gl3w_is_supported(3, 3)) {
        LOG("ERROR: OpenGL 3.3 isn't available on this system");
        return false;
    }

    if(!initSpriteState(WINDOW_WIDTH, WINDOW_HEIGHT)) {
        return false;
    }

    ims = imguiInit(WINDOW_WIDTH, WINDOW_HEIGHT);
    if(!ims) {
        LOG("ERROR: could not init imgui");
    }

    glClearColor(179.f/255.f, 204.f/255.f, 255.f/255.f, 1.0f);

    /*viewX = -2000;
    viewY = -1000;
    viewZoom = 3.f;

    tex_birdBody = loadTexture("../bird_body.png");
    tex_birdWing = loadTexture("../wing.png");
    tex_apple = loadTexture("../apple.png");
    if(tex_birdBody == -1 || tex_birdWing == -1 ||
       tex_apple == -1) {
        return false;
    }*/

    resetMap();

    return true;
}

void run()
{
    while(running) {
        timept t0 = timeGet();

        SDL_Event event;
        while(SDL_PollEvent(&event)) {
            handleEvent(&event);
        }

        updateCamera();

        newFrame();
        render();
        SDL_GL_SwapWindow(window);

        const i64 frameDtMicro = FRAME_DT/timeScale * 1000000;
        while(((frameDtMicro - timeToMicrosec(timeGet() - t0)) / 1000) > 1) {
            _mm_pause();
        }
    }
}

void handleEvent(SDL_Event* event)
{
    imguiHandleInput(ims, *event);

    if(event->type == SDL_QUIT) {
        running = false;
        return;
    }

    if(event->type == SDL_KEYDOWN) {
        if(event->key.keysym.sym == SDLK_ESCAPE) {
            running = false;
            return;
        }

        /*if(event->key.keysym.sym == SDLK_q) {
            birdInput[0].left = 1;
        }
        if(event->key.keysym.sym == SDLK_d) {
            birdInput[0].right = 1;
        }

        if(event->key.keysym.sym == SDLK_r) {
           resetBirds();
        }

        if(event->key.keysym.sym == SDLK_n) {
           resetTraining();
        }*/
    }

    /*if(event->type == SDL_KEYUP) {
        if(event->key.keysym.sym == SDLK_q) {
            birdInput[0].left = 0;
        }
        if(event->key.keysym.sym == SDLK_d) {
            birdInput[0].right = 0;
        }
    }*/

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

void resetMap()
{
    memset(mapData, MAP_TILE_GRASS, sizeof(mapData));
    i32 pondPos[MAP_POND_COUNT];
    i32 pondRadius[MAP_POND_COUNT];

    for(i32 i = 0; i < MAP_POND_COUNT; ++i) {
        pondPos[i] = xorshift64star() % mapSize;
        pondRadius[i] = randi64(MAP_POND_MIN_RADIUS, MAP_POND_MAX_RADIUS);
    }

    for(i32 i = 0; i < mapSize; ++i) {
        i32 x = i % MAP_WIDTH;
        i32 y = i / MAP_WIDTH;

        for(i32 p = 0; p < MAP_POND_COUNT; ++p) {
            i32 px = p % MAP_WIDTH;
            i32 py = p / MAP_WIDTH;
            i32 dist = sqrtf((px - x) * (px - x) + (py - y) * (py - y));

            if(dist <= pondRadius[p]) {
                mapData[i] = MAP_TILE_WATER;
            }
        }
    }
}

void newFrame()
{

}

void render()
{
    glClear(GL_COLOR_BUFFER_BIT);

    const f32 tileSize = 16;

    const f32 gl = 0;
    const f32 gr = tileSize * MAP_WIDTH;
    const f32 gt = 0;
    const f32 gb = tileSize * MAP_HEIGHT;
    const Color4 grassColor = {40, 89, 24, 255};
    const Color4 waterColor = {66, 138, 255, 255};

    Quad grassQuad = quadOneColor(gl, gr, gt, gb, grassColor);
    drawQuadBatch(&grassQuad, 1);

    constexpr i32 MAX_WATER_QUADS = 512;
    Quad waterQuad[MAX_WATER_QUADS];
    i32 waterCount = 0;

    for(i32 i = 0; i < mapSize; ++i) {
        i32 x = i % MAP_WIDTH;
        i32 y = i / MAP_WIDTH;

        if(mapData[i] == MAP_TILE_WATER) {
            waterQuad[waterCount++] = quadOneColor(x, x+tileSize, y, y+tileSize, waterColor);
            if(waterCount == MAX_WATER_QUADS) {
                drawQuadBatch(waterQuad, MAX_WATER_QUADS);
            }
        }
    }

    drawQuadBatch(waterQuad, waterCount);
}

void cleanup()
{

}

};

i32 main()
{
    LOG("=== F R O G S ===\n");
    LOG("   (o)____(o)");
    LOG(" _/          \\_ ");
    LOG("/ \\----------/ \\");
    LOG("\\   | |  | |   /");
    LOG(" ww ooo  ooo ww");
    LOG("\n");

    timeInit();

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
