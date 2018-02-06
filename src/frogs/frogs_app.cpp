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
#include "imgui/imgui_sdl2_setup.h"

#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 900
#define FRAMES_PER_SEC 60.0
#define FRAME_DT ((f64)(1.0/FRAMES_PER_SEC))

#define MAP_WIDTH 256
#define MAP_HEIGHT 128
#define TILE_SIZE 32
#define MAP_POND_MAX_COUNT 24
#define MAP_POND_MIN_RADIUS 12
#define MAP_POND_MAX_RADIUS 40

#define FROG_COUNT 1024
#define FROG_SIZE 60.0f
#define FROG_SPEED 200.0f
#define FROG_ANIM_JUMP 300
#define FROG_ANIM_TONGUE 200
#define FROG_JUMP_CD 450
#define FROG_TONGUE_CD 300

#define ENERGY_TOTAL 100.0f
#define ENERGY_JUMP_DRAIN 2.0f
#define ENERGY_EAT_DRAIN 5.0f

#define HYDRATION_TOTAL 100.0f
#define HYDRATION_DRAIN_PER_SEC 1.0f

#define VISION_SQUARE_WIDTH 15

enum {
    MAP_TILE_GRASS=0,
    MAP_TILE_WATER,
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
};

struct FrogInput
{
    f32 angle;
    i32 action; // 0: nothing, 1: jump, 2: tongue
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
i32 pondCount = 5;

i32 tex_frogStill;
i32 tex_frogJump;
i32 tex_frogTongue;

Vec2 frogPos[FROG_COUNT];
Transform frogTf[FROG_COUNT];
Color3 frogColor[FROG_COUNT];
u8 frogSprite[FROG_COUNT]; // still, jump, tongue
FrogInput frogInput[FROG_COUNT];
f64 frogJumpTime[FROG_COUNT];
f64 frogTongueTime[FROG_COUNT];

f32 frogEnergy[FROG_COUNT];
f32 frogHydration[FROG_COUNT];

bool dbgShowBars = true;

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

    glClearColor(0.2, 0.2, 0.2, 1.0f);

    tex_frogStill  = loadTexture("../frog_still.png");
    tex_frogJump   = loadTexture("../frog_jump.png");
    tex_frogTongue = loadTexture("../frog_tongue.png");
    if(tex_frogStill == -1 || tex_frogJump == -1 ||
       tex_frogTongue == -1) {
        return false;
    }

    resetMap();
    resetFrogColors();
    resetFrogs();

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
        }*/

        if(event->key.keysym.sym == SDLK_r) {
           resetMap();
           return;
        }

        /*if(event->key.keysym.sym == SDLK_n) {
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
    i32 pondPos[MAP_POND_MAX_COUNT];
    i32 pondRadius[MAP_POND_MAX_COUNT];

    for(i32 i = 0; i < pondCount; ++i) {
        pondPos[i] = xorshift64star() % mapSize;
        pondRadius[i] = randi64(MAP_POND_MIN_RADIUS, MAP_POND_MAX_RADIUS);
        //LOG("pond#%d pos=%d radius=%d", i, pondPos[i], pondRadius[i]);
    }

    for(i32 i = 0; i < mapSize; ++i) {
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
}

void resetFrogColors()
{
    const u32 colorMax = 0xFF;
    const u32 colorMin = 0x0;
    const u32 colorDelta = colorMax - colorMin;

    for(i32 i = 0; i < FROG_COUNT; ++i) {
        Color3 c;
        c.r = (rand() % colorDelta) + colorMin;
        c.g = (rand() % colorDelta) + colorMin;
        c.b = (rand() % colorDelta) + colorMin;
        frogColor[i] = c;
    }
}

void resetFrogs()
{
    memset(frogJumpTime, 0, sizeof(frogJumpTime));
    memset(frogTongueTime, 0, sizeof(frogTongueTime));

    for(i32 i = 0; i < FROG_COUNT; ++i) {
        frogPos[i] = vec2Make(randf64(0.0, MAP_WIDTH * TILE_SIZE), randf64(0.0, MAP_HEIGHT * TILE_SIZE));
        frogSprite[i] = randi64(0, 2);
    }

    for(i32 i = 0; i < FROG_COUNT; ++i) {
        frogInput[i].angle = randf64(0, TAU);
        frogInput[i].action = randi64(0, 2);
    }

    for(i32 i = 0; i < FROG_COUNT; ++i) {
        frogEnergy[i] = ENERGY_TOTAL;
        frogHydration[i] = HYDRATION_TOTAL;
    }
}

void doUI()
{
    ImGui::StyleColorsDark();

    ImGui::Begin("Options");

        ImGui::TextUnformatted("Map");
        ImGui::BeginGroup();
            ImGui::SliderInt("Pond count", &pondCount, 1, MAP_POND_MAX_COUNT);

            if(ImGui::Button("Reset map")) {
                resetMap();
            }
        ImGui::EndGroup();

        ImGui::Separator();

        ImGui::TextUnformatted("Game");
        ImGui::BeginGroup();
            ImGui::SliderInt("Time scale", &timeScale, 1, 10);

            if(ImGui::Button("Reset game")) {
                resetFrogs();
            }
        ImGui::EndGroup();

    ImGui::End();

    ImGui::Begin("Debug");

        ImGui::Checkbox("Show bars", &dbgShowBars);

    ImGui::End();
}

void updatePhysics()
{
    for(i32 i = 0; i < FROG_COUNT; ++i) {
        if(frogInput[i].action == INPUT_ACTION_JUMP) {
            i64 jumpDeltaMs = frogJumpTime[i] * 1000;
            bool doJump = jumpDeltaMs < FROG_ANIM_JUMP;

            if(doJump) {
                f32 angle = frogInput[i].angle - PI * 0.5;
                Vec2 moveVec = vec2Make(cosf(angle) * FROG_SPEED * FRAME_DT,
                                        sinf(angle) * FROG_SPEED * FRAME_DT);
                frogPos[i] = vec2Add(&frogPos[i], &moveVec);

                frogPos[i].x = clampf64(frogPos[i].x, 0, MAP_WIDTH * TILE_SIZE);
                frogPos[i].y = clampf64(frogPos[i].y, 0, MAP_HEIGHT * TILE_SIZE);
            }
        }
    }
}

void updateMechanics()
{
    for(i32 i = 0; i < FROG_COUNT; ++i) {
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
        frogHydration[i] -= HYDRATION_DRAIN_PER_SEC * FRAME_DT;
    }

    for(i32 i = 0; i < FROG_COUNT; ++i) {

    }
}

void newFrame()
{
    imguiUpdate(ims);

    updatePhysics();

    updateMechanics();

    for(i32 i = 0; i < FROG_COUNT; ++i) {
        frogTf[i].pos = frogPos[i];
        frogTf[i].rot = frogInput[i].angle;
        frogSprite[i] = frogInput[i].action;

        if(frogInput[i].action == INPUT_ACTION_JUMP) {
            i64 jumpDeltaMs = frogJumpTime[i] * 1000;
            bool doJump = jumpDeltaMs < FROG_ANIM_JUMP;
            frogSprite[i] = doJump ? FROG_TEX_JUMP : FROG_TEX_STILL;
        }
        else if(frogInput[i].action == INPUT_ACTION_EAT) {
            i64 tongueDeltaMs = frogTongueTime[i] * 1000;
            bool doTongue = tongueDeltaMs < FROG_ANIM_TONGUE;
            frogSprite[i] = doTongue ? FROG_TEX_TONGUE : FROG_TEX_STILL;
        }

        switch(frogSprite[i]) {
            case 0: {
                frogTf[i].size = vec2Make(FROG_SIZE, FROG_SIZE);
                frogTf[i].center = vec2Make(FROG_SIZE*0.5f, FROG_SIZE*0.5f);
            } break;

            case 1: {
                frogTf[i].size = vec2Make(FROG_SIZE, FROG_SIZE * 1.5625);
                frogTf[i].center = vec2Make(frogTf[i].size.x*0.5f, frogTf[i].size.y*0.4f);
            } break;

            case 2: {
                frogTf[i].size = vec2Make(FROG_SIZE, FROG_SIZE * 1.5625);
                frogTf[i].center = vec2Make(frogTf[i].size.x*0.5f, frogTf[i].size.y*0.65f);
            } break;
        }
    }
}

void render()
{
    doUI();

    glClear(GL_COLOR_BUFFER_BIT);

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

    // draw frogs
    constexpr i32 BATCH_MAX = 128;
    Transform tf[128];
    Color3 color[128];
    i32 batchCount = 0;

    const i32 texs[] = { tex_frogStill, tex_frogJump, tex_frogTongue };
    constexpr i32 frogTexCount = 3;

    for(i32 t = 0; t < frogTexCount; ++t) {
        batchCount = 0;

        for(i32 i = 0; i < FROG_COUNT; ++i) {
            if(frogSprite[i] == t) {
                i32 id = batchCount++;
                tf[id] = frogTf[i];
                color[id] = frogColor[i];
                if(batchCount == BATCH_MAX) {
                    drawSpriteBatch(texs[t], tf, color, BATCH_MAX);
                }
            }
        }

        drawSpriteBatch(texs[t], tf, color, batchCount);
    }

    if(dbgShowBars) {
        Quad barQuad[FROG_COUNT];
        i32 barCount = 0;
        const Color4 hydrationColor = {0, 255, 255, 200};
        const Color4 energyColor = {255, 255, 0, 200};
        constexpr f32 barWidth = 40.f;
        constexpr f32 barHeight = 5.f;

        for(i32 i = 0; i < FROG_COUNT; ++i) {
            f32 x = frogPos[i].x;
            f32 y = frogPos[i].y;
            f32 h = frogHydration[i] / HYDRATION_TOTAL;
            barQuad[barCount++] = quadOneColor(x - barWidth * 0.5, x + (barWidth * h),
                                               y - FROG_SIZE * 0.5, y - FROG_SIZE * 0.5 + barHeight,
                                               hydrationColor);
        }
        drawQuadBatch(barQuad, barCount);

        barCount = 0;
        for(i32 i = 0; i < FROG_COUNT; ++i) {
            f32 x = frogPos[i].x;
            f32 y = frogPos[i].y;
            f32 h = frogEnergy[i] / ENERGY_TOTAL;
            barQuad[barCount++] = quadOneColor(x - barWidth * 0.5, x + (barWidth * h),
                                               y - FROG_SIZE * 0.5 - barHeight, y - FROG_SIZE * 0.5,
                                               energyColor);
        }
        drawQuadBatch(barQuad, barCount);
    }

    imguiRender();
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
