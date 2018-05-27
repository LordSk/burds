#pragma once
#include "base.h"
#include "sprite.h"
#include <SDL2/SDL.h>
#include <gl3w.h>
#include "imgui/imgui.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui/imgui_internal.h"
#include "imgui/imgui_sdl2_setup.h"

#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 900

struct AppWindow
{
    SDL_Window* sdlWin;
    SDL_GLContext glContext;
    ImGuiGLSetup* ims;
    i32 winWidth;
    i32 winHeight;
    bool running = true;

    bool init(const char* title, const char* configName, i32 width = WINDOW_WIDTH,
              i32 height = WINDOW_HEIGHT, bool fullscreen = false)
    {
        winWidth = width;
        winHeight = height;

        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);

        sdlWin = SDL_CreateWindow(title,
                                  SDL_WINDOWPOS_CENTERED,
                                  SDL_WINDOWPOS_CENTERED,
                                  width, height,
                                  SDL_WINDOW_OPENGL|(fullscreen ? SDL_WINDOW_FULLSCREEN:0));

        if(!sdlWin) {
            LOG("ERROR: can't create SDL2 window (%s)",  SDL_GetError());
            return false;
        }

        glContext = SDL_GL_CreateContext(sdlWin);
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

        if(!initSpriteState(winWidth, winHeight)) {
            return false;
        }

        ims = imguiInit(winWidth, winHeight, configName);
        if(!ims) {
            LOG("ERROR: could not init imgui");
        }

        glClearColor(0.2, 0.2, 0.2, 1.0f);
        setView(0, 0, winWidth, winHeight);

        return true;
    }

    inline void uiNewFrame()
    {
        imguiUpdate(ims);
    }

    inline void uiRender()
    {
        imguiRender();
    }

    inline void uiHandleEvent(SDL_Event* event)
    {
        imguiHandleInput(ims, *event);
    }

    inline void swap()
    {
        SDL_GL_SwapWindow(sdlWin);
    }

    inline void cleanup()
    {
        free(ims);
        SDL_DestroyWindow(sdlWin);
    }
};
