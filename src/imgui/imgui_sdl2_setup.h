#pragma once
#include <gl3w.h>
#include <SDL2/SDL_events.h>
#include "vec_math.h"
#include "base.h"

struct ImGuiGLSetup;

struct ImGuiGLSetup* imguiInit(u32 width, u32 height, const char* configFilename = "imgui.ini");
void imguiDeinit(struct ImGuiGLSetup* ims);
void imguiUpdate(struct ImGuiGLSetup* ims);
void imguiHandleInput(struct ImGuiGLSetup* ims, SDL_Event event);
void imguiRender();
