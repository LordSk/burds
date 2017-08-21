#pragma once
#include <gl3w.h>
#include <SDL2/SDL_events.h>
#include "vec_math.h"
#include "base.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ImGuiGLSetup;

struct ImGuiGLSetup* imguiInit(u32 width, u32 height);
void imguiDeinit(struct ImGuiGLSetup* ims);
void imguiUpdate(struct ImGuiGLSetup* ims, f64 delta);
void imguiHandleInput(struct ImGuiGLSetup* ims, SDL_Event event);
void imguiRender();

void imguiTestWindow();

void imguiBegin(const char* name);
void imguiEnd();
u8 imguiSliderFloat(const char* label, f32* v, f32 v_min, f32 v_max);
u8 imguiButton(const char* label);
void imguiSameLine();

#ifdef __cplusplus
}
#endif
