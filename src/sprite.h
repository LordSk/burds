#pragma once
#include "base.h"

typedef union
{
    struct { f32 data[2]; };
    struct { f32 x, y; };
} Vec2;

f32 vec2Distance(Vec2* va, Vec2* vb);

typedef struct
{
    Vec2 pos;
    Vec2 size;
    Vec2 center;
    f32 rot;
} Transform;

typedef struct
{
    u8 r,g,b;
} Color3;

typedef struct
{
    Vec2 p1;
    Color3 c1;
    Vec2 p2;
    Color3 c2;
} Line;

i32 initSpriteState(i32 winWidth, i32 winHeight);
i32 loadTexture(const char* path);
void setView(i32 x, i32 y, i32 width, i32 height);
void drawSpriteBatch(i32 textureId, const Transform* transform, const Color3* color, const i32 count);
void drawLineBatch(const Line* lines, const i32 count);
