#pragma once
#include "base.h"
#include "vec_math.h"

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
    u8 r,g,b,a;
} Color4;

inline Color4 color4(u8 r, u8 g, u8 b, u8 a)
{
    Color4 c4 = {r,g,b,a};
    return c4;
}

typedef struct
{
    Vec2 p1;
    Color4 c1;
    Vec2 p2;
    Color4 c2;
} Line;

typedef struct
{
    Vec2 p[4]; // tl,tr,bl,br
    Color4 c[4];
} Quad;

inline Quad quadOneColor(f32 left, f32 right, f32 top, f32 bottom, Color4 color)
{
    Quad q;
    q.p[0] = vec2Make(left, top);
    q.p[1] = vec2Make(right, top);
    q.p[2] = vec2Make(left, bottom);
    q.p[3] = vec2Make(right, bottom);
    q.c[0] = color;
    q.c[1] = color;
    q.c[2] = color;
    q.c[3] = color;
    return q;
}

i32 initSpriteState(i32 winWidth, i32 winHeight);
i32 loadTexture(const char* path);
void setView(i32 x, i32 y, i32 width, i32 height);
void drawSpriteBatch(i32 textureId, const Transform* transform, const Color3* color, const i32 count);
void drawLineBatch(const Line* lines, const i32 count);
void drawQuadBatch(const Quad* quads, const i32 count);
