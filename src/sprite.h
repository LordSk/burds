#pragma once
#include "base.h"

typedef union
{
    struct { f32 data[2]; };
    struct { f32 x, y; };
} Vec2;

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

i32 initSpriteState(i32 winWidth, i32 winHeight);
i32 loadTexture(const char* path);
void drawSpriteBatch(i32 textureId, const Transform* transform, const Color3* color, const i32 count);
