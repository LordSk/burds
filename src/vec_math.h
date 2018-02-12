#pragma once
#include <math.h>
#include <string.h>
#include <assert.h>

typedef float f32;

typedef union Vec2
{
    struct { f32 data[2]; };
    struct { f32 x, y; };
} Vec2;

inline Vec2 vec2Make(f32 x, f32 y)
{
    Vec2 v = {x, y};
    return v;
}

inline f32 vec2Distance(const Vec2* va, const Vec2* vb)
{
    return sqrtf((vb->x - va->x) * (vb->x - va->x) +
           (vb->y - va->y) * (vb->y - va->y));
}

inline f32 vec2Dot(Vec2* va, Vec2* vb)
{
    return (va->x * vb->x) + (va->y * vb->y);
}

inline f32 vec2Cross(const Vec2* va, const Vec2* vb)
{
    return va->x * vb->y - va->y * vb->x;
}

inline f32 vec2Len(const Vec2* v)
{
    return sqrt((v->x * v->x) + (v->y * v->y));
}

inline Vec2 vec2Add(const Vec2* v1, const Vec2* v2)
{
    Vec2 v = {v1->x + v2->x, v1->y + v2->y};
    return v;
}

inline Vec2 vec2Minus(const Vec2* v1, const Vec2* v2)
{
    Vec2 v = {v1->x - v2->x, v1->y - v2->y};
    return v;
}

inline Vec2 vec2Normalize(const Vec2* v)
{
    f32 len = vec2Len(v);
    if(len == 0.0f) len = 1.0f;
    return Vec2{v->x / len, v->y / len};
}

inline f32 vec2Angle(const Vec2* v)
{
    Vec2 vx = {1.0f, 0.0f};
    Vec2 vn = vec2Normalize(v);
    f32 cross = vec2Cross(&vn, &vx);
    f32 dot = vec2Dot(&vn, &vx);
    if(dot > 1.0f) dot = 1.0f;
    if(dot < -1.0f) dot = -1.0f;
    return atan2(cross, dot);
}

inline f32 vec2AngleBetween(const Vec2* va, const Vec2* vb)
{
    Vec2 van = vec2Normalize(va);
    Vec2 vbn = vec2Normalize(vb);
    f32 cross = vec2Cross(&van, &vbn);
    f32 dot = vec2Dot(&van, &vbn);
    if(dot > 1.0f) dot = 1.0f;
    if(dot < -1.0f) dot = -1.0f;
    return atan2(cross, dot);
}

struct Mat4
{
    f32 md[16];
};

inline Mat4 mat4Mul(const Mat4* m1, const Mat4* m2)
{
    Mat4 m;
    f32* md = m.md;
    const f32* md1 = m1->md;
    const f32* md2 = m2->md;

    md[0] = md1[0] * md2[0] + md1[4] * md2[1] + md1[8] * md2[2] + md1[12] * md2[3];
    md[1] = md1[1] * md2[0] + md1[5] * md2[1] + md1[9] * md2[2] + md1[13] * md2[3];
    md[2] = md1[2] * md2[0] + md1[6] * md2[1] + md1[10] * md2[2] + md1[14] * md2[3];
    md[3] = md1[3] * md2[0] + md1[7] * md2[1] + md1[11] * md2[2] + md1[15] * md2[3];

    md[4] = md1[0] * md2[4] + md1[4] * md2[5] + md1[8] * md2[6] + md1[12] * md2[7];
    md[5] = md1[1] * md2[4] + md1[5] * md2[5] + md1[9] * md2[6] + md1[13] * md2[7];
    md[6] = md1[2] * md2[4] + md1[6] * md2[5] + md1[10] * md2[6] + md1[14] * md2[7];
    md[7] = md1[3] * md2[4] + md1[7] * md2[5] + md1[11] * md2[6] + md1[15] * md2[7];

    md[8] = md1[0] * md2[8] + md1[4] * md2[9] + md1[8] * md2[10] + md1[12] * md2[11];
    md[9] = md1[1] * md2[8] + md1[5] * md2[9] + md1[9] * md2[10] + md1[13] * md2[11];
    md[10] = md1[2] * md2[8] + md1[6] * md2[9] + md1[10] * md2[10] + md1[14] * md2[11];
    md[11] = md1[3] * md2[8] + md1[7] * md2[9] + md1[11] * md2[10] + md1[15] * md2[11];

    md[12] = md1[0] * md2[12] + md1[4] * md2[13] + md1[8] * md2[14] + md1[12] * md2[15];
    md[13] = md1[1] * md2[12] + md1[5] * md2[13] + md1[9] * md2[14] + md1[13] * md2[15];
    md[14] = md1[2] * md2[12] + md1[6] * md2[13] + md1[10] * md2[14] + md1[14] * md2[15];
    md[15] = md1[3] * md2[12] + md1[7] * md2[13] + md1[11] * md2[14] + md1[15] * md2[15];

    return m;
}

inline Mat4 mat4Translate(const Vec2* v2)
{
    Mat4 m;
    memset(&m, 0, sizeof(m));
    m.md[0] = 1.f;
    m.md[5] = 1.f;
    m.md[10] = 1.f;
    m.md[15] = 1.f;

    m.md[12] = v2->x;
    m.md[13] = v2->y;
    return m;
}

inline Mat4 mat4Scale(const Vec2* v2)
{
    Mat4 m;
    memset(&m, 0, sizeof(m));
    m.md[0] = v2->x;
    m.md[5] = v2->y;
    m.md[10] = 1.f;
    m.md[15] = 1.f;
    return m;
}

inline Mat4 mat4Rotate(f32 angle)
{
    Mat4 m;
    memset(&m, 0, sizeof(m));
    m.md[0] = cosf(angle);
    m.md[1] = -sinf(angle);
    m.md[4] = sinf(angle);
    m.md[5] = cosf(angle);
    m.md[10] = 1.f;
    m.md[15] = 1.f;
    return m;
}

inline Mat4 mat4Ortho(f32 left, f32 right, f32 top, f32 bottom, f32 nearPlane, f32 farPlane)
{
    Mat4 ortho;
    memset(&ortho, 0, sizeof(ortho));
    ortho.md[15] = 1.f;

    ortho.md[0] = 2.f / (right - left);
    ortho.md[5] = 2.f / (top - bottom);
    ortho.md[10] = -2.f / (farPlane - nearPlane);
    ortho.md[12] = -((right + left) / (right - left));
    ortho.md[13] = -((top + bottom) / (top - bottom));
    ortho.md[14] = -((farPlane + nearPlane) / (farPlane - nearPlane));
    return ortho;
}
