#pragma once
#include <stdio.h>
#include <stdint.h>
#include <time.h>

typedef int32_t i32;
typedef int64_t i64;
typedef uint32_t u32;
typedef uint64_t u64;
typedef uint8_t u8;
typedef float f32;
typedef double f64;

#define LOG(fmt, ...) (printf(fmt "\n", __VA_ARGS__), fflush(stdout))
#define TIME_MILLI() (clock() / (CLOCKS_PER_SEC / 1000))
#define TIME_MICRO() (clock() / CLOCKS_PER_SEC)
#define MAKE_STR(s) #s

#define TRUE 1
#define FALSE 0

#define PI (3.14159265359)
#define TAU (6.28318530718)

#define U64_MAX 0xffffffffffffffff


// TIME
typedef i64 timept;
void timeInit();
timept timeGet();
i64 timeToMicrosec(i64 delta);



// RANDOM
extern u64 g_RandSeed;

inline u64 xorshift64star()
{
    u64 x = g_RandSeed;	/* The state must be seeded with a nonzero value. */
    x ^= x >> 12; // a
    x ^= x << 25; // b
    x ^= x >> 27; // c
    g_RandSeed = x;
    return x * 0x2545F4914F6CDD1D;
}

inline f64 clampf64(f64 val, f64 vmin, f64 vmax)
{
    if(val < vmin) return vmin;
    if(val > vmax) return vmax;
    return val;
}

inline f64 randf64(f64 min, f64 max)
{
    u64 r = xorshift64star();
    return min + ((f64)r/U64_MAX) * (max - min);
}


inline f64 randi64(i64 min, i64 max)
{
    u64 r = xorshift64star();
    return min + (r % (max - min + 1));
}
