#pragma once
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <random>

typedef int8_t i8;
typedef uint8_t u8;
typedef int16_t i16;
typedef uint16_t u16;
typedef int32_t i32;
typedef int64_t i64;
typedef uint32_t u32;
typedef uint64_t u64;
typedef float f32;
typedef double f64;

#define LOG(fmt, ...) (printf(fmt "\n", __VA_ARGS__), fflush(stdout))
#define TIME_MILLI() (clock() / (CLOCKS_PER_SEC / 1000))
#define TIME_MICRO() (clock() / CLOCKS_PER_SEC)
#define MAKE_STR(s) #s
#define arr_count(arr) (sizeof(arr)/sizeof(arr[0]))
#define stack_arr(type, count) ((type*)alloca(sizeof(type) * count))
#define arr_zero(arr, count) (memset(arr, 0, sizeof(arr[0]) * count))
#define mem_zero(arr) (memset(arr, 0, sizeof(arr)))

#define TRUE 1
#define FALSE 0

#define PI (3.14159265359)
#define TAU (6.28318530718)

#define I32_MAX 0x7fffffff
#define U32_MAX 0xffffffff
#define U64_MAX 0xffffffffffffffff

#ifndef max
    #define max(a,b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
    #define min(a,b) (((a) < (b)) ? (a) : (b))
#endif


// TIME
typedef i64 timept;
void timeInit();
timept timeGet();
i64 timeToMicrosec(i64 delta);
i64 timeGetMicro();


// RANDOM
extern u64 g_RandSeed;

static std::random_device g_randomDevice;
static std::mt19937 g_randomMt(g_randomDevice());

inline void randSetSeed(u64 seed)
{
    g_RandSeed = seed;
}

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

template<typename T>
inline T clamp(T val, T vmin, T vmax)
{
    if(val < vmin) return vmin;
    if(val > vmax) return vmax;
    return val;
}

inline f64 randf64(f64 vmin, f64 vmax)
{
    std::uniform_real_distribution<f64> dis(vmin, vmax);
    return dis(g_randomMt);
    /*u64 r = xorshift64star();
    return vmin + ((f64)r/(f64)U64_MAX) * (vmax - vmin);*/
}


inline i64 randi64(i64 min, i64 max)
{
    u64 r = xorshift64star();
    return min + (r % (max - min + 1));
}
