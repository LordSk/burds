#include <stdio.h>
#include <stdint.h>
#include <time.h>

typedef int32_t i32;
typedef uint8_t u8;
typedef float f32;

#define LOG(fmt, ...) (printf(fmt "\n", __VA_ARGS__), fflush(stdout))
#define TIME_MILLI() (clock() / (CLOCKS_PER_SEC / 1000))
#define MAKE_STR(s) #s

#define TRUE 1
#define FALSE 0
