#include "base.h"
#include <windows.h>

timept startCounter;
i64 PERFORMANCE_FREQUENCY;

void timeInit()
{
    LARGE_INTEGER li;
    QueryPerformanceFrequency(&li);
    PERFORMANCE_FREQUENCY = (i64)li.QuadPart;
    QueryPerformanceCounter(&li);
    startCounter = (i64)li.QuadPart;
    LOG("performanceFrequency=%lld", PERFORMANCE_FREQUENCY);
}

timept timeGet()
{
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return ((timept)li.QuadPart - startCounter);
}

i64 timeToMicrosec(i64 delta)
{
    return (delta * 1000000) / PERFORMANCE_FREQUENCY;
}

u64 g_RandSeed= 0xdeadbeefcdcd;
