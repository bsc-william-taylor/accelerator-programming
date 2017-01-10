
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct NunmericData
{
    int number;
};

__global__ void calculate(NunmericData* a, NunmericData* b, NunmericData*c)
{
    c->number = a->number + b->number;
}

__global__ void add(int* a, int* b, int* c)
{
    *c = *a + *b;
}

__global__ void sub(int* a, int* b, int* c)
{
    *c = *a - *b;
}
