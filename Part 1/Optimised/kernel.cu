#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <vector>

#include "../cuda-utilities.h"
#include "../benchmark.h"

struct workload
{
    int offset, size;
};

struct rgb
{
    unsigned char r, g, b;
};

__constant__ const unsigned char MaxIterations = 255;
__constant__ const unsigned char numberShades = 16;
__constant__ const auto cx = -0.6, cy = 0.0;
__constant__ const rgb mapping[numberShades] =
{
    { 66,  30,   15 },
    { 25,   7,   26 },
    { 9,    1,   47 },
    { 4,    4,   73 },
    { 0,    7,   100 },
    { 12,   44,  138 },
    { 24,   82,  177 },
    { 57,   125, 209 },
    { 134,  181, 229 },
    { 211,  236, 248 },
    { 241,  233, 191 },
    { 248,  201, 95 },
    { 255,  170, 0 },
    { 204,  128, 0 },
    { 153,  87,  0 },
    { 106,  52,  3 }
};

#define PROCESSORS 8

__device__ workload getWorkLoad(int block, int thread, int width, int height)
{
    auto worksize = width*height / PROCESSORS;
    auto offset = block * worksize;
    return { offset, offset + worksize };
}

__global__ void mandel(rgb* image, int width, int height, double scale)
{
    auto workload = getWorkLoad(blockIdx.x, threadIdx.x, width, height);
    auto curX = 0, curY = static_cast<int>(blockIdx.x * height / PROCESSORS);
    auto px = image + workload.offset;

    for(auto i = workload.offset; i < workload.size; ++i, ++px)
    {
        auto y = (curY - height / 2) * scale + cy;
        auto x = (curX - width / 2) * scale + cx;
        auto zx = hypot(x - 0.25, y), zy = 0.0, zx2 = 0.0, zy2 = 0.0;

        unsigned char iter = 0;

        if (x < zx - 2 * zx * zx + .25 || (x + 1)*(x + 1) + y * y < 1 / 16)
        {
            iter = MaxIterations;
        }

        do
        {
            zy = 2 * zx * zy + y;
            zx = zx2 - zy2 + x;
            zx2 = zx * zx;
            zy2 = zy * zy;
        } while (iter++ < MaxIterations && zx2 + zy2 < 4);

        *px = { iter };
        ++curX;

        if (!(curX < width)) {
            ++curY;
            curX = 0;
        }
    }

    px = image + workload.offset;

    for(auto i = workload.offset; i < workload.size; ++i, ++px)
    {
        if (px->r == MaxIterations || px->r == 0)
        {
            *px = { 0 };
        }
        else
        {
            *px = mapping[px->r % numberShades];
        }
    }
}

void writePPM(const std::string& filename, std::vector<rgb>& image, const int width, const int height)
{
    const auto bytes = width * sizeof(rgb);
    const auto file = fopen(filename.c_str(), "w");

    fprintf(file, "P6\n%d %d\n255\n", width, height);

    for (auto i = height - 1; i >= 0; i--)
    {
        fwrite(image.data() + (i * width), 1, bytes, file);
    }
       
    fclose(file);
}

const auto onCUDAError = [](auto number, auto msg)
{
    std::cout << msg << std::endl;
    std::cin.get();
};

int main(int argc, char *argv[])
{
    const auto outputFilename = std::string("output.ppm");
    const auto height = 4096, width = 4096;

    benchmark<measure_in::ms, 1>([&]()
    {        
        std::vector<rgb> image(height * width);

        cuda::memory<rgb*> imagePointer(image.data(), image.size() * sizeof(rgb));
        cuda::start<PROCESSORS, 1>(mandel, imagePointer, width, height, 1.0 / (width / 4));
        
        imagePointer.transfer(image.data());
        writePPM(outputFilename, image, width, height);
    });
}