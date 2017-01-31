#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <memory>
#include <sstream>

#include "../cuda-utilities.h"
#include "../benchmark.h"

struct workload
{
    int offset, size, stride;
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

__device__ workload calculateWork(cuda::launchInfo& info, int block, int thread)
{
    auto blockSizeY = info.height / info.blocks;
    auto blockSize = info.width * blockSizeY;
    auto threadSize = blockSize / info.threads;

    auto offset = block * blockSize + thread * threadSize;
    auto stride = block * blockSizeY + thread * blockSizeY / info.threads;

    return { offset, offset + threadSize, stride };
}

__global__ void mandel(cuda::launchInfo info, rgb* image, double scale)
{
    auto workload = calculateWork(info, blockIdx.x, threadIdx.x);
    auto pointX = 0, pointY = workload.stride;
    auto px = image + workload.offset;

    for(auto i = workload.offset; i < workload.size; ++i, ++px)
    {
        auto y = (pointY - info.height / 2) * scale + cy;
        auto x = (pointX - info.width / 2) * scale + cx;
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
        ++pointX;

        if (!(pointX < info.width)) {
            ++pointY;
            pointX = 0;
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

void writePPM(const std::string& filename, rgb* image, const int width, const int height)
{
    const auto file = fopen(filename.c_str(), "w");
    fprintf(file, "P6\n%d %d\n255\n", width, height);

    for (auto i = height - 1; i >= 0; i--)
    {
        fwrite(image + i * width, 1, width * sizeof(rgb), file);
    }
       
    fclose(file);
}

void onCUDAError(int number, const char* msg)
{
    std::cout << number << ": " << msg << std::endl;
    std::cin.get();
};

int main(int argc, char *argv[])
{
    const auto height = 4096, width = 4096, threads = 512, blocks = 8;
    const auto scale = 1.0 / (width / 4);

    std::unique_ptr<rgb[]> image(new rgb[height*width]{ 0 });
    std::stringstream ss;
    ss << "mandelbrot-";
    ss << width << "x" << height;
    ss << ".ppm";

    benchmark<measure_in::ms, 1>([&]() 
    {
        cuda::launchInfo launchInfo { blocks, threads, width, height };
        cuda::memory<rgb*> imagePointer { image.get(), height*width*sizeof(rgb) };
        cuda::start(mandel, launchInfo, imagePointer, scale);
        cuda::errorCheck(onCUDAError);

        imagePointer.transfer(image.get());

        writePPM(ss.str(), image.get(), width, height);
    });

    return 0;
}