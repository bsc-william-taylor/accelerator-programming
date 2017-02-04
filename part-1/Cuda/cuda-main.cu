#include "cuda-utilities.h"
#include <cstdio>
#include <cuda_runtime.h>

struct rgb
{
    unsigned char r, g, b;
};

using uchar = unsigned char;
using uint = unsigned int;

__constant__ const uchar MaxIterations = std::numeric_limits<uchar>::max();
__constant__ const uint MappingsLength = 16;
__constant__ const double CenterX = -0.6;
__constant__ const double CenterY = 0.0;
__constant__ const rgb Mappings[MappingsLength]
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

__global__ void mandelbrot(cuda::launchInfo info, rgb* image, double scale)
{
    auto workload = cuda::allocateWork(info, blockIdx.x, threadIdx.x);
    auto px = image + workload.offset;

    for(auto i = workload.offset; i < workload.size; ++i, ++px)
    {
        auto x = (i % info.width - info.width / 2) * scale + CenterX;
        auto y = (i / info.width - info.height / 2) * scale + CenterY;
        auto zx = hypot(x - 0.25, y), zy = 0.0, zx2 = 0.0, zy2 = 0.0;

        uchar iterations = 0;
        
        if (x < zx - 2 * zx * zx + .25)
        {
            iterations = MaxIterations;
        }
        else if((x + 1)*(x + 1) + y * y < 1 / 16)
        {
            iterations = MaxIterations;
        }
        else
        {
            do
            {
                zy = 2 * zx * zy + y;
                zx = zx2 - zy2 + x;
                zx2 = zx * zx;
                zy2 = zy * zy;
            } while (iterations++ < MaxIterations && zx2 + zy2 < 4);
        }

        if (iterations == MaxIterations || iterations == 0)
        {
            *px = { 0 };
        }
        else
        {
            *px = Mappings[iterations % MappingsLength];
        }
    }
}

void writeOutput(const std::string& filename, rgb* image, int width, int height)
{
    auto file = fopen(filename.c_str(), "w");
        
    if(file != nullptr)
    {
        fprintf(file, "P6\n%d %d\n255\n", width, height);

        for (auto i = height - 1; i >= 0; --i) 
        {
            fwrite(image + i * width, 1, width * sizeof(rgb), file);
        }

        fclose(file);
    }
}

int main(int argc, char *argv[])
{
    const auto height = 256, width = 256, threads = 64, blocks = 64;
    const auto scale = 1.0 / (width / 4);

    std::vector<rgb> image(height * width);

    cuda::launchInfo launchInfo{ blocks, threads, width, height };
    cuda::memory<rgb*> imagePointer{ image.data(), image.size() * sizeof(rgb) };
    cuda::start(mandelbrot, launchInfo, imagePointer, scale);
    cuda::move(imagePointer, image.data());

    writeOutput("output.ppm", image.data(), width, height);
    return 0;
}