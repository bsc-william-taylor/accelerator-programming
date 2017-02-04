#include "cuda-utilities.h"
#include <cstdio>
#include "../benchmark.h"

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
    const auto i = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
    const auto x = (i % info.size - info.size / 2) * scale + CenterX;
    const auto y = (i / info.size - info.size / 2) * scale + CenterY;

    uchar iter = 0;

    auto zy = 0.0, zx2 = 0.0, zy2 = 0.0;
    auto zx = hypot(x - .25, y);

    if (x < zx - 2 * zx * zx + .25) 
    {
        iter = MaxIterations;
    }
    else if ((x + 1)*(x + 1) + y * y < 1 / 16)
    {
        iter = MaxIterations;
    }
    else
    {
        do 
        {
            zy = 2 * zx * zy + y;
            zx = zx2 - zy2 + x;
            zx2 = zx * zx;
            zy2 = zy * zy;
        } while (iter++ < 255 && zx2 + zy2 < 4);
    }


    if (iter == MaxIterations || iter == 0)
    {
        image[i] = { 0 };
    }
    else
    {
        image[i] = Mappings[iter % MappingsLength];
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
    const auto height = 4096, width = 4096;
    const auto scale = 1.0 / (width / 4.0);

    std::vector<rgb> image(height * width);

    cuda::launchInfo launchInfo = optimumLaunch(mandelbrot, image.size());
    cuda::memory<rgb*> imagePointer{ image.data(), image.size() * sizeof(rgb) };
    cuda::start(mandelbrot, launchInfo, imagePointer, scale);
    cuda::move(imagePointer, image.data());

    writeOutput("output.ppm", image.data(), width, height);   
    return 0;
}