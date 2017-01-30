#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <vector>

#include "../cuda-utilities.h"
#include "../benchmark.h"

struct rgb
{
    unsigned char r, g, b;
};

const auto OutputFilename = std::string("output.ppm");
const auto Height = 256;
const auto Width = 256;

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

__global__ void mandel(rgb* image, int width, int height, double scale)
{
    auto px = image;

    for (auto i = 0; i < height; i++)
    {
        px = image + i * width;

        for (auto j = 0; j < width; j++, px++)
        {
            auto y = (i - height / 2) * scale + cy;
            auto x = (j - width / 2) * scale + cx;
            auto zx = hypot(x - .25, y), zy = 0.0, zx2 = 0.0, zy2 = 0.0;


            unsigned char iter = 0;
            if (x < zx - 2 * zx * zx + .25 || (x + 1)*(x + 1) + y * y < 1 / 16)
                iter = MaxIterations;

            do
            {
                zy = 2 * zx * zy + y;
                zx = zx2 - zy2 + x;
                zx2 = zx * zx;
                zy2 = zy * zy;
            } while (iter++ < MaxIterations && zx2 + zy2 < 4);

            *px = { iter };
        }
    }

    px = image;

    for(auto i = 0; i < width*height; ++i)
    {
        if (px->r == MaxIterations || px->r == 0)
        {
            *px = { 0 };
        }
        else
        {
            *px = mapping[px->r % numberShades];
        }

        ++px;
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
    benchmark<measure_in::ms, 1>([&]()
    {        
        std::vector<rgb> image(Height * Width);

        cuda::memory<rgb*> imagePointer(image.data(), image.size() * sizeof(rgb));
        cuda::start<1, 1>(mandel, imagePointer, Width, Height, 1.0 / (Width / 4));
        
        imagePointer.transfer(image.data());
        writePPM(OutputFilename, image, Width, Height);
    });
    
    return 0;
}