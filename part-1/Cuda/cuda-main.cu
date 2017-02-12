#include "cuda-utilities.h"

struct rgb_t 
{
    unsigned char r, g, b;
};

__constant__ const uint8_t MaxIterations = std::numeric_limits<uint8_t>::max();
__constant__ const double CenterX = -0.6, CenterY = 0.0;
__constant__ const uint32_t MappingsLength = 16;
__constant__ const rgb_t Mappings[MappingsLength]
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

__global__ void mandelbrot(cuda::launchInfo info, rgb_t* image, double scale)
{
    const auto j = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
    const auto i = static_cast<int>(threadIdx.y + blockIdx.y * blockDim.y);

    if (i >= info.size || j >= info.size)
    {
        return;
    }

    const auto x = (j - info.size / 2) * scale + CenterX;
    const auto y = (i - info.size / 2) * scale + CenterY;

    auto zx = hypot(x - 0.25, y);
    
    if (x < zx - 2 * zx * zx + 0.25 || (x + 1)*(x + 1) + y * y < 1 / 16)
    {
        return;
    }

    uint8_t iter = 0;
    auto zy = 0.0, zx2 = 0.0, zy2 = 0.0;
    zx = 0.0;

    do
    {
        zy = 2.0 * zx * zy + y;
        zx = zx2 - zy2 + x;
        zx2 = zx * zx;
        zy2 = zy * zy;
    } while (iter++ < 255 && zx2 + zy2 < 4);

    if (iter != MaxIterations && iter != 0)
    {
        const auto flippedY = info.size - 1 - i;
        image[j + info.size * flippedY] = Mappings[iter % MappingsLength];
    }
}

void writeOutput(const std::string& filename, void* image, int width, int height)
{ 
    std::ofstream file("gpu-mandelbrot.ppm");
    file << "P6\n" << width << " " << height << "\n255\n"; 
    file.write(static_cast<char*>(image), height*width*sizeof(rgb_t));
    file.close();
}

int main(int argc, char *argv[])
{
    const auto height = argc > 1 ? atoi(argv[1]) : 4096*4;
    const auto width = argc > 2 ? atoi(argv[2]) : 4096*4;
    const auto scale = 1.0 / (width / 4);

    std::vector<rgb_t> image(height * width);

    cuda::launchInfo launchInfo = optimumLaunch(mandelbrot, image.size());
    cuda::memory<rgb_t*> imagePointer{ image.size() * sizeof(rgb_t), 0 };
    cuda::benchmark<10>([&]()
    {
        start(mandelbrot, launchInfo, imagePointer, scale);
    }) ;

    cuda::move(imagePointer, image.data());

    writeOutput("gpu-mandelbrot.ppm", image.data(), width, height);
    return 0;
}