#include "cuda-utilities.h"

struct rgb_t 
{
    unsigned char r, g, b;
};

__constant__ const uint8_t CharMax = std::numeric_limits<uint8_t>::max();
__constant__ const uint32_t ColoursSize = 16;
__constant__ const rgb_t Colours[ColoursSize]
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

__constant__ const double Threshold = 0.0625;
__constant__ const double Unknown = 0.25;
__constant__ const double CenterX = -0.6;
__constant__ const double CenterY = 0.0;

__global__ void mandelbrot(cuda::launchInfo info, rgb_t* image, double scale)
{
    const auto j = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
    const auto i = static_cast<int>(threadIdx.y + blockIdx.y * blockDim.y);

    if (j >= info.height || i >= info.width)
    {
        return;
    }

    const auto y = (i - info.height / 2) * scale + CenterY;
    const auto x = (j - info.width / 2) * scale + CenterX;

    auto zx = hypot(x - Unknown, y);
    
    if (zx - 2 * pow(zx, 2) + Unknown >= x)
    {
        return;
    }

    if (pow(x + 1, 2) + pow(y, 2) < Threshold)
    {
        return;
    }

    double zy, zx2, zy2;
    zx = zy = zx2 = zy2 = 0.0;
    uint8_t iter = 0;
   
    do
    {
        zy = 2.0 * zx * zy + y;
        zx = zx2 - zy2 + x;
        zx2 = pow(zx, 2);
        zy2 = pow(zy, 2);
    } 
    while (++iter < CharMax && zx2 + zy2 < 4.0);

    if (iter != CharMax && iter != 0)
    {
        const auto index = j + info.width * (info.height - i - 1);
        image[index] = Colours[iter % ColoursSize];
    }
}

void writeOutput(const char* filename, void* data, int width, int height)
{ 
    std::ofstream file(filename);
    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(static_cast<char*>(data), height * width * sizeof(rgb_t));
}

int main(int argc, char *argv[])
{
    const auto height = argc > 1 ? atoi(argv[1]) : 4096;
    const auto width = argc > 2 ? atoi(argv[2]) : 4096;
    const auto scale = 1.0 / (width / 4);

    std::vector<rgb_t> hostMemory(height * width);

    cuda::launchInfo launchInfo = optimumLaunch(mandelbrot, width, height, hostMemory.size());
    cuda::memory<rgb_t*> deviceMemory{ hostMemory.size() * sizeof(rgb_t), 0 };
    cuda::start(mandelbrot, launchInfo, deviceMemory, scale); cuda::move(deviceMemory, hostMemory.data());

    writeOutput("gpu-mandelbrot.ppm", hostMemory.data(), width, height);   
    return 0;
}