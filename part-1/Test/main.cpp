
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

const auto GpuOutLocation = "../Cuda/gpu-mandelbrot.ppm";
const auto CpuOutLocation = "../Cpu/cpu-mandelbrot.ppm";

struct FilePPM
{
    std::string version;
    std::int32_t width, height, maxValue;
    std::vector<char> pixels;
};

FilePPM readPortablePixmap(std::ifstream& fs)
{    
    FilePPM ppm;
    fs >> ppm.version;
    fs >> ppm.width;
    fs >> ppm.height;
    fs >> ppm.maxValue;
    fs.get();

    const auto size = ppm.width * ppm.height * 3;
    ppm.pixels.resize(size);
    fs.read(ppm.pixels.data(), size);
    return ppm;
}

int main(int argc, char* argv[])
{
    std::ifstream gpuOutput(GpuOutLocation, std::ios::binary);
    std::ifstream cpuOutput(CpuOutLocation, std::ios::binary);

    if(gpuOutput.is_open() && cpuOutput.is_open())
    {
        const auto gpuFile = readPortablePixmap(gpuOutput);
        const auto cpuFile = readPortablePixmap(cpuOutput);

        const auto sameData = strcmp(cpuFile.pixels.data(), gpuFile.pixels.data()) == 0;
        const auto sameVersion = cpuFile.version == gpuFile.version;
        const auto sameHeight = cpuFile.height == gpuFile.height;
        const auto sameWidth = cpuFile.width == gpuFile.width;

        std::stringstream ss;
        ss << "Same Data?    " << (sameData ? "True" : "False") << std::endl;
        ss << "Same Height?  " << (sameHeight ? "True" : "False") << std::endl;
        ss << "Same Width?   " << (sameWidth ? "True" : "False") << std::endl;
        ss << "Same Version? " << (sameVersion ? "True" : "False");

        std::cout << ss.str() << std::endl;
    }
    else
    {
        std::cout << "Error: Could not find files" << std::endl;
    }
}