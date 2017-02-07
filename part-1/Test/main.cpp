
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

void findPixelErrors(const FilePPM& gpuFile, const FilePPM& cpuFile)
{
    for (auto i = 0; i < gpuFile.pixels.size(); i++)
    {
        auto& gpuPixels = gpuFile.pixels;
        auto& cpuPixels = cpuFile.pixels;

        if (gpuPixels[i] != cpuPixels[i])
        {
            std::cout << "(index) -> " << i / 3 << " ";
            i += 2;
        }
    }
}

int main(int argc, char* argv[])
{
    std::ifstream gpuOutput(GpuOutLocation, std::ios::binary);
    std::ifstream cpuOutput(CpuOutLocation, std::ios::binary);

    if(gpuOutput.is_open() && cpuOutput.is_open())
    {
        const auto gpuFile = readPortablePixmap(gpuOutput);
        const auto cpuFile = readPortablePixmap(cpuOutput);
        const auto size = cpuFile.width * cpuFile.height * 3;
        
        const auto sameData = memcmp(cpuFile.pixels.data(), gpuFile.pixels.data(), size) == 0;
        const auto sameVersion = cpuFile.version == gpuFile.version;
        const auto sameHeight = cpuFile.height == gpuFile.height;
        const auto sameWidth = cpuFile.width == gpuFile.width;

        std::stringstream ss;
        ss << "Same Data?    " << (sameData ? "True" : "False") << std::endl;
        ss << "Same Height?  " << (sameHeight ? "True" : "False") << std::endl;
        ss << "Same Width?   " << (sameWidth ? "True" : "False") << std::endl;
        ss << "Same Version? " << (sameVersion ? "True" : "False");

        if (!sameData)
        {
            findPixelErrors(gpuFile, cpuFile);
        }

        std::cout << ss.str() << std::endl;
    }
    else
    {
        std::cout << "Error: Could not find files" << std::endl;
    }

    return 0;
}