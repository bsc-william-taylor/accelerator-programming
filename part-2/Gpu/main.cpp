#pragma warning (disable: 4996)

#include "../Cpu/ppm.hpp"
#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

namespace cl {
    /*
     * How is this not part of the fucking standard!
     */
    template<int L>
    cl::size_t<L> new_size_t(std::vector<int> numbers) {
        cl::size_t<L> sz;
        for (int i = 0; i < L; i++)
            sz[i] = numbers[i];
        return sz;
    }
}

const auto arg = [](auto argc, auto argv, auto index, auto value)
{
    return argc > index ? argv[index] : value;
};

auto kernel(const std::string& filename)
{
    std::ifstream file(filename);
    std::stringstream ss;
    std::string str;
    while (std::getline(file, str))
    {
        ss << str;
    }
    return ss.str();
}

auto rgb_to_rgba(std::vector<std::uint8_t>& rgb, int width, int height)
{
    std::vector<std::uint8_t> rgba(width*height * 4);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            auto rgb_index = (y * width + x) * 3;
            auto rgba_index = (y * width + x) * 4;

            rgba[rgba_index + 0] = rgb[rgb_index + 0];
            rgba[rgba_index + 1] = rgb[rgb_index + 1];
            rgba[rgba_index + 2] = rgb[rgb_index + 2];
            rgba[rgba_index + 3] = 255;
        }
    }

    return rgba;
}
auto rgb_from_rgba(std::vector<std::uint8_t>& rgba, int width, int height)
{
    std::vector<std::uint8_t> rgb(width*height * 3);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            auto rgb_index = (y * width + x) * 3;
            auto rgba_index = (y * width + x) * 4;

            rgb[rgb_index + 0] = rgba[rgba_index + 0];
            rgb[rgb_index + 1] = rgba[rgba_index + 1];
            rgb[rgb_index + 2] = rgba[rgba_index + 2];
        }
    }

    return rgb;
}

int main(int argc, const char * argv[])
{
    auto radius = std::atoi(arg(argc, argv, 3, "5"));
    auto output = arg(argc, argv, 2, "./cl-out.ppm");
    auto input = arg(argc, argv, 1, "../Cpu/lena.ppm");

    ppm image(input);

    auto rgba = rgb_to_rgba(image.data, image.w, image.h);

    std::string src = kernel("kernels.cl"), name, version;
    std::vector<cl::Device> devices;
    std::vector<std::uint8_t> outputPixels(rgba.size());

    cl::Platform p = cl::Platform::getDefault();
    cl::Program::Sources kernel(1, std::make_pair(src.c_str(), src.length()));

    p.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    p.getInfo(CL_PLATFORM_VERSION, &version);
    p.getInfo(CL_PLATFORM_NAME, &name);

    cl::Device device = devices.front();
    cl::Context context = cl::Context(device);
    cl::CommandQueue queue(context, device);
    cl::Program program(context, kernel);

    try
    {
        program.build();
    }
    catch (const cl::Error& e)
    {
        std::cerr
            << "Exception Caught: " << e.what() << std::endl
            << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)
            << std::endl;
    }

    try
    {
        cl::ImageFormat format{ CL_RGBA, CL_UNORM_INT8 };
        cl::Image2D imageBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, format, image.w, image.h, 0, rgba.data());
        cl::Image2D resultBuffer(context, CL_MEM_READ_WRITE, format, image.w, image.h);
        cl::size_t<3> region = cl::new_size_t<3>({ (int)image.w, (int)image.h, 1 });
        cl::size_t<3> origin = cl::new_size_t<3>({ 0, 0, 0 });

        cl::NDRange local(1, 1), global(image.w, image.h);
        cl::Kernel copy(program, "copy");
        copy.setArg(0, imageBuffer);
        copy.setArg(1, resultBuffer);

        queue.enqueueNDRangeKernel(copy, cl::NullRange, global, local);
        queue.enqueueReadImage(resultBuffer, CL_TRUE, origin, region, 0, 0, outputPixels.data());

        image.write(output, rgb_from_rgba(outputPixels, image.w, image.h));
    }
    catch (const cl::Error& e)
    {
        std::cerr << "Exception Caught: " << e.what() << std::endl;
        std::cerr << "Code: " << e.err() << std::endl;
    }

    return std::cin.get();
}