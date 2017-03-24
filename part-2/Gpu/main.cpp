
#pragma warning (disable: 4996)

#define _USE_MATH_DEFINES

#include "../library/benchmark.hpp"
#include "../library/utilities.hpp"
#include "../library/ppm.hpp"

#include <cl/cl.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <math.h>

std::vector<float> filter(const int radius, const float weight = 1.0f)
{
    std::vector<float> matrix;
    matrix.reserve(radius*radius);

    float stdv = weight, s = 2.0 * stdv * stdv;  
    float sum = 0.0;  

    const int size = floor(radius / 2.0);

    for (int x = -size; x <= size; x++)
    {
        for (int y = -size; y <= size; y++)
        {
            float r = sqrt(x*x + y*y);
            auto value = (exp(-(r*r) / s)) * 1.0 / (sqrt(2.0 * M_PI) * stdv);
            sum += value;
            matrix.push_back(value);
        }
    }

    for (int i = 0; i < matrix.size(); i++)
        matrix[i] /= sum;

    return matrix;
}

int main(int argc, const char * argv[])
{
    ppm image(arg(argc, argv, 1, "../Library/lena.ppm"));

    auto testing = pow((3 * 2 - 1), 2);
    auto radius = (int)pow(std::atoi(arg(argc, argv, 3, "3")), 2);
    auto output = arg(argc, argv, 2, "./cl-out.ppm");
    auto rgba = rgb_to_rgba(image.data, image.w, image.h);

    std::string src = kernel("kernels.cl");
    std::vector<cl::Device> devices;
    std::vector<std::uint8_t> outputPixels(rgba.size());
    std::vector<float> mask = filter(radius, 5.0f);

    cl::Program::Sources kernel(1, std::make_pair(src.c_str(), src.length()));
    cl::Platform p = cl::Platform::getDefault();
    p.getDevices(CL_DEVICE_TYPE_GPU, &devices);

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
        std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
    }

    try
    {
        cl::ImageFormat format{ CL_RGBA, CL_UNORM_INT8 };
        cl::Buffer maskBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mask.size() * sizeof(float), mask.data());
        cl::Image2D imageBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, format, image.w, image.h, 0, rgba.data());
        cl::Image2D outputBuffer(context, CL_MEM_READ_WRITE, format, image.w, image.h);

        cl::size_t<3> region = cl::new_size_t<3>({ (int)image.w, (int)image.h, 1 });
        cl::size_t<3> origin = cl::new_size_t<3>({ 0, 0, 0 });
        cl::NDRange local(16, 16), global(image.w, image.h);
        cl::Kernel blur(program, "unsharp_mask");

        blur.setArg(0, imageBuffer);
        blur.setArg(1, outputBuffer);
        blur.setArg(2, maskBuffer);
        blur.setArg(3, radius);

        queue.enqueueNDRangeKernel(blur, cl::NullRange, global, local);
        queue.enqueueReadImage(outputBuffer, CL_TRUE, origin, region, 0, 0, outputPixels.data());
    }
    catch (const cl::Error& e)
    {
        std::cerr << "Exception Caught: " << e.what() << std::endl;
    }
   
    image.write(output, rgb_from_rgba(outputPixels, image.w, image.h));
    return 0;
}