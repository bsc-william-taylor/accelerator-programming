
#pragma once

#pragma warning (disable: 4996)
#pragma warning (disable: 4018)
#pragma warning (disable: 4244)

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <functional>

enum Channels { R, G, B, A };

inline std::vector<std::uint8_t> rgb_to_rgba(std::vector<std::uint8_t>& rgb, int width, int height)
{
    std::vector<std::uint8_t> rgba(width * height * 4);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            auto rgb_index = (y * width + x) * 3;
            auto rgba_index = (y * width + x) * 4;

            rgba[rgba_index + R] = rgb[rgb_index + R];
            rgba[rgba_index + G] = rgb[rgb_index + G];
            rgba[rgba_index + B] = rgb[rgb_index + B];
            rgba[rgba_index + A] = 255;
        }
    }

    return rgba;
}

inline std::vector<std::uint8_t> rgb_from_rgba(std::vector<std::uint8_t>& rgba, int width, int height)
{
    std::vector<std::uint8_t> rgb(width*height * 3);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            auto rgb_index = (y * width + x) * 3;
            auto rgba_index = (y * width + x) * 4;

            rgb[rgb_index + R] = rgba[rgba_index + R];
            rgb[rgb_index + G] = rgba[rgba_index + G];
            rgb[rgb_index + B] = rgba[rgba_index + B];
        }
    }

    return rgb;
}

template<typename L, typename T>
L clamp(T value)
{
    const auto minimum = std::numeric_limits<L>::min();
    const auto maximum = std::numeric_limits<L>::max();

    return static_cast<L>(value < minimum ? minimum : value > maximum ? maximum : value);
}

template<typename T>
T clamp(T value, T min, T max)
{
    return value < min ? min : value >= max ? max - 1 : value;
}

#ifndef IGNORE_CL

#include <Windows.h>
#include <cl/cl.hpp>
#include <CL/cl_gl_ext.h>

inline cl::Platform findPlatform()
{
    return cl::Platform::getDefault();
}

inline cl::Context createContext(cl::Platform& platform, cl::Device& device, bool shared)
{
    const auto hGLRC = wglGetCurrentContext();
    const auto hDC = wglGetCurrentDC();

    cl_context_properties properties[] =
    {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform(),
        CL_GL_CONTEXT_KHR, (cl_context_properties)hGLRC,
        CL_WGL_HDC_KHR, (cl_context_properties)hDC,
        0
    };

    return cl::Context(device, shared ? properties : nullptr);
}

inline cl::Device findDevice(cl::Platform& platform)
{
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    return devices.front();
}

inline cl::Program createKernel(cl::Context& context, cl::Device& device, const char* filename)
{
    std::ifstream file(filename);
    std::stringstream ss;
    std::string src;

    while (std::getline(file, src))
    {
        ss << src << std::endl;
    }
    
    src = ss.str();
    
    cl::Program::Sources kernel(1, std::make_pair(src.c_str(), src.size()));
    cl::Program program(context, kernel);

    try
    {
        std::stringstream options;
        options << "-Dalpha=1.5 ";
        options << "-Dgamma=0.0 ";
        options << "-Dbeta=-0.5 ";
        options << "-cl-unsafe-math-optimizations ";
        options << "-cl-mad-enable ";
        program.build(options.str().c_str());
    }
    catch (const cl::Error& e)
    {
        std::cerr 
            << "cl::Program Build Error "
            << e.what()
            << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) 
            << std::endl;
        std::cin.get();
    }

    return program;
}

namespace cl {
    template<int L>
    cl::size_t<L> new_size_t(const std::vector<int>& numbers) {
        cl::size_t<L> sz;
        for (int i = 0; i < L; i++)
            sz[i] = numbers[i];
        return sz;
    }
}

#endif
