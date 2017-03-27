
#pragma once

#pragma warning (disable: 4996)
#pragma warning (disable: 4018)
#pragma warning (disable: 4244)

#include <Windows.h>
#include <cl/cl.hpp>
#include <CL/cl_gl_ext.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <functional>

namespace cl {
    template<int L>
    cl::size_t<L> new_size_t(const std::vector<int>& numbers) {
        cl::size_t<L> sz;
        for (int i = 0; i < L; i++)
            sz[i] = numbers[i];
        return sz;
    }

    inline cl::Context getContext(cl::Platform& platform, cl::Device& device, bool shared)
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

    inline cl::Device getDevice(cl::Platform& platform)
    {
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        return devices.front();
    }

    inline cl::Program getKernel(cl::Context& context, cl::Device& device, const char* filename, std::function<std::string()> options)
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
            program.build(options().c_str());
        }
        catch (const cl::Error& e)
        {
            std::cerr
                << "cl::Program Build Error "
                << e.what()
                << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)
                << std::endl;
        }

        return program;
    }
}
