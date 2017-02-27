#pragma warning (disable: 4996)

#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

const size_t N = 100000;

std::string kernel(const std::string& filename)
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

int main(int argc, const char * argv[]) 
{
    std::string source = kernel("helloworld.cl");
    std::vector<double> a(N, 1), b(N, 2), c(N);
    std::vector<cl::Platform> platform;
    std::vector<cl::Device> device;

    cl::Platform::get(&platform);
    cl::Context context;

    for (auto p = platform.begin(); device.empty() && p != platform.end(); p++) 
    {
        std::vector<cl::Device> pldev;
        p->getDevices(CL_DEVICE_TYPE_ALL, &pldev);

        for (auto d = pldev.begin(); device.empty() && d != pldev.end(); d++) 
        {
            if (!d->getInfo<CL_DEVICE_AVAILABLE>())
            {
                continue;
            }

            std::string ext = d->getInfo<CL_DEVICE_EXTENSIONS>();

            if (ext.find("cl_khr_fp64") == std::string::npos && ext.find("cl_amd_fp64") == std::string::npos)
            {
                continue;
            }

            device.push_back(*d);
            context = cl::Context(device);
        }
    }

    cl::CommandQueue queue(context, device.front());
    cl::Program program(context, cl::Program::Sources(1, std::make_pair(source.c_str(), source.length())));

    try
    {
        program.build(device);
    }
    catch (const cl::Error&)
    {
        std::cerr
            << "OpenCL compilation error" << std::endl
            << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device[0])
            << std::endl;
        std::cin.get();
        return 1;
    }

    try
    {
        cl::Kernel add(program, "add");
        cl::Buffer A(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a.size() * sizeof(double), a.data());
        cl::Buffer B(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b.size() * sizeof(double), b.data());
        cl::Buffer C(context, CL_MEM_READ_WRITE, c.size() * sizeof(double));

        add.setArg(0, static_cast<cl_ulong>(N));
        add.setArg(1, A);
        add.setArg(2, B);
        add.setArg(3, C);

        queue.enqueueNDRangeKernel(add, cl::NullRange, N, cl::NullRange);
        queue.enqueueReadBuffer(C, CL_TRUE, 0, c.size() * sizeof(double), c.data());
    }
    catch (const cl::Error& err)
    {
        std::cerr
            << "OpenCL error" << std::endl
            << err.what()
            << std::endl;
        std::cin.get();
        return 1;
    }

    std::cout << "Success = " << (c[0] == 3 ? "true" : "false") << std::endl;
    std::cin.get();
    return 0;
}