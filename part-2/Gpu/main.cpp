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
    std::string src = kernel("kernels.cl"), name, version;
    std::vector<float> a(N, 1), b(N, 2), c(N);
    std::vector<cl::Device> devices;

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
        // Placeholder kernel launch
        cl::Kernel add(program, "add");
        cl::Buffer A(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a.size() * sizeof(float), a.data());
        cl::Buffer B(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b.size() * sizeof(float), b.data());
        cl::Buffer C(context, CL_MEM_READ_WRITE, c.size() * sizeof(float));

        add.setArg(0, static_cast<cl_ulong>(N));
        add.setArg(1, A);
        add.setArg(2, B);
        add.setArg(3, C);

        queue.enqueueNDRangeKernel(add, cl::NullRange, N, cl::NullRange);
        queue.enqueueReadBuffer(C, CL_TRUE, 0, c.size() * sizeof(float), c.data());
    }
    catch (const cl::Error& e)
    {
        std::cerr << "Exception Caught: " << e.what() << std::endl;
    }

   return EXIT_SUCCESS;
}