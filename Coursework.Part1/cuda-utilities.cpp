
#include "cuda-utilities.h"
#include <iostream>

using std::function;
using std::string;
using std::vector;

template<typename T>
void repeat(unsigned int count, T functor)
{
    for(auto i{0}; i < count; ++i)
    {
        functor(i);
    }
}

void cuda::debugInfo(const cudaDeviceProp& device)
{
    std::cout << "CUDA: " << device.major << "." << device.minor << std::endl;
    std::cout << "GPU: " << device.name << ", " << device.clockRate / 1000000.0 << "GHz" << std::endl;
    std::cout << "MultiProcessors: " << device.multiProcessorCount << std::endl;
    std::cout << "Global Memory" << device.totalGlobalMem * 1e-9 << "GBs" << std::endl;
}

cudaError cuda::errorCheck(cudaErrorHandler functor)
{
    auto errorCode = cudaGetLastError();

    if(errorCode != 0)
    {
        functor(errorCode, cudaGetErrorString(errorCode));
    }

    return errorCode;
}

cudaDeviceProp cuda::findDevice(cudaDeviceFilter functor)
{
    int count;
    cudaGetDeviceCount(&count); 

    if(count <= 0)
    {
        throw std::exception("CUDA Device Not Available");
    }

    vector<cudaDeviceProp> props(count);
    repeat(count, [&](int i){ cudaGetDeviceProperties(&props[i], i);});
    return functor(props);
}