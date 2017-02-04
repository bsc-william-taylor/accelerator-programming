
#include "cuda-utilities.h"

using std::function;
using std::string;
using std::vector;

template<typename T>
void repeat(unsigned int count, T functor)
{
    for (auto i{ 0u }; i < count; ++i)
    {
        functor(i);
    }
}

cudaError cuda::errorCheck(cudaErrorHandler functor)
{
    auto errorCode = cudaGetLastError();

    if (errorCode != 0)
    {
        functor(errorCode, cudaGetErrorString(errorCode));
    }

    return errorCode;
}

cudaDeviceProp cuda::findDevice(cudaDeviceFilter functor)
{
    int count;
    cudaGetDeviceCount(&count);

    if (count <= 0)
    {
        throw std::runtime_error("CUDA Device Not Available (cudaGetDeviceCount <= 0)");
    }

    vector<cudaDeviceProp> props(count);
    repeat(count, [&](int i) { cudaGetDeviceProperties(&props[i], i); });
    return functor(props);
}