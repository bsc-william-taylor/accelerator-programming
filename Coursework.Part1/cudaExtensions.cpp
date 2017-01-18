
#include "cudaExtensions.h"

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

cudaError cuda::errorCheck(cudaErrorHandler functor)
{
    auto errorCode = cudaGetLastError();

    if(errorCode != 0)
    {
        functor(errorCode, cudaGetErrorString(errorCode));
    }

    return errorCode;
}

cudaDeviceProp cuda::chooseDevice(cudaDeviceFilter functor)
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