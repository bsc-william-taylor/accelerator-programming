
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <functional>
#include <vector>

#include "cuda-launch.h"
#include "cuda-memory.h"

namespace cuda
{
    using cudaDeviceList = std::vector<cudaDeviceProp>;
    using cudaErrorHandler = std::function<void(int, const char*)>;
    using cudaDeviceFilter = std::function<cudaDeviceProp(cudaDeviceList&)>;

    cudaDeviceProp findDevice(cudaDeviceFilter functor = [](cudaDeviceList& list){ return list.front(); });
    cudaError errorCheck(cudaErrorHandler functor = [](int, const char*){});

    template<typename T>
    cudaError move(memory<T>& mem, T destination)
    {
        auto err = errorCheck();

        if(err == 0)
        {
            mem.transfer(destination);
        }
        
        return err == 0 ? err : errorCheck();
    }
}