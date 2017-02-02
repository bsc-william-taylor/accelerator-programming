
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <functional>
#include <iostream>
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

    template<unsigned times, typename Functor, typename... Args>
    void benchmark(Functor&& method, Args&&... args)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for(auto i = 0u; i < times; ++i)
        {
            cudaEventRecord(start);
            method(std::forward(args)...);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            std::cout << "ms: "  << milliseconds << std::endl;
        }
         
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

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