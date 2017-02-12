
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <functional>
#include <iostream>
#include <fstream>
#include <vector>

#include "../csv-file.h"
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
        fs::csv<2> table("kernel.csv", "Kernel Results");
        table.append_row("ID, Time (ms)");

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        auto milliseconds = 0.0f;
        auto total = 0.0;
        
        for(auto i = 1u; i <= times; ++i)
        {
            cudaEventRecord(start);
            method(std::forward(args)...);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
           
            table.append(i, ",", milliseconds, "\n");
            total += milliseconds;
        }
         
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        table.append_row("Total (ms), Average (ms)");
        table.append("total", ",", total / times);
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