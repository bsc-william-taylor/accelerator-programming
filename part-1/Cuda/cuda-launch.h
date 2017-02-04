
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

namespace cuda
{
    struct launchInfo
    {
        int blocks, threads, size;
    };

    launchInfo optimumLaunch(void* kernel, int datasetSize);

    template<typename K, typename... Args>
    cudaError_t start(K kernel, launchInfo& launch, Args&&... args)
    {
        kernel<<<launch.blocks, launch.threads>>>(launch, std::forward<Args>(args)...);
        return cudaDeviceSynchronize();
    }
}