
#pragma once

#include <cuda_runtime.h>
#include <algorithm>

namespace cuda
{
    struct launchInfo
    {
        int blocks, threads, width, height;
    };

    struct workload
    {
        int offset, size, stride;
    };

    template<typename K, typename... Args>
    cudaError_t start(K kernel, launchInfo& launch, Args&&... args)
    {
        kernel<<<launch.blocks, launch.threads>>>(launch, std::forward<Args>(args)...);
        return cudaDeviceSynchronize();
    }

    inline __device__ workload allocateWork(cuda::launchInfo& info, int block, int thread)
    {
        auto blockSizeY = info.height / info.blocks;
        auto blockSize = info.width * blockSizeY;
        auto threadSize = blockSize / info.threads;

        auto offset = block * blockSize + thread * threadSize;
        auto stride = block * blockSizeY + thread * blockSizeY / info.threads;

        return { offset, offset + threadSize, stride };
    }
}