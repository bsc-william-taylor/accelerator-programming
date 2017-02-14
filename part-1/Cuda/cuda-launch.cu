
#include "cuda-launch.h"
#include <stdexcept>

cuda::launchInfo cuda::optimumLaunch(void* kernel, int width, int height, int dataLength)
{
    auto minGridSize = 0, blockSize = 0;
    auto cudaError = cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0);
  
    if(cudaError != 0)
    {
        throw std::runtime_error("cudaOccupancyMaxPotentialBlockSize failed");
    }

    const auto blocks = static_cast<int>(pow(2, ceil(log(sqrt(blockSize)) / log(2))));
    const auto grid = static_cast<int>((sqrt(dataLength) + blocks - 1) / blocks);

    return { dim3(grid, grid), dim3(blocks, blocks), width, height };
}