
#include "cuda-launch.h"
#include <stdexcept>

// http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
int P2(int v)
{
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    ++v;
    return v;
}

cuda::launchInfo cuda::optimumLaunch(void* kernel, int dataLength)
{
    auto minGridSize = 0, blockSize = 0;
    auto cudaError = cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, dataLength);

    if(cudaError != 0)
    {
        throw std::runtime_error("cudaOccupancyMaxPotentialBlockSize failed");
    }
    
    const auto gridSize = (dataLength + blockSize - 1) / blockSize;
    const auto thread = P2(sqrt(blockSize));
    const auto block = P2(sqrt(gridSize));
 
    return { dim3(block, block), dim3(thread, thread), static_cast<int>(sqrt(dataLength)) };
}
