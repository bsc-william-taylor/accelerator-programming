
#include "cuda-launch.h"
#include <stdexcept>

cuda::launchInfo cuda::optimumLaunch(void* kernel, int dataLength)
{
    auto minGridSize = 0, blockSize = 0;
    auto cudaError = cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, dataLength);

    if(cudaError != 0)
    {
        throw std::runtime_error("cudaOccupancyMaxPotentialBlockSize failed");
    }
    
    const auto gridSize = (dataLength + blockSize - 1) / blockSize;

    return { gridSize, blockSize, (int)sqrt(dataLength) };
}
