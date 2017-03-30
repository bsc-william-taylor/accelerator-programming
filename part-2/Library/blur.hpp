
#pragma once

#include <vector>

const float PI = 3.14159265359f;

inline std::vector<float> gaussianFilter2D(const int radii)
{
    auto radius = (int)ceil(radii * 2.57);
    auto deviation = 2 * radii * radii;
    auto sum = 0.0f;

    std::vector<float> kernel;
    kernel.reserve((int)pow(radius * 2 + 1, 2));

    for (int row = -radius; row <= radius; row++)
    {
        for (int col = -radius; col <= radius; col++)
        {
            const auto dist = sqrt(pow(col, 2) + pow(row, 2));
            const auto value = (float)(exp(-(pow(dist, 2.0f)) / deviation)) / (PI * deviation);
            kernel.push_back(value);
            sum += value;
        }
    }

    for (auto& v : kernel)
    {
        v /= sum;
    }

    return kernel;
}

inline std::vector<float> gaussianFilter1D(const int radii)
{
    auto radius = (int)ceil(radii * 2.57);
    auto deviation = 2 * radii * radii;
    auto sum = 0.0f;

    std::vector<float> kernel(radius * 2 + 1);
  
    for (auto col = -radius; col <= radius; col++)
    {
        const auto dist = sqrt(col*col);
        const auto value = (float)(exp(-(pow(dist, 2.0f)) / deviation)) / (PI * deviation);
        kernel[col+radius] = value;
        sum += value;
    }

    for (auto& v : kernel)
    {
        v /= sum;
    }

    return kernel;
}
