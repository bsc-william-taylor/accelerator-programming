
#pragma once

#include <vector>

const float PI = 3.14159265359f;

inline std::vector<float> gaussianFilter2D(const int radii)
{
    const int radius = (int)ceil(radii * 2.57);
    std::vector<float> kernel;
    kernel.reserve(radius * 2 + 1 * radius * 2 + 1);
    auto sum = 0.0f;

    for (int row = -radius; row <= radius; row++)
    {
        for (int col = -radius; col <= radius; col++)
        {
            const auto deviation = 2 * radii * radii;
            const auto distance = sqrt(col*col + row*row);
            const auto value = (float)(exp(-(distance*distance) / deviation)) / (PI * deviation);
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
    const auto radius = (int)ceil(radii * 2.57);
    std::vector<float> kernel;
    kernel.reserve(radius * 2 + 1);

    auto sum = 0.0f;
    for (int col = -radius; col <= radius; col++)
    {
        const auto deviation = 2 * radii * radii;
        const auto distance = sqrt(col*col);
        const auto value = (float)(exp(-(pow(distance, 2.0f)) / deviation)) / (PI * deviation);
        kernel.push_back(value);
        sum += value;
    }

    for (auto& v : kernel)
    {
        v /= sum;
    }

    return kernel;
}
