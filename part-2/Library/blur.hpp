
#pragma once

#include <vector>

inline float gaussian(float x, float mu, float sigma)
{
    return std::exp(-(((x - mu) / (sigma)) * ((x - mu) / (sigma))) / 2.0f);
}

const float PI = 3.14159265359f;

inline std::vector<float> gaussianFilter(int givenRadius)
{
    const int radius = ceil(givenRadius * 2.57);
    auto sum = 0.0f;
    std::vector<float> kernel;
    kernel.reserve(radius*2+1 * radius*2+1);
    for (int row = -radius; row <= radius; row++) {
        for (int col = -radius; col <= radius; col++) {
            float s = 2 * givenRadius * givenRadius;
            float r = sqrt(col*col + row*row);
            float value = (exp(-(r*r) / s)) / (PI * s);
            kernel.push_back(value);
            sum += value;
        }
    }

    for (auto& v : kernel) {
        v /= sum;
    }

    return kernel;
}
