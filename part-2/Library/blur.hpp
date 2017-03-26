
#pragma once

#include <vector>

inline float gaussian(float x, float mu, float sigma)
{
    return std::exp(-(((x - mu) / (sigma)) * ((x - mu) / (sigma))) / 2.0f);
}

inline std::vector<float> gaussianFilter(const int radius)
{
    auto sigma = radius / 2.0f, sum = 0.0f;
    auto size = 2 * radius + 1;

    std::vector<float> kernel;
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            float x = gaussian(row, radius, sigma) * gaussian(col, radius, sigma);
            kernel.push_back(x);
            sum += x;
        }
    }

    for (auto& v : kernel) {
        v /= sum;
    }

    return kernel;
}
