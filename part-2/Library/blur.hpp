
#pragma once

#include <vector>

const int PI = 3.1415926535897;

std::vector<float> filter(const int radius, const float weight = 1.0f)
{
    std::vector<float> matrix;
    matrix.reserve(radius*radius);

    float stdv = weight, s = 2.0 * stdv * stdv;
    float sum = 0.0;

    const int size = floor(radius / 2.0);

    for (int x = -size; x <= size; x++)
    {
        for (int y = -size; y <= size; y++)
        {
            float r = sqrt(x*x + y*y);
            auto value = (exp(-(r*r) / s)) * 1.0 / (sqrt(2.0 * PI) * stdv);
            sum += value;
            matrix.push_back(value);
        }
    }

    for (int i = 0; i < matrix.size(); i++)
        matrix[i] /= sum;

    return matrix;
}

