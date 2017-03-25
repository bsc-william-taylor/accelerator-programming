
#pragma once

#include <vector>

double gaussian(double x, double mu, double sigma) {
    return std::exp(-(((x - mu) / (sigma))*((x - mu) / (sigma))) / 2.0);
}

typedef std::vector<double> kernel_row;
typedef std::vector<kernel_row> kernel_type;

std::vector<float> gaussianFilter(const int radius, const float weight = 1.0) 
{
    double sigma = radius / 2.;
    kernel_type kernel2d(2 * radius + 1, kernel_row(2 * radius + 1));
    double sum = 0;
    // compute values
    for (int row = 0; row < kernel2d.size(); row++)
        for (int col = 0; col < kernel2d[row].size(); col++) {
            double x = gaussian(row, radius, sigma) * gaussian(col, radius, sigma);
            kernel2d[row][col] = x;
            sum += x;
        }
    // normalize
    for (int row = 0; row < kernel2d.size(); row++)
        for (int col = 0; col < kernel2d[row].size(); col++)
            kernel2d[row][col] /= sum;

    std::vector<float> kernel1d;
    for (int row = 0; row < kernel2d.size(); row++)
        for (int col = 0; col < kernel2d[row].size(); col++)
            kernel1d.push_back(kernel2d[row][col]);
    return kernel1d;
}
