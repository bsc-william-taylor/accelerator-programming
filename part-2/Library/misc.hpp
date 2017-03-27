
#pragma once

#include <limits>
#include <vector>

enum Channels
{
    R, G, B, A
};

template<typename T>
inline std::vector<T> toRGBA(std::vector<T>& rgb, int width, int height)
{
    std::vector<T> rgba(width * height * 4);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            auto rgb_index = (y * width + x) * 3;
            auto rgba_index = (y * width + x) * 4;

            rgba[rgba_index + R] = rgb[rgb_index + R];
            rgba[rgba_index + G] = rgb[rgb_index + G];
            rgba[rgba_index + B] = rgb[rgb_index + B];
            rgba[rgba_index + A] = 255;
        }
    }

    return rgba;
}

template<typename T>
inline std::vector<T> toRGB(std::vector<T>& rgba, int width, int height)
{
    std::vector<T> rgb(width*height * 3);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            auto rgb_index = (y * width + x) * 3;
            auto rgba_index = (y * width + x) * 4;

            rgb[rgb_index + R] = rgba[rgba_index + R];
            rgb[rgb_index + G] = rgba[rgba_index + G];
            rgb[rgb_index + B] = rgba[rgba_index + B];
        }
    }

    return rgb;
}

template<typename L, typename T>
L clamp(T value)
{
    const auto minimum = std::numeric_limits<L>::min();
    const auto maximum = std::numeric_limits<L>::max();

    return static_cast<L>(value < minimum ? minimum : value > maximum ? maximum : value);
}

template<typename T>
T clamp(T value, T min, T max)
{
    return value < min ? min : value >= max ? max - 1 : value;
}