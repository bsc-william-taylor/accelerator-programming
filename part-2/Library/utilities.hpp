
#pragma once
#pragma warning (disable: 4996)

#include <cl/cl.hpp>
#include <string>
#include <fstream>
#include <sstream>

enum Channels { R, G, B };

auto rgb_to_rgba(std::vector<std::uint8_t>& rgb, int width, int height)
{
    std::vector<std::uint8_t> rgba(width*height * 4);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            auto rgb_index = (y * width + x) * 3;
            auto rgba_index = (y * width + x) * 4;

            rgba[rgba_index + 0] = rgb[rgb_index + 0];
            rgba[rgba_index + 1] = rgb[rgb_index + 1];
            rgba[rgba_index + 2] = rgb[rgb_index + 2];
            rgba[rgba_index + 3] = 255;
        }
    }

    return rgba;
}

auto rgb_from_rgba(std::vector<std::uint8_t>& rgba, int width, int height)
{
    std::vector<std::uint8_t> rgb(width*height * 3);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            auto rgb_index = (y * width + x) * 3;
            auto rgba_index = (y * width + x) * 4;

            rgb[rgb_index + 0] = rgba[rgba_index + 0];
            rgb[rgb_index + 1] = rgba[rgba_index + 1];
            rgb[rgb_index + 2] = rgba[rgba_index + 2];
        }
    }

    return rgb;
}

auto kernel(const std::string& filename)
{
    std::ifstream file(filename);
    std::stringstream ss;
    std::string str;
    while (std::getline(file, str))
    {
        ss << str << std::endl;
    }
    return ss.str();
}

template<typename L, typename T>
L clamp(T value)
{
    const auto min = std::numeric_limits<L>::min();
    const auto max = std::numeric_limits<L>::max();

    return static_cast<L>(value < min ? min : value > max ? max : value);
}

namespace cl {
    template<int L>
    cl::size_t<L> new_size_t(std::vector<int> numbers) {
        cl::size_t<L> sz;
        for (int i = 0; i < L; i++)
            sz[i] = numbers[i];
        return sz;
    }
}

template<typename T>
T clamp(T value, T min, T max)
{
    return value < min ? min : value >= max ? max - 1 : value;
}

const auto arg = [](auto argc, auto argv, auto index, auto value)
{
    return argc > index ? argv[index] : value;
};