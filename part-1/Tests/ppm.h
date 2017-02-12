
#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

namespace ppm 
{
    enum class section
    {
        MagicNum,
        Width,
        Height,
        MaxColVal,
        RawData
    };

    class file
    {
        std::string magicNumber;
        int width, height, maxValue;
        std::vector<char> pixels;
        bool okay;
    public:
        explicit file(const std::string& filename);

        bool compare(section field, file& right);
        bool is_open();

        int size();
    };
}