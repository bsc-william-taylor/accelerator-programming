
#include "ppm.h"

ppm::file::file(const std::string& filename) :
    magicNumber(""), width(0), height(0), maxValue(255), okay(false)
{
    std::ifstream fs(filename, std::ios_base::binary);

    if (fs.is_open())
    {
        fs >> magicNumber;
        fs >> width;
        fs >> height;
        fs >> maxValue;
        fs.get();

        pixels.resize(size());
        fs.read(pixels.data(), size());
        okay = true;
    }

}

bool ppm::file::is_open()
{
    return okay;
}

bool ppm::file::compare(section field, file& right)
{
    switch (field)
    {
        case section::RawData: 
            return memcmp(pixels.data(), right.pixels.data(), size()) == 0;
        case section::MagicNum: 
            return magicNumber == right.magicNumber;
        case section::MaxColVal:
             return maxValue == right.maxValue;
        case section::Height: 
            return height == right.height;
        case section::Width: 
            return width == right.width;
        default: 
            return "";
    }
}


int ppm::file::size()
{
    return width * height * 3;
}