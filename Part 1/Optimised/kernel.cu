
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>
#include <vector>
#include <limits>

struct rgb
{
    unsigned char r, g, b;
};

// Setting Constants
const auto MaxIterations = std::numeric_limits<unsigned char>::max();
const auto FilenameOut = std::string("output.ppm");
const auto DefaultHeight = 4096;
const auto DefaultWidth = 4096;
const auto cx = -0.6, cy = 0.0;

// Moved out of function 
const unsigned char numberShades = 16;
const rgb mapping[numberShades] =
{
    { 66, 30, 15 },{ 25,7,26 },{ 9,1,47 },{ 4,4,73 },{ 0,7,100 },
    { 12, 44, 138 },{ 24,82,177 },{ 57,125,209 },{ 134,181,229 },{ 211,236,248 },
    { 241, 233, 191 },{ 248,201,95 },{ 255,170,0 },{ 204,128,0 },{ 153,87,0 },
    { 106, 52, 3 }
};

void writeOutput(std::vector<rgb*>& rows, const int width, const int height)
{
    const auto file = fopen(FilenameOut.c_str(), "w");
    fprintf(file, "P6\n%d %d\n255\n", width, height);

    for (auto i = height - 1; i >= 0; i--)
    {
        fwrite(rows[i], 1, width * sizeof(rgb), file);
    }
       
    fclose(file);
}

void allocateArrays(std::vector<rgb>& img, std::vector<rgb*>& rows, const int width, const int height)
{
    rows[0] = img.data();

    for (auto i = 1; i < height; ++i)
    {
        rows[i] = rows[i - 1] + width;
    }
}

void map_colour(rgb * const px)
{
    if (px->r == MaxIterations || px->r == 0) 
    {
        px->r = 0; 
        px->g = 0; 
        px->b = 0;
    }
    else 
    {
        *px = mapping[px->r % numberShades];
    }
}

void calculateMandel(std::vector<rgb*>& rows, const int width, const int height, const double scale)
{
    for (auto i = 0; i < height; i++) 
    {
        const auto y = (i - height / 2) * scale + cy;
        auto* px = rows[i];

        for (auto j = 0; j < width; j++, px++) 
        {
            const auto x = (j - width / 2) * scale + cx;
         
            unsigned char iter = 0;
      
            auto zx = hypot(x - .25, y), zy = 0.0, zx2 = 0.0, zy2 = 0.0;

            if (x < zx - 2 * zx * zx + .25) 
            {
                iter = MaxIterations;
            }

            if ((x + 1)*(x + 1) + y * y < 1 / 16)
            {
             iter = MaxIterations;
            }

            zx = zy = zx2 = zy2 = 0;
            do 
            {
                zy = 2 * zx * zy + y;
                zx = zx2 - zy2 + x;
                zx2 = zx * zx;
                zy2 = zy * zy;
            } while (iter++ < MaxIterations && zx2 + zy2 < 4);

            px->r = iter;
            px->g = iter;
            px->b = iter;
        }
    }

    for (auto i = 0; i < height; ++i) 
    {
        auto* px = rows[i];

        for (auto j = 0; j < width; ++j, ++px) 
        {
            map_colour(px);
        }
    }
}

int main(int argc, char *argv[])
{
    const auto height = argc > 2 ? atoi(argv[2]) : DefaultHeight;
    const auto width = argc > 1 ? atoi(argv[1]) : DefaultWidth;
    const auto scale = 1.0 / (width / 4);

    std::vector<rgb> image(width * height);
    std::vector<rgb*> rows(height);

    allocateArrays(image, rows, width, height);
    calculateMandel(rows, width, height, scale);
    writeOutput(rows, width, height);
    return 0;
}