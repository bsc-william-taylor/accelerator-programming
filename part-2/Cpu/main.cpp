
#include "../library/benchmark.hpp"
#include "../Library/utilities.hpp"
#include "../library/ppm.hpp"

const auto set_colour = [](auto pixels, auto offset, auto r, auto g, auto b)
{
    pixels[offset + R] = clamp<std::uint8_t>(r);
    pixels[offset + G] = clamp<std::uint8_t>(g);
    pixels[offset + B] = clamp<std::uint8_t>(b);
};

void pixel_average(std::uint8_t *out, std::uint8_t *in, int x, int y, int radius, int w, int h, int channels)
{
    auto samples = pow((radius * 2 - 1), 2);
    auto r = 0.0f, g = 0.0f, b = 0.0f;

    for (auto j = y - radius + 1; j < y + radius; ++j)
    {
        for (auto i = x - radius + 1; i < x + radius; ++i)
        {
            const auto x = clamp(i, 0, w), y = clamp(j, 0, h);
            const auto offset = (y * w + x) * channels;

            r += in[offset + R];
            g += in[offset + G];
            b += in[offset + B];
        }
    }

    const auto offset = (y * w + x) * channels;
    set_colour(out, offset, r / samples, g / samples, b / samples);
}

void blur(std::uint8_t *out, std::uint8_t *in, int radius, int w, int h, int channels)
{
    for (auto y = 0; y < h; ++y)
    {
        for (auto x = 0; x < w; ++x)
        {
            pixel_average(out, in, x, y, radius, w, h, channels);
        }
    }
}

void add_weighted(std::uint8_t *out, std::uint8_t *in1, std::uint8_t *in2, int w, int h, int nchannels)
{
    const auto alpha = 1.5, beta = -0.5, gamma = 0.0;
    const auto weight = [&](auto colour, auto offset)
    {
        return in1[offset + colour] * alpha + in2[offset + colour] * beta + gamma;
    };

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            auto offset = (y * w + x) * nchannels;
            auto r = weight(R, offset);
            auto g = weight(G, offset);
            auto b = weight(B, offset);

            set_colour(out, offset, r, g, b);
        }
    }
}

void unsharp_mask(std::uint8_t *out, std::uint8_t *in, int radius, int w, int h, int channels)
{
    const auto size = w * h * channels;
    std::vector<std::uint8_t> blur1(size), blur2(size), blur3(size);

    blur(blur1.data(), in, radius, w, h, channels);
    blur(blur2.data(), blur1.data(), radius, w, h, channels);
    blur(out, blur2.data(), radius, w, h, channels);

    //add_weighted(out, in, blur3.data(), w, h, channels);
}

int main(int argc, char *argv[])
{
    auto radius = std::atoi(arg(argc, argv, 3, "3"));
    auto output = arg(argc, argv, 2, "./out.ppm");
    auto input = arg(argc, argv, 1, "../library/lena.ppm");

    ppm image(input);

    std::vector<std::uint8_t> data_sharp(image.w * image.h * image.nchannels);
    unsharp_mask(data_sharp.data(), image.data.data(), radius, image.w, image.h, image.nchannels);
    image.write(output, data_sharp);
    
    return 0;
}

