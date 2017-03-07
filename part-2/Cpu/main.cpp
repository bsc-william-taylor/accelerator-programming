#pragma once

#include "ppm.hpp"
#include <climits>
#include <chrono>

// Averages the nsamples pixels within blur_radius of (x,y). Pixels which
// would be outside the image, replicate the value at the image border.
void pixel_average(std::uint8_t *out, std::uint8_t *in, int x, int y, int blur_radius, int w, int h, int nchannels)
{
    auto red_total = 0.0f, green_total = 0.0f, blue_total = 0.0f;
    auto nsamples = (blur_radius * 2 - 1) * (blur_radius * 2 - 1);

    for (int j = y - blur_radius + 1; j < y + blur_radius; ++j) {
        for (int i = x - blur_radius + 1; i < x + blur_radius; ++i) {
            const unsigned r_i = i < 0 ? 0 : i >= w ? w - 1 : i;
            const unsigned r_j = j < 0 ? 0 : j >= h ? h - 1 : j;
            unsigned byte_offset = (r_j*w + r_i)*nchannels;
            red_total += in[byte_offset + 0];
            green_total += in[byte_offset + 1];
            blue_total += in[byte_offset + 2];
        }
    }

    auto byte_offset = (y * w + x) * nchannels;
    out[byte_offset + 0] = red_total / nsamples;
    out[byte_offset + 1] = green_total / nsamples;
    out[byte_offset + 2] = blue_total / nsamples;
}

void blur(std::uint8_t *out, std::uint8_t *in, int blur_radius, int w, int h, int nchannels)
{
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            pixel_average(out, in, x, y, blur_radius, w, h, nchannels);
        }
    }
}

// Calculates the weighted sum of two arrays, in1 and in2 according
// to the formula: out(I) = saturate(in1(I)*alpha + in2(I)*beta + gamma)
template <typename T>
void add_weighted(std::uint8_t *out, std::uint8_t *in1, T alpha, std::uint8_t *in2, T  beta, T gamma, const unsigned w, const unsigned h, const unsigned nchannels)
{
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            unsigned byte_offset = (y*w + x)*nchannels;

            T tmp = in1[byte_offset + 0] * alpha + in2[byte_offset + 0] * beta + gamma;
            out[byte_offset + 0] = tmp < 0 ? 0 : tmp > UCHAR_MAX ? UCHAR_MAX : tmp;

            tmp = in1[byte_offset + 1] * alpha + in2[byte_offset + 1] * beta + gamma;
            out[byte_offset + 1] = tmp < 0 ? 0 : tmp > UCHAR_MAX ? UCHAR_MAX : tmp;

            tmp = in1[byte_offset + 2] * alpha + in2[byte_offset + 2] * beta + gamma;
            out[byte_offset + 2] = tmp < 0 ? 0 : tmp > UCHAR_MAX ? UCHAR_MAX : tmp;
        }
    }
}

void unsharp_mask(std::uint8_t *out, std::uint8_t *in, int radius, int w, int h, int nchannels)
{
    const auto length = w * h * nchannels;
    std::vector<std::uint8_t> blur1(length), blur2(length), blur3(length);

    blur(blur1.data(), in, radius, w, h, nchannels);
    blur(blur2.data(), blur1.data(), radius, w, h, nchannels);
    blur(blur3.data(), blur2.data(), radius, w, h, nchannels);

    add_weighted(out, in, 1.5f, blur3.data(), -0.5f, 0.0f, w, h, nchannels);
}

// Apply an unsharp mask to the 24-bit PPM loaded from the file path of
// the first input argument; then write the sharpened output to the file path
// of the second argument. The third argument provides the blur radius.
int main(int argc, char *argv[])
{
    const auto input = argc > 1 ? argv[1] : "./ghost-town-8k.ppm";
    const auto output = argc > 2 ? argv[2] : "./out.ppm";
    const auto radius = argc > 3 ? std::atoi(argv[3]) : 5;

    std::vector<unsigned char> data_in, data_sharp;

    ppm img;
    img.read(input, data_in);
    data_sharp.resize(img.w * img.h * img.nchannels);

    auto start = std::chrono::steady_clock::now();
    unsharp_mask(data_sharp.data(), data_in.data(), radius, img.w, img.h, img.nchannels);
    auto stop = std::chrono::steady_clock::now();

    std::cout << std::chrono::duration<double>(stop - start).count() << " seconds.\n";
    img.write(output, data_sharp);
    return 0;
}

