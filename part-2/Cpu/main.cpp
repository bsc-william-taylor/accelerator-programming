
#include "../Library/benchmark.hpp"
#include "../Library/ppm.hpp"

#include <climits>

// Calculates the weighted sum of two arrays, in1 and in2 according
// to the formula: out(I) = saturate(in1(I)*alpha + in2(I)*beta + gamma)
template <typename T>
void add_weighted(unsigned char *out,
    const unsigned char *in1, const T alpha,
    const unsigned char *in2, const T  beta, const T gamma,
    const unsigned w, const unsigned h, const unsigned nchannels)
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


// Averages the nsamples pixels within blur_radius of (x,y). Pixels which
// would be outside the image, replicate the value at the image border.
void pixel_average(unsigned char *out,
    const unsigned char *in,
    const int x, const int y, const int blur_radius,
    const unsigned w, const unsigned h, const unsigned nchannels)
{
    float red_total = 0, green_total = 0, blue_total = 0;
    const unsigned nsamples = (blur_radius * 2 - 1) * (blur_radius * 2 - 1);
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

    unsigned byte_offset = (y*w + x)*nchannels;
    out[byte_offset + 0] = red_total / nsamples;
    out[byte_offset + 1] = green_total / nsamples;
    out[byte_offset + 2] = blue_total / nsamples;
}

void blur(unsigned char *out, const unsigned char *in,
    const int blur_radius,
    const unsigned w, const unsigned h, const unsigned nchannels)
{
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            pixel_average(out, in, x, y, blur_radius, w, h, nchannels);
        }
    }
}


void unsharp_mask(unsigned char *out, const unsigned char *in,
    const int blur_radius,
    const unsigned w, const unsigned h, const unsigned nchannels)
{
    std::vector<unsigned char> blur1, blur2, blur3;

    blur1.resize(w * h * nchannels);
    blur2.resize(w * h * nchannels);
    blur3.resize(w * h * nchannels);

    blur(blur1.data(), in, blur_radius, w, h, nchannels);
    blur(blur2.data(), blur1.data(), blur_radius, w, h, nchannels);
    blur(blur3.data(), blur2.data(), blur_radius, w, h, nchannels);

    add_weighted(out, in, 1.5f, blur3.data(), -0.5f, 0.0f, w, h, nchannels);
}

int main(int argc, char *argv[])
{
    const char *ifilename = argc > 1 ? argv[1] : "../library/lena.ppm";
    const char *ofilename = argc > 2 ? argv[2] : "./out.ppm";
    const int blur_radius = argc > 3 ? std::atoi(argv[3]) : 5;

    ppm img(ifilename);

    std::vector<unsigned char> data_sharp(img.w * img.h * img.nchannels);
    unsharp_mask(data_sharp.data(), img.data.data(), blur_radius, img.w, img.h, img.nchannels);

    img.write(ofilename, data_sharp);
    return 0;
}

