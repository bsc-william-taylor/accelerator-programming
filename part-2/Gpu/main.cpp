
#include "../library/benchmark.hpp"
#include "../Library/blur.hpp"
#include "../Library/misc.hpp"
#include "../Library/clu.hpp"
#include "../Library/ppm.hpp"

int main(int argc, const char * argv[])
{
    const auto input = argc > 1 ? argv[1] : "../library/lena.ppm";
    const auto output = argc > 2 ? argv[2] : "./out.ppm";
    const auto radius = argc > 3 ? std::atoi(argv[3]) : 5;

    ppm image(input);

    auto rgba = toRGBA(image.data, image.w, image.h);
    auto gaussianMask = gaussianFilter1D(radius);
    auto gaussianSize = gaussianMask.size() * sizeof(float);
    auto format = cl::ImageFormat{ CL_RGBA, CL_UNORM_INT8 };
    auto region = cl::new_size_t<3>({ (int)image.w, (int)image.h, 1 });
    auto origin = cl::new_size_t<3>({ 0, 0, 0 });

    cl::Platform platform = cl::Platform::get();
    cl::Device device = cl::getDevice(platform);
    cl::Context context(device);
    cl::CommandQueue queue(context, device);
    cl::Buffer blurmask(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, gaussianSize, gaussianMask.data());
    cl::Image2D inputImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, format, image.w, image.h, 0, rgba.data());
    cl::Image2D bufferImage(context, CL_MEM_READ_WRITE, format, image.w, image.h);
    cl::Image2D outputImage(context, CL_MEM_WRITE_ONLY, format, image.w, image.h);
    cl::Program program = cl::getKernel(context, device, "../Gpu/kernels.cl", [&]() {
        std::stringstream options;
        options << " -Dalpha=" << 1.5;
        options << " -Dgamma=" << 0.0;
        options << " -Dbeta=" << -0.5;
        options << " -Dradius=" << (int)ceil(radius * 2.57);
        options << " -Dmasksize=" << gaussianSize;
        options << " -cl-fast-relaxed-math";
        return options.str();
    });

    benchmark<1>("gpu-benchmark.csv", [&] {
        cl::Kernel first(program, "unsharp_mask_pass_one");
        first.setArg(0, inputImage);
        first.setArg(1, bufferImage);
        first.setArg(2, blurmask);

        queue.enqueueNDRangeKernel(first, cl::NullRange, { image.w, image.h }, cl::NullRange);
        queue.finish();

        cl::Kernel second(program, "unsharp_mask_pass_two");
        second.setArg(0, inputImage);
        second.setArg(1, bufferImage);
        second.setArg(2, outputImage);
        second.setArg(3, blurmask);

        queue.enqueueNDRangeKernel(second, cl::NullRange, { image.w, image.h }, cl::NullRange);
        queue.enqueueReadImage(outputImage, CL_TRUE, origin, region, 0, 0, rgba.data());
        queue.finish();
    });

    image.write(output, toRGB(rgba, image.w, image.h));
    return 0;
}