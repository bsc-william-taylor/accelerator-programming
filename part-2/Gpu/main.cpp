
#include "../library/benchmark.hpp"
#include "../library/utilities.hpp"
#include "../Library/blur.hpp"
#include "../library/ppm.hpp"

int main(int argc, const char * argv[])
{
    const auto input = argc > 1 ? argv[1] : "../library/lena.ppm";
    const auto output = argc > 2 ? argv[2] : "./out.ppm";
    const auto radius = argc > 3 ? std::atoi(argv[3]) : 3;

    ppm image(input);
    auto rgba = rgb_to_rgba(image.data, image.w, image.h);
    auto gaussianMask = gaussianFilter(radius);
    auto gaussianSize = gaussianMask.size() * sizeof(float);
    auto region = cl::new_size_t<3>({ (int)image.w, (int)image.h, 1 });
    auto origin = cl::new_size_t<3>({ 0, 0, 0 });
    auto flags = cl_mem_flags{ CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR };
    auto format = cl::ImageFormat{ CL_RGBA, CL_UNORM_INT8 };

    cl::Platform platform = findPlatform();
    cl::Device device = findDevice(platform);
    cl::Context context = cl::Context(device);
    cl::Image2D inputImage(context, flags, format, image.w, image.h, 0, rgba.data());
    cl::Image2D outputImage(context, CL_MEM_READ_WRITE, format, image.w, image.h);
    cl::Buffer mask(context, flags, gaussianSize, gaussianMask.data());
    cl::Program program = createKernel(context, device, "../Gpu/kernels.cl", [&]() {
        std::stringstream options;
        options << " -Dalpha=" << 1.5;
        options << " -Dgamma=" << 0.0;
        options << " -Dbeta=" << -0.5;
        options << " -Dradius=" << (int)ceil(radius * 2.57);
        options << " -cl-unsafe-math-optimizations ";
        options << " -cl-mad-enable ";
        return options.str();
    });

    cl::CommandQueue queue(context, device);
    cl::Kernel kernel(program, "unsharp_mask_full");
    kernel.setArg(0, inputImage);
    kernel.setArg(1, outputImage);
    kernel.setArg(2, mask);

    benchmark<1>("gpu-benchmark.csv", [&] {
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, { image.w, image.h }, { 1, 1 });
        queue.enqueueReadImage(outputImage, CL_TRUE, origin, region, 0, 0, rgba.data());
    });

    image.write(output, rgb_from_rgba(rgba, image.w, image.h));
    return 0;
}