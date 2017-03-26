
#include "../library/benchmark.hpp"
#include "../library/utilities.hpp"
#include "../Library/blur.hpp"
#include "../library/ppm.hpp"

int main(int argc, const char * argv[])
{
    const auto input = argc > 1 ? argv[1] : "../library/lena.ppm";
    const auto output = argc > 2 ? argv[2] : "./out.ppm";
    const auto radius = argc > 3 ? std::atoi(argv[3]) : 5;

    ppm image(input);
    auto rgba = rgb_to_rgba(image.data, image.w, image.h);

    benchmark<1>("gpu-benchmark.csv", [&] {
        auto gaussianMask = gaussianFilter(radius);
        auto gaussianSize = gaussianMask.size() * sizeof(float);
        auto region = cl::new_size_t<3>({ (int)image.w, (int)image.h, 1 });
        auto origin = cl::new_size_t<3>({ 0, 0, 0 });
        auto flags = cl_mem_flags{ CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR};
        auto format = cl::ImageFormat{ CL_RGBA, CL_UNORM_INT8 };

        cl::Platform platform = findPlatform();
        cl::Device device = findDevice(platform);
        cl::Context context = cl::Context(device);
        cl::Image2D input(context, flags, format, image.w, image.h, 0, rgba.data());
        cl::Image2D output(context, CL_MEM_READ_WRITE, format, image.w, image.h);
        cl::Buffer mask(context, flags, gaussianSize, gaussianMask.data());

        cl::Program program = createKernel(context, device, "../Gpu/kernels.cl");
        cl::Kernel kernel(program, "unsharp_mask_full");
        kernel.setArg(0, input);
        kernel.setArg(1, output);
        kernel.setArg(2, mask);
        kernel.setArg(3, radius);

        cl::CommandQueue queue(context, device);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, { image.w, image.h }, { 1, 1 });
        queue.enqueueReadImage(output, CL_TRUE, origin, region, 0, 0, rgba.data());
        queue.finish();
    });

    image.write(output, rgb_from_rgba(rgba, image.w, image.h));
    return 0;
}