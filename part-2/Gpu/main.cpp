
#include "../library/benchmark.hpp"
#include "../library/utilities.hpp"
#include "../Library/blur.hpp"
#include "../library/ppm.hpp"

int main(int argc, const char * argv[])
{
    benchmark<1>([&]{
        const char *input = argc > 1 ? argv[1] : "../library/lena.ppm";
        const char *output = argc > 2 ? argv[2] : "./out.ppm";
        const int radius = argc > 3 ? std::atoi(argv[3]) : 5;

        ppm image(input);
        
        auto rgba = rgb_to_rgba(image.data, image.w, image.h);

        cl::Platform platform = findPlatform();
        cl::Device device = findDevice(platform);
        cl::Context context = cl::Context(device);
        cl::Program program = createKernel(context, device, "../Gpu/kernels.cl");
  
        std::vector<float> mask = gaussianFilter(radius);

        cl::ImageFormat format{ CL_RGBA, CL_UNORM_INT8 };
        cl::CommandQueue queue(context, device);
        cl::Kernel imageKernel(program, "unsharp_mask_full");
        cl::Buffer maskBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mask.size() * sizeof(float), mask.data());
        cl::Image2D imageBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, format, image.w, image.h, 0, rgba.data());
        cl::Image2D outputBuffer(context, CL_MEM_READ_WRITE, format, image.w, image.h);

        cl::size_t<3> region = cl::new_size_t<3>({ (int)image.w, (int)image.h, 1 });
        cl::size_t<3> origin = cl::new_size_t<3>({ 0, 0, 0 });
        cl::NDRange local(1, 1), global(image.w, image.h);
      
        imageKernel.setArg(0, imageBuffer);
        imageKernel.setArg(1, outputBuffer);
        imageKernel.setArg(2, maskBuffer);
        imageKernel.setArg(3, radius);

        queue.enqueueNDRangeKernel(imageKernel, cl::NullRange, global, local);
        queue.enqueueReadImage(outputBuffer, CL_TRUE, origin, region, 0, 0, rgba.data());
        queue.finish();

        image.write(output, rgb_from_rgba(rgba, image.w, image.h));
    });
    return 0;
}