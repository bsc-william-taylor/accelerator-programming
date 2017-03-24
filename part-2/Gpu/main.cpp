
#include "../library/benchmark.hpp"
#include "../library/utilities.hpp"
#include "../Library/blur.hpp"
#include "../library/ppm.hpp"

int main(int argc, const char * argv[])
{
    ppm image(arg(argc, argv, 1, "../Library/lena.ppm"));
    auto radius = (int)pow(std::atoi(arg(argc, argv, 3, "3")), 2);
    auto output = arg(argc, argv, 2, "./cl-out.ppm");
    auto rgba = rgb_to_rgba(image.data, image.w, image.h);

    cl::Platform platform = findPlatform();
    cl::Device device = findDevice(platform);
    cl::Context context = createContext(platform, device, true);
    cl::Program program = createKernel(context, device, "../Gpu/kernels.cl");
  
    std::vector<std::uint8_t> outputPixels(rgba.size());
    std::vector<float> mask = filter(radius, 5.0f);

    try
    {
        cl::ImageFormat format{ CL_RGBA, CL_UNORM_INT8 };
        cl::CommandQueue queue(context, device);
        cl::Kernel imageKernel(program, "unsharp_mask");
        cl::Buffer maskBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mask.size() * sizeof(float), mask.data());
        cl::Image2D imageBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, format, image.w, image.h, 0, rgba.data());
        cl::Image2D outputBuffer(context, CL_MEM_READ_WRITE, format, image.w, image.h);

        cl::size_t<3> region = cl::new_size_t<3>({ (int)image.w, (int)image.h, 1 });
        cl::size_t<3> origin = cl::new_size_t<3>({ 0, 0, 0 });
        cl::NDRange local(16, 16), global(image.w, image.h);
      
        imageKernel.setArg(0, imageBuffer);
        imageKernel.setArg(1, outputBuffer);
        imageKernel.setArg(2, maskBuffer);
        imageKernel.setArg(3, radius);

        queue.enqueueNDRangeKernel(imageKernel, cl::NullRange, global, local);
        queue.enqueueReadImage(outputBuffer, CL_TRUE, origin, region, 0, 0, outputPixels.data());
        queue.finish();
    }
    catch (const cl::Error& e)
    {
        std::cerr << "Exception Caught: " << e.what() << std::endl;
    }
   
    image.write(output, rgb_from_rgba(outputPixels, image.w, image.h));
    return 0;
}