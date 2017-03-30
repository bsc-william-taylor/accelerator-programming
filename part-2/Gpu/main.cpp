
#include "../library/benchmark.hpp"
#include "../Library/blur.hpp"
#include "../Library/misc.hpp"
#include "../Library/clu.hpp"
#include "../Library/ppm.hpp"

int main(int argc, const char * argv[])
{
    const auto inFilename = argc > 1 ? argv[1] : "../library/lena.ppm";
    const auto outFilename = argc > 2 ? argv[2] : "./out.ppm";
    const auto radius = argc > 3 ? std::atoi(argv[3]) : 5;

    ppm image(inFilename);

    cl::Platform platform(cl::Platform::get());
    cl::Device device(cl::getDevice(platform));
    cl::Events events(cl::getEvents(2));
    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    auto dataRGBA = toRGBA(image.data, image.w, image.h);
    auto format = cl::ImageFormat{ CL_RGBA, CL_UNORM_INT8 };
    auto gaussianFilter = gaussianFilter1D(radius);
    auto maskSize = gaussianFilter.size() * sizeof(float);
    auto maskFlags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
    auto timeTaken = 0.0;

    cl::Image2D inputImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, format, image.w, image.h, 0, dataRGBA.data());
    cl::Image2D passImage(context, CL_MEM_READ_WRITE, format, image.w, image.h);
    cl::Image2D outImage(context, CL_MEM_WRITE_ONLY, format, image.w, image.h);
    cl::Buffer blurMask(context, maskFlags, maskSize, gaussianFilter.data());
    cl::Program program = cl::getKernel(context, device, "../Gpu/kernels.cl", [&](auto& options) {
        options << " -Dalpha=" << 1.5;
        options << " -Dgamma=" << 0.0;
        options << " -Dbeta=" << -0.5;
        options << " -Dradius=" << (int)ceil(radius * 2.57);
        options << " -cl-fast-relaxed-math";
    });

    auto passOne = cl::getKernel(program, "unsharp_mask_pass_one", inputImage, passImage, blurMask);
    auto passTwo = cl::getKernel(program, "unsharp_mask_pass_two", inputImage, passImage, outImage, blurMask);
    auto region = cl::new_size_t<3>({ (int)image.w, (int)image.h, 1 });
    auto origin = cl::new_size_t<3>({ 0, 0, 0 });

    queue.enqueueNDRangeKernel(passOne, cl::NullRange, { image.w, image.h }, cl::NullRange, nullptr, &events[0]);
    queue.finish();

    queue.enqueueNDRangeKernel(passTwo, cl::NullRange, { image.w, image.h }, cl::NullRange, nullptr, &events[1]);
    queue.enqueueReadImage(outImage, CL_TRUE, origin, region, 0, 0, dataRGBA.data());

    cl::waitEvents(events);
    cl::timeEvents(events[0], events[1], timeTaken);

    image.write(outFilename, toRGB(dataRGBA, image.w, image.h));

#ifdef BENCHMARK
    std::cout << "Compute Time: (ms) " << timeTaken << std::endl;
#endif
}