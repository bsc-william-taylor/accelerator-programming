
#pragma once

#include "../library/utilities.hpp"
#include "../library/benchmark.hpp"
#include "../library/blur.hpp"
#include "../library/ppm.hpp"
#include "../library/glfw3.h"

class App
{
    cl::ImageGL outputCL, inputCL;
    cl::NDRange global, local;
    cl::Buffer maskBuffer;

    cl::CommandQueue queue;
    cl::Platform platform;
    cl::Context context;
    cl::Program program;
    cl::Device device;
    cl::Kernel kernel;
    
    int offsetX, offsetY, radius;
    GLuint outputID, inputID;
    std::string filename;
    ppm source;
public:
    App();
    virtual ~App();

    void setupTexture(GLuint& gl, cl::ImageGL& cl, int type, int w, int h, void* data);
    void setupMask(bool refreshOutput = false);
    void setupOpenCL();

    bool hasImage();

    void updateTexture();
    void updateRadius(int increase);
    void updateFile();

    std::string toString();
};
