
#pragma once

#include "../library/benchmark.hpp"
#include "../library/blur.hpp"
#include "../library/ppm.hpp"
#include "../library/glfw3.h"
#include "../Library/misc.hpp"
#include "../library/clu.hpp"

class App
{
    cl::ImageGL outputCL, inputCL, bufferCL;
    cl::NDRange global;
    cl::Buffer maskBuffer;

    cl::CommandQueue queue;
    cl::Platform platform;
    cl::Context context;
    cl::Program program;
    cl::Device device;
    cl::Kernel kernel;
    
    GLuint outputID, inputID, bufferID;
    int blurRadius, radius, texture;
    double alpha, beta, gamma;
    std::string filename;
    ppm source;
    bool outputted;
public:
    App();
    virtual ~App();

    void setupTexture(GLuint& gl, cl::ImageGL& cl, int type, int w, int h, void* data);
    void setupMask(bool refreshOutput = false);
    void setupOpenCL();

    bool hasImage();

    void showSource();
    void showSecondPass();
    void showFinalResult();

    void updateTexture();
    void updateRadius(int increase);
    void updateFile();

    std::string toString();
};
