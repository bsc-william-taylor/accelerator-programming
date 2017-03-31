
#include "Gui.h"
#include "App.h"

App::App() :
    radius(5), outputted(false), texture(0)
{
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &inputID);
    glGenTextures(1, &outputID);
    glGenTextures(1, &bufferID);
}


App::~App()
{
    glDeleteTextures(1, &inputID);
    glDeleteTextures(1, &outputID);
    glDeleteTextures(1, &bufferID);
    glDisable(GL_TEXTURE_2D);
}

void App::setupTexture(GLuint& textureID, cl::ImageGL& imageGL, int type, int w, int h, void* data)
{
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);

    imageGL = cl::ImageGL(context, type, GL_TEXTURE_2D, 0, textureID);
}

void App::setupOpenCL() 
{
    platform = cl::Platform::get();
    device = getDevice(platform);
    context = getContext(platform, device, true);
    queue = cl::CommandQueue(context, device);
    program = getKernel(context, device, "../Gpu/kernels.cl", [&](auto& options) {
        options << " -Dalpha=" << 1.5;
        options << " -Dgamma=" << 0.0;
        options << " -Dbeta=" << -0.5;
        options << " -cl-fast-relaxed-math";
    });
}

void App::setupMask(bool refreshOutput)
{
    auto mask = gaussianFilter1D(radius = clamp(radius, 0, 50));
    auto size = mask.size() * sizeof(float);
    
    maskBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, mask.data());
    blurRadius = (int)ceil(radius * 2.57);

    if (refreshOutput && !filename.empty())
    {
        setupTexture(outputID, outputCL, CL_MEM_WRITE_ONLY, source.w, source.h, nullptr);
        glBindTexture(GL_TEXTURE_2D, texture);
    }
}

void App::updateTexture() 
{
    if (hasImage() && !outputted)
    {
        std::vector<cl::Memory> objects{ inputCL, bufferCL, outputCL };
        queue.enqueueAcquireGLObjects(&objects);

        auto passOne = cl::getKernel(program, "unsharp_mask_pass_one", inputCL, bufferCL, maskBuffer, blurRadius);
        queue.enqueueNDRangeKernel(passOne, cl::NullRange, { source.w, source.h }, cl::NullRange);
        queue.finish();

        auto passTwo = cl::getKernel(program, "unsharp_mask_pass_two", inputCL, bufferCL, outputCL, maskBuffer, blurRadius);
        queue.enqueueNDRangeKernel(passTwo, cl::NullRange, { source.w, source.h }, cl::NullRange);
        queue.enqueueReleaseGLObjects(&objects);

        outputted = true;
    }
}

std::string App::toString()
{
    auto lastBackSlash = filename.find_last_of("\\") + 1;
    auto shortfilename = filename.substr(lastBackSlash);
    auto empty = shortfilename.empty();

    std::stringstream ss;
    ss << "UnsharpFilter: ";
    ss << "Radius = " << radius << ", ";
    ss << "Image = " << (!empty ? shortfilename : "None");
    return ss.str();
}

void App::updateRadius(int increase) 
{
    do
    {
        radius += increase;
    } while (radius % 2 == 0);

    outputted = false;
    setupMask(true);
}

bool App::hasImage() 
{
    return !filename.empty();
}

void App::updateFile()
{
    char buffer[MAX_PATH]{ 0 };

    OPENFILENAME ofn{ 0 };
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = GetDesktopWindow();
    ofn.lpstrFilter = "PPM File\0*.ppm";
    ofn.lpstrFile = buffer;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrTitle = "Select a File!";
    ofn.Flags = OFN_DONTADDTORECENT | OFN_FILEMUSTEXIST;

    if (GetOpenFileNameA(&ofn))
    {
        source.read(buffer);
        filename = buffer;
        global = cl::NDRange(source.w, source.h);

        auto rgba = toRGBA(source.data, source.w, source.h);
        setupTexture(inputID, inputCL, CL_MEM_READ_ONLY, source.w, source.h, rgba.data());
        setupTexture(bufferID, bufferCL, CL_MEM_READ_WRITE, source.w, source.h, nullptr);
        setupTexture(outputID, outputCL, CL_MEM_WRITE_ONLY, source.w, source.h, nullptr);
        texture = outputID;
    }
}

void App::showSource()
{
    glBindTexture(GL_TEXTURE_2D, inputID);
    texture = inputID;
}

void App::showSecondPass()
{
    glBindTexture(GL_TEXTURE_2D, bufferID);
    texture = bufferID;
}

void App::showFinalResult()
{
    glBindTexture(GL_TEXTURE_2D, outputID);
    texture = outputID;
}