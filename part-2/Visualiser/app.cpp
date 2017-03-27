
#include "Gui.h"
#include "App.h"

App::App() :
    platform(findPlatform()),
    device(findDevice(platform)),
    radius(3)
{
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &inputID);
    glGenTextures(1, &outputID);
}


App::~App()
{
    glDeleteTextures(1, &inputID);
    glDeleteTextures(1, &outputID);
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
    context = createContext(platform, device, true);
    program = createKernel(context, device, "../Gpu/kernels.cl");

    kernel = cl::Kernel(program, "unsharp_mask_sections");
    queue = cl::CommandQueue(context, device);
}

void App::setupMask(bool refreshOutput)
{
    offsetX = 0;
    offsetY = 0;

    auto mask = gaussianFilter((radius = clamp(radius, 0, 64)));
    auto size = mask.size() * sizeof(float);

    maskBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, mask.data());

    if (refreshOutput && !filename.empty())
    {
        setupTexture(outputID, outputCL, CL_MEM_WRITE_ONLY, source.w, source.h, nullptr);
    }
}

void App::updateTexture() 
{
    if (hasImage() && offsetX <= source.w && offsetY <= source.h)
    {
        std::vector<cl::Memory> objects{ inputCL, outputCL };
        queue.enqueueAcquireGLObjects(&objects);

        kernel.setArg(0, inputCL);
        kernel.setArg(1, outputCL);
        kernel.setArg(2, maskBuffer);
        kernel.setArg(3, (int)ceil(radius * 2.57));
        kernel.setArg(4, offsetX);
        kernel.setArg(5, offsetY);

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
        queue.enqueueReleaseGLObjects(&objects);
        queue.finish();

        offsetX += source.w  / 10;

        if (offsetX >= source.w) 
        {
            offsetY += source.h / 10;
            offsetX = 0;
        }
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
        global = cl::NDRange(source.w / 10, source.h / 10);
        local = cl::NDRange(1, 1);

        auto rgba = rgb_to_rgba(source.data, source.w, source.h);
        setupTexture(inputID, inputCL, CL_MEM_READ_ONLY, source.w, source.h, rgba.data());
        setupTexture(outputID, outputCL, CL_MEM_WRITE_ONLY, source.w, source.h, nullptr);
    }
}
