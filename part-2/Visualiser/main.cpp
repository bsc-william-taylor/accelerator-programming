
#include "../library/utilities.hpp"
#include "../library/benchmark.hpp"
#include "../library/blur.hpp"
#include "../library/ppm.hpp"

#include "glfw3.h"

#pragma comment(lib, "glfw3.lib")

struct RenderWork
{
    std::pair<GLuint, cl::ImageGL> output;
    std::pair<GLuint, cl::ImageGL> input;
    std::string filename;

    cl::Buffer mask;
    cl::Context context;
    cl::NDRange global;
    cl::NDRange local;

    ppm source;
    int radius;
};

RenderWork task;

struct GLFWLibrary {
    GLFWLibrary() { glfwInit(); }
    ~GLFWLibrary() { glfwTerminate(); }
};

const auto onKeyDown = [](auto input, auto action, auto key, auto callback) 
{
    if (input == key && action == GLFW_PRESS)
    {
        callback();
    }
};

void createSharedTexture(std::pair<GLuint, cl::ImageGL>& texturePair, int type, int w, int h, void* data)
{
    glBindTexture(GL_TEXTURE_2D, texturePair.first);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);

    texturePair.second = cl::ImageGL(task.context, type, GL_TEXTURE_2D, 0, texturePair.first);
}

void openFile()
{
    char buffer[MAX_PATH];

    OPENFILENAME ofn;
    ZeroMemory(&buffer, sizeof(buffer));
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = GetDesktopWindow();  
    ofn.lpstrFilter = "PPM File\0*.ppm";
    ofn.lpstrFile = buffer;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrTitle = "Select a PPM File!";
    ofn.Flags = OFN_DONTADDTORECENT | OFN_FILEMUSTEXIST;

    if (GetOpenFileNameA(&ofn))
    {
        task.source.read(buffer);
        task.filename = buffer;
        task.global = cl::NDRange(task.source.w, task.source.h);
        task.local = cl::NDRange(1, 1);

        auto rgba = rgb_to_rgba(task.source.data, task.source.w, task.source.h);
        createSharedTexture(task.input, CL_MEM_READ_ONLY, task.source.w, task.source.h, rgba.data());
        createSharedTexture(task.output, CL_MEM_WRITE_ONLY, task.source.w, task.source.h, nullptr);
    }
}

void generateMask(RenderWork& task)
{
    std::vector<float> mask = filter(task.radius, 5.0f);
    task.mask = cl::Buffer(task.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mask.size() * sizeof(float), mask.data());
}

void onKey(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    const auto radiusCopy = task.radius;

    onKeyDown(key, action, GLFW_KEY_ESCAPE, [&]() { glfwDestroyWindow(window); });
    onKeyDown(key, action, GLFW_KEY_DOWN, [&]() { task.radius--; });
    onKeyDown(key, action, GLFW_KEY_UP, [&]() { task.radius++; });
    onKeyDown(key, action, GLFW_KEY_O, [&]() { openFile(); });

    task.radius = clamp(task.radius, 0, 150);

    if (task.radius != radiusCopy)
    {
        generateMask(task);
    }
}

void onResize(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void updateTitle(GLFWwindow* window)
{
    auto lastBackSlash = task.filename.find_last_of("\\")+1;
    auto rawFilename = task.filename.substr(lastBackSlash);

    std::stringstream ss;
    ss << "UnsharpFilter: ";
    ss << "Radius = " << task.radius << ", ";
    ss << "Filename = " << rawFilename;

    glfwSetWindowTitle(window, ss.str().c_str());
}

int WINAPI WinMain(HINSTANCE instance, HINSTANCE prev, LPSTR args, int cmdShow)
{
    GLFWLibrary library;
    GLFWwindow* window = glfwCreateWindow(640, 480, "", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, onKey);
    glfwSetWindowSizeCallback(window, onResize);

    cl::Platform platform = findPlatform();
    cl::Device device = findDevice(platform);
    cl::Context context = createContext(platform, device, true);
    cl::Program program = createKernel(context, device, "../Gpu/kernels.cl");  
    cl::CommandQueue queue(context, device);
    cl::Kernel imageKernel(program, "unsharp_mask");

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &task.input.first);
    glGenTextures(1, &task.output.first);

    task.context = context;
    task.radius = 3;
    generateMask(task);

    while (!glfwWindowShouldClose(window))
    {
        glClear(GL_COLOR_BUFFER_BIT);
        glClearColor(0.2f, 0.2, 0.2f, 0.2f);

        if (!task.source.data.empty())
        {
            std::vector<cl::Memory> objects{ task.input.second, task.output.second };
            queue.enqueueAcquireGLObjects(&objects);

            imageKernel.setArg(0, task.input.second());
            imageKernel.setArg(1, task.output.second());
            imageKernel.setArg(2, task.mask);
            imageKernel.setArg(3, task.radius);

            queue.enqueueNDRangeKernel(imageKernel, cl::NullRange, task.global, task.local);
            queue.enqueueReleaseGLObjects(&objects);
            queue.finish();

            glBegin(GL_QUADS);
            glTexCoord2d(0.0, 0.0); glVertex2d(-1.0, 1.0);
            glTexCoord2d(1.0, 0.0); glVertex2d(1.0, 1.0);
            glTexCoord2d(1.0, 1.0); glVertex2d(1.0, -1.0);
            glTexCoord2d(0.0, 1.0); glVertex2d(-1.0, -1.0);
            glEnd();
        }
        
        glfwSwapBuffers(window);
        glfwPollEvents();

        updateTitle(window);
    }

    glDeleteTextures(1, &task.input.first);
    glDeleteTextures(1, &task.output.first);
    glDisable(GL_TEXTURE_2D);

    return EXIT_SUCCESS;
}