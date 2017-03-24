
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#pragma comment(lib, "glfw3.lib")

#include "../library/utilities.hpp"
#include "../library/benchmark.hpp"
#include "../library/ppm.hpp"

#include <Windows.h>
#include <iostream>
#include <sstream>

#include "glfw3.h"
#include "glfw3native.h"
#include <cl/cl.hpp>
#include <CL/cl_gl_ext.h>

struct RenderWork
{
    std::string filename;
    GLuint textureID;
    cl::ImageGL image;
    ppm source;
    int radius;
};

RenderWork task{ "", 0, {}, {}, 3 };

cl::Context context;

struct GLFWLibrary {
    GLFWLibrary() { glfwInit(); }
    ~GLFWLibrary() { glfwTerminate(); }
};

const auto onKeyDown = [](auto input, auto action, auto key, auto callback) {
    if (input == key && action == GLFW_PRESS) {
        callback();
    }
};

void openFile()
{
    char buffer[MAX_PATH];

    OPENFILENAME ofn;
    ZeroMemory(&buffer, sizeof(buffer));
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;  // If you have a window to center over, put its HANDLE here
    ofn.lpstrFilter = "PPM File\0*.ppm";
    ofn.lpstrFile = buffer;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrTitle = "Select a PPM File!";
    ofn.Flags = OFN_DONTADDTORECENT | OFN_FILEMUSTEXIST;

    if (GetOpenFileNameA(&ofn))
    {
        task.source.read(buffer);
        task.filename = buffer;
       
        auto rgba = rgb_to_rgba(task.source.data, task.source.w, task.source.h);

        glBindTexture(GL_TEXTURE_2D, task.textureID);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, task.source.w, task.source.h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());

        task.image = cl::ImageGL(context, CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0, task.textureID, nullptr);
    }
}

void onKey(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    onKeyDown(key, action, GLFW_KEY_ESCAPE, [&]() { glfwDestroyWindow(window); });
    onKeyDown(key, action, GLFW_KEY_DOWN, [&]() { task.radius--; });
    onKeyDown(key, action, GLFW_KEY_UP, [&]() { task.radius++; });
    onKeyDown(key, action, GLFW_KEY_O, [&]() { openFile(); });

    task.radius = clamp(task.radius, 1, 150);
}

void onResize(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

int main(int argc, char* argv[])
{
    std::string src = kernel("kernel.cl");
    std::vector<cl::Device> devices;

    GLFWLibrary library;
    const auto window = glfwCreateWindow(640, 480, "", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, onKey);
    glfwSetWindowSizeCallback(window, onResize);

    cl::Program::Sources kernel(1, std::make_pair(src.c_str(), src.length()));
    cl::Platform p = cl::Platform::getDefault();
    p.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    auto hGLRC = wglGetCurrentContext();
    auto hDC = wglGetCurrentDC();

    cl_context_properties cps[] =
    {
        CL_CONTEXT_PLATFORM, (cl_context_properties)p(),
        CL_GL_CONTEXT_KHR, (cl_context_properties)hGLRC,
        CL_WGL_HDC_KHR, (cl_context_properties)hDC,
        0
    };

    cl::Device device = devices.front();
    context = cl::Context(device, cps);
    cl::CommandQueue queue(context, device);
    cl::Program program(context, kernel);

    try
    {
        program.build();
    }
    catch (const cl::Error& e)
    {
        std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
    }

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &task.textureID);

    cl::Kernel imageKernel(program, "copy");

    while (!glfwWindowShouldClose(window))
    {
        auto fn = task.filename;

        std::stringstream ss;
        ss << "UnsharpFilter: ";
        ss << "Radius = " << task.radius << ", ";
        ss << "Filename = " << fn.substr(fn.find_last_of("\\") + 1);

        glfwSetWindowTitle(window, ss.str().c_str());
        glClear(GL_COLOR_BUFFER_BIT);
        glClearColor(0.2f, 0.2, 0.2f, 0.2f);

        if (!fn.empty() && task.textureID > 0)
        {
            clEnqueueAcquireGLObjects(queue(), 1, &task.image(), 0, 0, NULL);

            cl::size_t<3> region = cl::new_size_t<3>({ (int)task.source.w, (int)task.source.w, 1 });
            cl::size_t<3> origin = cl::new_size_t<3>({ 0, 0, 0 });
            cl::NDRange local(1, 1), global(task.source.w, task.source.h);

            imageKernel.setArg(0, task.image());

            queue.enqueueNDRangeKernel(imageKernel, cl::NullRange, global, local);
            clEnqueueReleaseGLObjects(queue(), 1, &task.image(), 0, 0, NULL);
          
            glBegin(GL_QUADS);
            glBindTexture(GL_TEXTURE_2D, task.textureID);
            glTexCoord2d(0.0, 0.0);
            glVertex2d(-1.0, 1.0);
            glTexCoord2d(1.0, 0.0);
            glVertex2d(1.0, 1.0);
            glTexCoord2d(1.0, 1.0);
            glVertex2d(1.0, -1.0);
            glTexCoord2d(0.0, 1.0);
            glVertex2d(-1.0, -1.0);
            glEnd();
        }
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }


    glDeleteTextures(1, &task.textureID);
    return EXIT_SUCCESS;
}