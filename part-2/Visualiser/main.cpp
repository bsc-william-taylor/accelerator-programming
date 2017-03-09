
#include <iostream>
#include "glfw3.h"

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glfw3.lib")

struct GLFWLibrary {
    GLFWLibrary() { glfwInit(); }
    ~GLFWLibrary() { glfwTerminate(); }
};

int main(int argc, char* argv[])
{
    GLFWLibrary library;
    const auto window = glfwCreateWindow(640, 480, "Visualiser", NULL, NULL);
    glfwMakeContextCurrent(window);
    while (!glfwWindowShouldClose(window))
    {
        glClear(GL_COLOR_BUFFER_BIT);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    return EXIT_SUCCESS;
}