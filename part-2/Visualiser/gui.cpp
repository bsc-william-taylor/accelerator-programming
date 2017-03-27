
#include "Gui.h"

Gui::Gui(std::string title)
{
    glfwInit();

    window = glfwCreateWindow(640, 480, title.c_str(), NULL, NULL);

    glfwMakeContextCurrent(window);
    glfwSetWindowUserPointer(window, this);
    glfwSetWindowSizeCallback(window, [](auto win, auto w, auto h) { glViewport(0, 0, w, h); });
    glfwSetKeyCallback(window, [](auto win, int key, int scan, int action, int mods)
    {
        auto display = static_cast<Gui*>(glfwGetWindowUserPointer(win));

        if (display->keyHandler != nullptr)
        {
            display->keyHandler(key, action);
        }
    });
}

Gui::~Gui()
{
    glfwTerminate();
}

void Gui::title(std::string title)
{
    glfwSetWindowTitle(window, title.c_str());
}

void Gui::quad(float size)
{
    glBegin(GL_QUADS);
    glTexCoord2d(0.0, 0.0); glVertex2d(-size, size);
    glTexCoord2d(1.0, 0.0); glVertex2d(size, size);
    glTexCoord2d(1.0, 1.0); glVertex2d(size, -size);
    glTexCoord2d(0.0, 1.0); glVertex2d(-size, -size);
    glEnd();
}

void Gui::onInput(KeyInputCallback keyCallback)
{
    keyHandler = keyCallback;
}

void Gui::onPaint(RenderCallback renderCallback)
{
    while (!glfwWindowShouldClose(window))
    {
        glClear(GL_COLOR_BUFFER_BIT);
        glClearColor(0.2f, 0.2f, 0.2f, 0.2f);

        renderCallback();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}