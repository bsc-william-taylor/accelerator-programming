
#include "App.h"
#include "Gui.h"

int WINAPI WinMain(HINSTANCE instance, HINSTANCE prev, LPSTR args, int cmd)
{
    Gui display("Visualiser: Press 'O' to open PPM file");

    App app;
    app.setupOpenCL();
    app.setupMask();

    display.onInput([&](int key, int state) {
        if (key == GLFW_KEY_DOWN && state == GLFW_PRESS)
            app.updateRadius(-1);
        if (key == GLFW_KEY_UP && state == GLFW_PRESS) 
            app.updateRadius(1);
        if (key == GLFW_KEY_1 && state == GLFW_PRESS)
            app.showSource();
        if (key == GLFW_KEY_2 && state == GLFW_PRESS)
            app.showSecondPass();
        if (key == GLFW_KEY_3 && state == GLFW_PRESS)
            app.showFinalResult();
        if (key == GLFW_KEY_O && state == GLFW_PRESS)
            app.updateFile();
    });

    display.onPaint([&]() {
        auto title = app.toString();
        app.updateTexture();

        if (app.hasImage())
        {
            display.title(title);
            display.quad(1.0);
        }
    });
}