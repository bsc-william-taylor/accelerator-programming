
#pragma once

#include "../library/utilities.hpp"
#include "../library/glfw3.h"

class Gui
{
    using KeyInputCallback = std::function<void(int, int)>;
    using RenderCallback = std::function<void()>;

    KeyInputCallback keyHandler;
    GLFWwindow* window;
public:
    Gui(std::string title);
    virtual ~Gui();
    
    void onInput(KeyInputCallback onKey);
    void onPaint(RenderCallback onDraw);
    void title(std::string title);
    void quad(float size = 1.0f);
};
