
#pragma once

#include "../Library/misc.hpp"
#include "../Library/glfw3.h"
#include "../Library/clu.hpp"

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
