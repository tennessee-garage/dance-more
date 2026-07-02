#pragma once
#include "led_driver.h"

class LedDriverAT : public ILedDriver {
public:
    void init() override;
    void push(const PixelBuffer &buf) override;
};
