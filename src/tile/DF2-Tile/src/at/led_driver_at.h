#pragma once
#include "led_driver.h"
#include <Adafruit_NeoPixel.h>

class LedDriverAT : public ILedDriver {
    Adafruit_NeoPixel strip;
public:
    LedDriverAT();
    void init() override;
    void push(const PixelBuffer &buf) override;
};
