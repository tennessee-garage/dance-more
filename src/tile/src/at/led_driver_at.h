#pragma once
#include "led_driver.h"
#include <tinyNeoPixel.h>

class LedDriverAT : public ILedDriver {
    tinyNeoPixel strip;
public:
    LedDriverAT();
    void init() override;
    void push(const PixelBuffer &buf) override;

    void test_pattern();
    void test_light(uint8_t idx, uint8_t r, uint8_t g, uint8_t b);
    void clear() { strip.clear(); strip.show(); }
};
