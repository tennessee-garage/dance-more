#pragma once
#include "led_driver.h"
#include <tinyNeoPixel.h>

class LedDriverAT : public ILedDriver {
    tinyNeoPixel strip;
public:
    LedDriverAT();
    void init() override;
    void push(const PixelBuffer &buf) override;

    // Blocking: ~3.6 s of delay() calls. Fine for the standalone LED test
    // harness, but do not call it from the main firmware's setup() - that is
    // exactly what made tiles miss the row controller's boot discovery. See
    // the start-up pattern comment in src/main.cpp.
    void test_pattern();
    void test_light(uint8_t idx, uint8_t r, uint8_t g, uint8_t b);
    void clear() { strip.clear(); strip.show(); }
};
