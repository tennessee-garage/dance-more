#pragma once
#include <stdint.h>

struct Pixel {
    uint8_t r, g, b;
};

struct PixelBuffer {
    static constexpr uint8_t NUM_LEDS = 40;
    Pixel leds[NUM_LEDS];
    bool  latch_pending = false;
};
