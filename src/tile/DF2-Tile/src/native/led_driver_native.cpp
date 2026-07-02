#include "led_driver_native.h"
#include <cstdio>

// Implemented in issue #8.
LedDriverNative::LedDriverNative(uint8_t addr) : tile_addr(addr) {}

void LedDriverNative::init() {
    printf("[tile 0x%02X] LED driver ready (40 LEDs, native/log mode)\n", tile_addr);
}

void LedDriverNative::push(const PixelBuffer &) {}
