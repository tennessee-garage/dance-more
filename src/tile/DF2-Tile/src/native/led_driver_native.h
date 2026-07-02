#pragma once
#include "led_driver.h"
#include <stdint.h>

class LedDriverNative : public ILedDriver {
    uint8_t  tile_addr;
    uint32_t frame_count = 0;
public:
    explicit LedDriverNative(uint8_t addr);
    void init() override;
    void push(const PixelBuffer &buf) override;
};
