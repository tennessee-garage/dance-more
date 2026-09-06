#pragma once
#include <stdint.h>
// LEDS_PER_TILE lives with the wire protocol because MAX_PAYLOAD is derived
// from it - one number, one place. Relative include for the same reason as
// command_handler.h: PlatformIO's LDF doesn't propagate lib_extra_dirs
// include paths into other lib/ folders' own compile steps.
#include "../../../common/tile_bus_protocol/protocol.h"

struct Pixel {
    uint8_t r, g, b;
};

struct PixelBuffer {
    static constexpr uint8_t NUM_LEDS = LEDS_PER_TILE;
    Pixel leds[NUM_LEDS];
    bool  latch_pending = false;
};
