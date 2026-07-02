#include "transport.h"
#include "led_driver.h"
#include "sense.h"
#include "command_handler.h"
#include "native/transport_native.h"
#include "native/led_driver_native.h"
#include "native/sense_native.h"

static constexpr uint8_t MY_ADDR = 0x01; // TODO: accept as CLI argument
static constexpr uint8_t SLOT    = 0;
static constexpr int     ROW     = 0;

int main() {
    SenseNative     sense(-1); // fd: -1 until socket connected (issue #10)
    TransportNative transport(MY_ADDR, SLOT, ROW);
    LedDriverNative led_driver(MY_ADDR);
    PixelBuffer     pixel_buf;
    FrameParser     parser;

    transport.init();
    led_driver.init();

    // TODO (issue #6): event loop — poll transport, dispatch frames, push LEDs
    return 0;
}
