#include <Arduino.h>
#include "transport.h"
#include "led_driver.h"
#include "sense.h"
#include "command_handler.h"
#include "at/transport_at.h"
#include "at/led_driver_at.h"
#include "at/sense_at.h"

static constexpr uint8_t MY_ADDR = 0x01; // TODO: read from EEPROM

static TransportAT transport;
static LedDriverAT led_driver;
static SenseAT     sense;
static PixelBuffer pixel_buf;
static FrameParser parser;

void setup() {
    sense.init();
    transport.init();
    led_driver.init();
}

void loop() {
    transport.poll(parser);
    // TODO (issue #3): wire parser.feed() result → handle_command → response TX

    if (pixel_buf.latch_pending) {
        led_driver.push(pixel_buf);
        pixel_buf.latch_pending = false;
    }
}
