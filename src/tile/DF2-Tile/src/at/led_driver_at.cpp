#include <Arduino.h>
#include "led_driver_at.h"
#include "pins.h"

LedDriverAT::LedDriverAT()
    : strip(PixelBuffer::NUM_LEDS, PIN_LED_DATA, NEO_GRB + NEO_KHZ800) {}

void LedDriverAT::init() {
    strip.begin();
    strip.show();            // drive all pixels off immediately
    strip.setBrightness(255);
}

void LedDriverAT::push(const PixelBuffer &buf) {
    for (uint8_t i = 0; i < PixelBuffer::NUM_LEDS; i++)
        strip.setPixelColor(i, buf.leds[i].r, buf.leds[i].g, buf.leds[i].b);
    strip.show();            // blocking ~1.2 ms, interrupts disabled during push
}
