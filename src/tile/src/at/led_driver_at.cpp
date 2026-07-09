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

void LedDriverAT::test_pattern() {
    // Startup flash so we know the MCU booted and can drive LEDs.
    strip.fill(strip.Color(255, 0, 0)); strip.show(); delay(600);
    strip.fill(strip.Color(0, 255, 0)); strip.show(); delay(600);
    strip.fill(strip.Color(0, 0, 255)); strip.show(); delay(600);

    strip.fill(strip.Color(255, 0, 0)); strip.show(); delay(600);
    strip.fill(strip.Color(0, 255, 0)); strip.show(); delay(600);
    strip.fill(strip.Color(0, 0, 255)); strip.show(); delay(600);

    strip.clear();
    strip.show();
}

void LedDriverAT::test_light(uint8_t idx, uint8_t r, uint8_t g, uint8_t b) {
    if (idx >= PixelBuffer::NUM_LEDS) return;
    strip.setPixelColor(idx, r, g, b);
    strip.show();
}