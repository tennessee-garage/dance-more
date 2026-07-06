#include <Arduino.h>
#include <tinyNeoPixel.h>
#include "at/pins.h"

// Standalone LED connectivity test — no protocol, no RS-485.
// Cycles through solid colours, a chase, and a rainbow to verify
// the MCU → WS2815 data line and each colour channel are working.

static constexpr uint8_t NUM_LEDS = 40;
static tinyNeoPixel strip(NUM_LEDS, PIN_LED_DATA, NEO_GRB + NEO_KHZ800);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void fill(uint8_t r, uint8_t g, uint8_t b, uint16_t hold_ms)
{
    strip.fill(strip.Color(r, g, b));
    strip.show();
    delay(hold_ms);
}

static uint32_t hsv_color(uint8_t hue)   // hue 0-255
{
    return strip.ColorHSV((uint16_t)hue << 8, 255, 255);
}

// ---------------------------------------------------------------------------
// Patterns
// ---------------------------------------------------------------------------

static void pattern_solid_cards()
{
    // Each primary, secondary and white — long enough to count LEDs by eye.
    fill(255,   0,   0, 1000);   // red
    fill(  0, 255,   0, 1000);   // green
    fill(  0,   0, 255, 1000);   // blue
    fill(255, 255,   0, 1000);   // yellow
    fill(  0, 255, 255, 1000);   // cyan
    fill(255,   0, 255, 1000);   // magenta
    fill(255, 255, 255, 1000);   // white
    fill(  0,   0,   0,  500);   // off
}

static void pattern_chase(uint8_t r, uint8_t g, uint8_t b, uint8_t laps = 2)
{
    // Single bright pixel with a short fading trail, laps × NUM_LEDS frames.
    static constexpr uint8_t TRAIL = 5;
    static constexpr uint16_t STEP_MS = 30;
    uint16_t total = (uint16_t)laps * NUM_LEDS;
    for (uint16_t t = 0; t < total; t++) {
        strip.clear();
        for (uint8_t i = 0; i <= TRAIL; i++) {
            uint8_t idx  = (uint8_t)((t - i + NUM_LEDS) % NUM_LEDS);
            float   fade = 1.0f - (float)i / (TRAIL + 1);
            strip.setPixelColor(idx, strip.Color(
                (uint8_t)(r * fade),
                (uint8_t)(g * fade),
                (uint8_t)(b * fade)));
        }
        strip.show();
        delay(STEP_MS);
    }
}

static void pattern_rainbow(uint8_t laps = 2)
{
    // Hue gradient rotating around all LEDs.
    static constexpr uint16_t STEP_MS = 30;
    uint16_t total = (uint16_t)laps * 256;
    for (uint16_t t = 0; t < total; t++) {
        for (uint8_t i = 0; i < NUM_LEDS; i++) {
            uint8_t hue = (uint8_t)((i * 256 / NUM_LEDS + t) & 0xFF);
            strip.setPixelColor(i, strip.gamma32(hsv_color(hue)));
        }
        strip.show();
        delay(STEP_MS);
    }
}

// ---------------------------------------------------------------------------
// Arduino entry points
// ---------------------------------------------------------------------------

void setup()
{
    strip.begin();
    strip.setBrightness(255);
    strip.show();   // all off at power-on
}

void loop()
{
    pattern_solid_cards();
    pattern_chase(255, 255, 255);   // white chase
    pattern_chase(255,   0,   0);   // red chase
    pattern_chase(  0, 255,   0);   // green chase
    pattern_chase(  0,   0, 255);   // blue chase
    pattern_rainbow();
}
