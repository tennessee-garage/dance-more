#include <Arduino.h>
#include <Adafruit_NeoPixel.h>
#include "protocol.h"   // src/common/tile_bus_protocol/, for LEDS_PER_SIDE

// Hand-held tester for freshly soldered WS2815 strips.
//
// One button. Press it (any length of press) and the strip runs a solid
// red -> green -> blue sequence, long enough to eyeball every LED on each
// channel. If the button has been released by the time that finishes, the
// strip goes dark. If it is still held, a rainbow rolls along the strip
// until it is released.
//
// Nothing here is time-critical except the WS2815 bitstream, which
// Adafruit_NeoPixel hands to the ESP32-C3's RMT peripheral.

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

// XIAO ESP32-C3: D2 = GPIO4, D1 = GPIO3. GPIO8/GPIO9 are boot-strapping pins
// and are deliberately not used. Button goes between PIN_BUTTON and GND; the
// internal pull-up does the rest.
static constexpr uint8_t PIN_LED_DATA = 4;
static constexpr uint8_t PIN_BUTTON   = 3;

// Up to four sides' worth of strip can be chained onto the tester at once
// (one tile). Shorter chains just leave the tail of the buffer unused.
static constexpr uint16_t NUM_LEDS = 4 * LEDS_PER_SIDE;

// ---------------------------------------------------------------------------
// Power budget
// ---------------------------------------------------------------------------
//
// The tester runs off two 9 V batteries in series. A PP3 alkaline is happy
// at ~100 mA and sags badly much past 250 mA, so the strip has to be kept
// well short of what it could draw: a WS2815 pulls roughly 15 mA at full
// white, i.e. ~5 mA per channel, so 60 LEDs on one solid primary is ~300 mA
// at 12 V at full brightness, and full white would be ~900 mA.
//
// BRIGHTNESS scales every colour below (0-255). At 64 (25 %) the solid
// phases sit at ~75 mA and the rainbow, which lights at most two channels
// per LED, at ~110 mA - comfortably inside what the batteries will give for
// an afternoon of soldering. Still bright enough to spot a dead LED or a
// swapped channel at arm's length.
static constexpr uint8_t BRIGHTNESS = 64;

// How long each solid colour holds. Long enough to count LEDs.
static constexpr uint16_t SOLID_HOLD_MS = 1000;

// Rainbow animation: hue advance per frame, and frame period.
static constexpr uint16_t RAINBOW_HUE_STEP = 512;   // of 65536
static constexpr uint16_t RAINBOW_FRAME_MS = 20;

// Button must read the same for this long before a change is believed.
static constexpr uint16_t DEBOUNCE_MS = 20;

// ---------------------------------------------------------------------------
// Button
// ---------------------------------------------------------------------------

static bool debounced_pressed = false;

// Polls the pin and returns the debounced state: true while the button is
// held. Cheap enough to call from inside animation loops.
static bool button_pressed()
{
    static bool     last_raw   = false;
    static uint32_t changed_at = 0;

    bool raw = digitalRead(PIN_BUTTON) == LOW;   // active low, pulled up
    if (raw != last_raw) {
        last_raw   = raw;
        changed_at = millis();
    } else if (raw != debounced_pressed &&
               (uint32_t)(millis() - changed_at) >= DEBOUNCE_MS) {
        debounced_pressed = raw;
    }
    return debounced_pressed;
}

// ---------------------------------------------------------------------------
// Patterns
// ---------------------------------------------------------------------------

static Adafruit_NeoPixel strip(NUM_LEDS, PIN_LED_DATA, NEO_GRB + NEO_KHZ800);

static void fill(uint8_t r, uint8_t g, uint8_t b, uint16_t hold_ms)
{
    strip.fill(strip.Color(r, g, b));
    strip.show();
    delay(hold_ms);
}

// Runs to completion regardless of the button - that is the point of it:
// however brief the tap, the tester gets a full look at all three channels.
static void pattern_rgb()
{
    fill(255,   0,   0, SOLID_HOLD_MS);
    fill(  0, 255,   0, SOLID_HOLD_MS);
    fill(  0,   0, 255, SOLID_HOLD_MS);
}

// Hue gradient rolling along the strip, for as long as the button is held.
static void pattern_rainbow_while_held()
{
    uint16_t first_hue = 0;
    while (button_pressed()) {
        strip.rainbow(first_hue, 1, 255, 255, true);
        strip.show();
        first_hue += RAINBOW_HUE_STEP;
        delay(RAINBOW_FRAME_MS);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

void setup()
{
    pinMode(PIN_BUTTON, INPUT_PULLUP);
    strip.begin();
    strip.setBrightness(BRIGHTNESS);
    strip.clear();
    strip.show();
}

void loop()
{
    // Idle: dark, waiting for a press.
    if (!button_pressed()) {
        delay(1);
        return;
    }

    pattern_rgb();

    // Cycle done. If they let go during it, we are finished; if they are
    // still holding, keep going with the rainbow until they release.
    if (button_pressed())
        pattern_rainbow_while_held();

    strip.clear();
    strip.show();

    // Whichever path got us here, the button is either released or about to
    // be. Wait for the release so one long hold can't retrigger the cycle.
    while (button_pressed())
        delay(1);
}
