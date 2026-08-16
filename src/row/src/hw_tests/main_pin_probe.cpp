#include <Arduino.h>
#include "rp2350/pins.h"

// Standalone GPIO discovery probe — answers "which GPIO is the Row Bus
// signal actually arriving on?" without trusting the variant header, the
// KiCad symbol, or the footprint's pad numbering.
//
// Every hypothesis about a mute Row Bus RX eventually rests on the chain
//   PCB net -> Xiao pad -> silkscreen Dn -> variant header -> GPIO number
// and a break anywhere in that chain looks identical from the firmware's
// side (no bytes). So don't assume any of it: configure every usable GPIO
// as a plain input, watch them all while the Pi transmits, and report which
// ones actually toggle. The pin carrying Row Bus traffic will stand out
// with thousands of transitions; everything else sits at 0.
//
// Usage:
//   1. Flash this (env:seeed_xiao_rp2350_pin_probe).
//   2. Have the Pi stream Row Bus frames continuously.
//   3. Read the report on the Tile Bus (1 Mbps ASCII), same channel the
//      ROW_DEBUG build uses — at IC2's RS-485 pair or on D6/GPIO0.
//
// Expected: exactly one pin with a high transition count = the real Row Bus
// RX pin. Compare it against PIN_PI_RX. If they differ, pins.h (or the
// board's footprint) is wrong. If NOTHING toggles, the signal isn't
// reaching the module at all and the fault is upstream of the MCU.

// GPIO0 is the debug UART's TX and GPIO3 its transceiver direction pin —
// both are outputs we're actively using, so they're excluded from scanning.
static constexpr uint8_t DEBUG_TX_GPIO   = 0;   // PIN_ROW_TX  / D6
static constexpr uint8_t DEBUG_XDIR_GPIO = 3;   // PIN_ROW_XDIR / D10
static constexpr uint8_t MAX_GPIO        = 29;  // RP2350A (Xiao) has GPIO0-29

static constexpr uint32_t SAMPLE_WINDOW_MS = 1000;

static uint32_t transitions[MAX_GPIO + 1];
static bool     last_level[MAX_GPIO + 1];

static bool scannable(uint8_t gpio) {
    return gpio != DEBUG_TX_GPIO && gpio != DEBUG_XDIR_GPIO;
}

static void debug_print(const char *fmt, ...) {
    char buf[256];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    digitalWrite(DEBUG_XDIR_GPIO, HIGH);  // key the Tile Bus transceiver
    Serial1.write((const uint8_t *)buf, strlen(buf));
    Serial1.flush();
    digitalWrite(DEBUG_XDIR_GPIO, LOW);
}

void setup() {
    Serial1.begin(1000000, SERIAL_8N1);
    digitalWrite(DEBUG_XDIR_GPIO, LOW);
    pinMode(DEBUG_XDIR_GPIO, OUTPUT);

    // Float every scannable pin as a plain input. No pull: a pull would
    // fight whatever is driving the net and could mask a real signal.
    for (uint8_t gpio = 0; gpio <= MAX_GPIO; gpio++) {
        if (!scannable(gpio)) continue;
        pinMode(gpio, INPUT);
    }

    delay(50);
    debug_print("\r\n=== GPIO pin probe ===\r\n");
    debug_print("pins.h expects Row Bus RX on GPIO%u, TX on GPIO%u\r\n",
                (unsigned)PIN_PI_RX, (unsigned)PIN_PI_TX);
    debug_print("scanning GPIO0-%u (excluding %u/%u, used for this output)\r\n",
                (unsigned)MAX_GPIO, (unsigned)DEBUG_TX_GPIO, (unsigned)DEBUG_XDIR_GPIO);
}

void loop() {
    for (uint8_t gpio = 0; gpio <= MAX_GPIO; gpio++) {
        transitions[gpio] = 0;
        if (scannable(gpio)) last_level[gpio] = digitalRead(gpio);
    }

    // Tight sample loop. This won't catch every edge of a 4 Mbps signal -
    // it doesn't need to. It only needs enough hits to make the active pin
    // obvious against pins sitting at a steady level.
    uint32_t start = millis();
    while (millis() - start < SAMPLE_WINDOW_MS) {
        for (uint8_t gpio = 0; gpio <= MAX_GPIO; gpio++) {
            if (!scannable(gpio)) continue;
            bool level = digitalRead(gpio);
            if (level != last_level[gpio]) {
                transitions[gpio]++;
                last_level[gpio] = level;
            }
        }
    }

    debug_print("--- %lus ---\r\n", (unsigned long)(millis() / 1000));
    bool any = false;
    for (uint8_t gpio = 0; gpio <= MAX_GPIO; gpio++) {
        if (!scannable(gpio) || transitions[gpio] == 0) continue;
        any = true;
        debug_print("  GPIO%-2u transitions=%lu%s\r\n", (unsigned)gpio,
                    (unsigned long)transitions[gpio],
                    gpio == PIN_PI_RX ? "   <-- pins.h says this is PIN_PI_RX" : "");
    }
    if (!any) {
        debug_print("  no activity on any scanned pin - is the Pi transmitting?\r\n");
        // Levels help distinguish "idle high" (a connected, quiet UART line)
        // from "floating/low" (nothing driving the pin at all).
        debug_print("  levels: ");
        for (uint8_t gpio = 0; gpio <= MAX_GPIO; gpio++) {
            if (!scannable(gpio)) continue;
            debug_print("%u=%u ", (unsigned)gpio, (unsigned)digitalRead(gpio));
        }
        debug_print("\r\n");
    }
}
