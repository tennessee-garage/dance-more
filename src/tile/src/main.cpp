#include <Arduino.h>
#include "transport.h"
#include "led_driver.h"
#include "sense.h"
#include "command_handler.h"
#include "at/transport_at.h"
#include "at/led_driver_at.h"
#include "at/sense_at.h"

// Assigned by the row during its SENSE walk, not stored anywhere: the walk
// runs at every boot and derives the address from physical position, so a
// tile is interchangeable with any other and a replacement needs no
// programming. Until then it answers broadcasts only, which is enough to
// receive DETECT_SENSE and SET_ADDRESS. See docs/tile-bus-protocol.md §3.
static uint8_t my_addr = ADDR_UNASSIGNED;

static TransportAT transport;
static LedDriverAT led_driver;
static SenseAT     sense;
static PixelBuffer pixel_buf;
static FrameParser parser;
static PatternEngine pattern;

// ---- Start-up pattern ----
// Six 600 ms colour steps (red/green/blue, twice) confirming the MCU booted
// and can drive the strip.
//
// This used to run as blocking delay() calls inside setup(), which left the
// tile deaf to the Tile Bus for its first ~3.7 s - two orders of magnitude
// longer than the row controller's boot discovery sweep, which finishes about
// 35 ms after power-up. Every tile was therefore still in setup() while the
// only discovery of the boot went past, so on a cold power-up no row ever
// found any tiles.
//
// Driven from loop() instead. The tile parses Tile Bus frames throughout, so
// it answers DETECT_SENSE while the pattern is still playing, and the
// diagnostic keeps its full run time without costing anything. Real display
// data cancels it, so a LATCH is never fought over by the start-up pattern.
//
// Note this does not make the tile continuously listening: led_driver.push()
// bit-bangs the WS2815 line with interrupts off for ~1.2 ms, so there is a
// brief deaf window per step (and per latch, in normal operation too). The
// Tile Bus retries in SenseMapper cover it.
static constexpr uint8_t  STARTUP_STEPS   = 6;
static constexpr uint16_t STARTUP_STEP_MS = 600;

static PixelBuffer startup_buf;
static uint8_t     startup_step    = 0;
static uint32_t    startup_next_ms = 0;
static bool        startup_done    = false;

static void startup_pattern_poll(uint32_t now_ms) {
    if (startup_done) return;
    if ((int32_t)(now_ms - startup_next_ms) < 0) return;

    if (startup_step >= STARTUP_STEPS) {
        led_driver.clear();
        startup_done = true;
        return;
    }

    static const uint8_t kColours[3][3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}};
    const uint8_t *c = kColours[startup_step % 3];
    for (uint8_t i = 0; i < PixelBuffer::NUM_LEDS; i++)
        startup_buf.leds[i] = {c[0], c[1], c[2]};
    led_driver.push(startup_buf);

    startup_step++;
    startup_next_ms = now_ms + STARTUP_STEP_MS;
}

void setup() {
    sense.init();
    transport.init();

    delay(100);

    led_driver.init();
}

void loop() {
    Frame f;
    if (transport.poll(parser, &f)) {
        // An unassigned tile must not match on address: ADDR_UNASSIGNED is
        // reserved and never legitimately targeted, but matching it would
        // make every unaddressed tile on the bus answer the same frame.
        const bool for_us = (my_addr != ADDR_UNASSIGNED && f.addr == my_addr);
        if (for_us || f.addr == ADDR_BROADCAST) {
            const Frame *resp = handle_command(f, pixel_buf, sense, my_addr, &pattern);
            if (resp) transport.send(*resp);
        }
    }

    const uint32_t now = millis();

    if (pixel_buf.latch_pending) {
        // Real display data supersedes the start-up diagnostic.
        startup_done = true;
        // Before the push, so a pattern armed since the last LATCH snapshots
        // this frame as its base and starts in step with the rest of the row.
        pattern.on_latch(pixel_buf, now);
        led_driver.push(pixel_buf);
        pixel_buf.latch_pending = false;
    } else if (pattern.poll(pixel_buf, now)) {
        // A running pattern drives the strip on its own clock - no Tile Bus
        // traffic per frame, and no LATCH needed. See docs/tile-patterns.md.
        startup_done = true;
        led_driver.push(pixel_buf);
    } else {
        startup_pattern_poll(now);
    }
}
