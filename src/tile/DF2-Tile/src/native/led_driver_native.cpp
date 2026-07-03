#include "led_driver_native.h"
#include <cstdio>
#include <unistd.h>

LedDriverNative::LedDriverNative(uint8_t addr) : tile_addr(addr) {}

void LedDriverNative::init() {
    printf("[tile 0x%02X] LED driver ready (40 LEDs, native/log mode)\n", tile_addr);
}

void LedDriverNative::push(const PixelBuffer &buf) {
    ++frame_count;

    bool uniform = true;
    for (uint8_t i = 1; i < PixelBuffer::NUM_LEDS; ++i) {
        if (buf.leds[i].r != buf.leds[0].r ||
            buf.leds[i].g != buf.leds[0].g ||
            buf.leds[i].b != buf.leds[0].b) {
            uniform = false;
            break;
        }
    }

    if (uniform) {
        printf("[tile 0x%02X | frame %u] #%02X%02X%02X \xC3\x97 %u\n",
               tile_addr, frame_count,
               buf.leds[0].r, buf.leds[0].g, buf.leds[0].b,
               (unsigned)PixelBuffer::NUM_LEDS);
    } else {
        printf("[tile 0x%02X | frame %u]", tile_addr, frame_count);
        for (uint8_t i = 0; i < PixelBuffer::NUM_LEDS; ++i)
            printf(" led[%02u]=#%02X%02X%02X", i,
                   buf.leds[i].r, buf.leds[i].g, buf.leds[i].b);
        printf("\n");
    }
    fflush(stdout);

    if (!isatty(STDOUT_FILENO)) return;

    // TTY: draw a 12-row × 21-col square border to stderr mirroring the
    // physical perimeter layout.  LEDs run clockwise:
    //   top row    left→right  : leds[0..9]
    //   right col  top→bottom  : leds[10..19]
    //   bottom row right→left  : leds[20..29]
    //   left col   bottom→top  : leds[30..39]
    //
    // Top/bottom rows: ' ' LED ' ' LED ' ' ... LED ' '  (21 chars)
    // Middle rows:     LED  <19 spaces>  LED             (21 chars)

    auto block = [](const Pixel &p) {
        fprintf(stderr, "\e[38;2;%u;%u;%um\xe2\x96\x88", p.r, p.g, p.b);
    };

    // Top row: leds[0..9]
    fprintf(stderr, " ");
    for (int i = 0; i < 10; ++i) { block(buf.leds[i]); fprintf(stderr, " "); }
    fprintf(stderr, "\n");

    // Middle rows 1-10:
    //   left  (top→bottom): leds[39], leds[38], ..., leds[30]
    //   right (top→bottom): leds[10], leds[11], ..., leds[19]
    for (int r = 1; r <= 10; ++r) {
        block(buf.leds[40 - r]);
        fprintf(stderr, "                   ");   // 19 spaces
        block(buf.leds[9 + r]);
        fprintf(stderr, "\n");
    }

    // Bottom row: leds[29..20] left→right
    fprintf(stderr, " ");
    for (int i = 29; i >= 20; --i) { block(buf.leds[i]); fprintf(stderr, " "); }
    fprintf(stderr, "\e[0m\n\n");
}
