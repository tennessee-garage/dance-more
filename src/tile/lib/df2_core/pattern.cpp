#include "pattern.h"
#include <string.h>

// PROGMEM keeps the 256-byte wave table out of the ATtiny3224's SRAM; on the
// native build the macros collapse to plain memory access.
#if defined(__AVR__)
#include <avr/pgmspace.h>
#define DF2_PROGMEM PROGMEM
#define DF2_READ_U8(p) pgm_read_byte(p)
#else
#define DF2_PROGMEM
#define DF2_READ_U8(p) (*(p))
#endif

// Raised cosine over a full turn: 127.5 * (1 - cos(2*pi*i/256)), rounded.
// A table rather than a computed approximation because the AVR has no
// hardware divide and this is evaluated 60 times per frame.
static const uint8_t kWave[256] DF2_PROGMEM = {
      0,   0,   0,   0,   1,   1,   1,   2,   2,   3,   4,   5,   5,   6,   7,   9,
     10,  11,  12,  14,  15,  17,  18,  20,  21,  23,  25,  27,  29,  31,  33,  35,
     37,  40,  42,  44,  47,  49,  52,  54,  57,  59,  62,  65,  67,  70,  73,  76,
     79,  82,  85,  88,  90,  93,  97, 100, 103, 106, 109, 112, 115, 118, 121, 124,
    127, 131, 134, 137, 140, 143, 146, 149, 152, 155, 158, 162, 165, 167, 170, 173,
    176, 179, 182, 185, 188, 190, 193, 196, 198, 201, 203, 206, 208, 211, 213, 215,
    218, 220, 222, 224, 226, 228, 230, 232, 234, 235, 237, 238, 240, 241, 243, 244,
    245, 246, 248, 249, 250, 250, 251, 252, 253, 253, 254, 254, 254, 255, 255, 255,
    255, 255, 255, 255, 254, 254, 254, 253, 253, 252, 251, 250, 250, 249, 248, 246,
    245, 244, 243, 241, 240, 238, 237, 235, 234, 232, 230, 228, 226, 224, 222, 220,
    218, 215, 213, 211, 208, 206, 203, 201, 198, 196, 193, 190, 188, 185, 182, 179,
    176, 173, 170, 167, 165, 162, 158, 155, 152, 149, 146, 143, 140, 137, 134, 131,
    128, 124, 121, 118, 115, 112, 109, 106, 103, 100,  97,  93,  90,  88,  85,  82,
     79,  76,  73,  70,  67,  65,  62,  59,  57,  54,  52,  49,  47,  44,  42,  40,
     37,  35,  33,  31,  29,  27,  25,  23,  21,  20,  18,  17,  15,  14,  12,  11,
     10,   9,   7,   6,   5,   5,   4,   3,   2,   2,   1,   1,   1,   0,   0,   0,
};

static inline uint8_t wave8(uint8_t theta) { return DF2_READ_U8(&kWave[theta]); }

// xorshift16 (7, 9, 8) - full period over the 65535 non-zero states. Used only
// to scatter per-LED phase, so the quality bar is "looks unpatterned", but the
// seed has to be reproducible: the same seed on two tiles gives two tiles that
// shimmer in step, and different seeds de-correlate them.
static inline uint16_t xs16(uint16_t &s) {
    s ^= (uint16_t)(s << 7);
    s ^= (uint16_t)(s >> 9);
    s ^= (uint16_t)(s << 8);
    return s;
}

bool PatternEngine::arm(const uint8_t *payload, uint8_t len) {
    if (len < PATTERN_LEN) return false;

    const uint8_t raw = payload[0];
    if (raw & ~PATTERN_ID_MASK) return false;   // reserved bits 7:5 must be 0

    switch (raw) {
    case OFF:
    case SOLID:
    case SHIMMER:
        break;
    default:
        return false;                            // reserved / not implemented
    }

    id_ = raw;
    memcpy(params_, &payload[1], sizeof(params_));
    armed_ = true;
    return true;
}

void PatternEngine::on_latch(PixelBuffer &buf, uint32_t now_ms) {
    if (!armed_) return;
    armed_ = false;

    switch (id_) {
    case OFF:
        // Static: paint once on this LATCH's push, then stop rendering.
        for (uint8_t i = 0; i < PixelBuffer::NUM_LEDS; i++)
            buf.leds[i] = {0, 0, 0};
        active_ = false;
        return;

    case SOLID:
        for (uint8_t i = 0; i < PixelBuffer::NUM_LEDS; i++)
            buf.leds[i] = {params_[0], params_[1], params_[2]};
        active_ = false;
        return;

    case SHIMMER:
        start_shimmer(buf, now_ms);
        return;
    }
}

void PatternEngine::start_shimmer(const PixelBuffer &buf, uint32_t now_ms) {
    const uint8_t speed  = params_[0];
    const uint8_t spread = params_[2];
    const uint8_t seed   = params_[3];

    memcpy(base_, buf.leds, sizeof(base_));

    // 65536 phase units per cycle, PATTERN_HZ frames per second, speed in
    // cycles per minute: 65536 / (60 * 50) = 21.85, rounded to 22 (+0.7%).
    phase_inc_ = (uint16_t)speed * 22;
    phase_     = 0;

    // Seed 0 is xorshift's fixed point; map it to an arbitrary non-zero state
    // so that seed 0 is still a usable (and reproducible) scatter.
    uint16_t s = seed ? (uint16_t)(0x1000 | (seed << 4) | seed) : 0xACE1;
    for (uint8_t i = 0; i < PixelBuffer::NUM_LEDS; i++) {
        const uint8_t raw = (uint8_t)(xs16(s) >> 8);
        offset_[i] = (uint8_t)(((uint16_t)raw * spread) >> 8);
    }

    next_frame_ms_ = now_ms + FRAME_MS;
    active_        = true;
}

void PatternEngine::render_shimmer(PixelBuffer &buf) {
    const uint8_t depth = params_[1];
    const uint8_t theta = (uint8_t)(phase_ >> 8);

    for (uint8_t i = 0; i < PixelBuffer::NUM_LEDS; i++) {
        const uint8_t w = wave8((uint8_t)(theta + offset_[i]));
        // depth 0 -> scale pinned at 255 (base untouched);
        // depth 255 -> scale swings the full 255..0. The base image is the
        // ceiling: a pattern only ever darkens, so it can't clip.
        const uint8_t scale =
            (uint8_t)(255 - (uint8_t)(((uint16_t)depth * (uint16_t)(255 - w)) >> 8));

        // >>8 rather than /255: one part in 256 dark at full scale, invisible,
        // and it keeps this to a multiply and a shift per channel.
        buf.leds[i].r = (uint8_t)(((uint16_t)base_[i].r * scale) >> 8);
        buf.leds[i].g = (uint8_t)(((uint16_t)base_[i].g * scale) >> 8);
        buf.leds[i].b = (uint8_t)(((uint16_t)base_[i].b * scale) >> 8);
    }

    phase_ += phase_inc_;   // wraps at 65536 = one cycle
}

bool PatternEngine::poll(PixelBuffer &buf, uint32_t now_ms) {
    if (!active_) return false;
    if ((int32_t)(now_ms - next_frame_ms_) < 0) return false;

    // Reschedule from now, not from the missed deadline: a tile that went deaf
    // for a long WS2815 push shouldn't then burst to catch up.
    next_frame_ms_ = now_ms + FRAME_MS;

    switch (id_) {
    case SHIMMER:
        render_shimmer(buf);
        return true;
    default:
        // Only continuously-rendered patterns should ever leave active_ set.
        active_ = false;
        return false;
    }
}
