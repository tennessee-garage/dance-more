#include "effect.h"
#include <string.h>

// PROGMEM keeps the 256-byte wave table out of the ATtiny3226's SRAM; on the
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

// Hue 0-255 once round the wheel at full saturation: pure red at 0, green
// at 85, blue at 170, linear crossfades between. Evaluated once per CHASE
// start, never per LED.
static Pixel hue_to_rgb(uint8_t hue) {
    if (hue < 85)  return {(uint8_t)(255 - hue * 3), (uint8_t)(hue * 3), 0};
    hue -= 85;
    if (hue < 85)  return {0, (uint8_t)(255 - hue * 3), (uint8_t)(hue * 3)};
    hue -= 85;
    return {(uint8_t)(hue * 3), 0, (uint8_t)(255 - hue * 3)};
}

// c * (s + 1) / 256: s = 255 is exact identity, s = 0 is exact black.
static inline uint8_t scale8(uint8_t c, uint8_t s) {
    return (uint8_t)(((uint16_t)c * (uint16_t)(s + 1)) >> 8);
}

static inline bool is_black(const Pixel &p) { return (p.r | p.g | p.b) == 0; }

bool EffectEngine::stage(const uint8_t *payload, uint8_t len) {
    if (len < EFFECT_LEN) return false;

    const uint8_t raw = payload[0];
    if (raw & ~EFFECT_ID_MASK) return false;   // reserved bits 7:5 must be 0

    const uint8_t *p = &payload[1];
    switch (raw) {
    case NONE:
    case SHIMMER:
        break;                                   // every param value is valid
    case CHASE:
        // spacing: 1 .. NUM_LEDS-1 unlit LEDs between chase LEDs.
        if (p[3] == 0 || p[3] >= NUM_LEDS) return false;
        break;
    case HUE_SPLIT:
        if (p[1] >= NUM_LEDS) return false;      // split offset
        if (p[2] || p[3]) return false;          // reserved, must be 0
        break;
    case FADE:
        if (p[1] || p[2] || p[3]) return false;  // reserved, must be 0
        break;
    default:
        return false;                            // unassigned / not implemented
    }

    staged_id_ = raw;
    memcpy(staged_params_, p, sizeof(staged_params_));
    staged_ = true;
    return true;
}

void EffectEngine::reset() {
    staged_  = false;
    id_      = NONE;
    memset(params_, 0, sizeof(params_));
    running_ = false;
}

void EffectEngine::commit(uint32_t now_ms) {
    staged_ = false;
    id_     = staged_id_;
    memcpy(params_, staged_params_, sizeof(params_));
    running_ = false;

    switch (id_) {
    case SHIMMER:
        start_shimmer();
        running_ = phase_inc_ != 0;              // speed 0 is a still frame
        next_ms_ = now_ms + FRAME_MS;
        break;
    case CHASE:
        start_chase();
        running_ = true;
        next_ms_ = now_ms + chase_period_ms_;
        break;
    case HUE_SPLIT:
        start_hue_split();
        break;
    case FADE:
        // The history is whatever the tile was already showing - out_ as it
        // stands - so arming FADE causes no visible jump.
        break;
    default:
        break;
    }
}

void EffectEngine::on_latch(const PixelBuffer &buf, uint32_t now_ms) {
    if (staged_) commit(now_ms);

    if (id_ == FADE) {
        // Follow what the host drives; leave released pixels where they are
        // and let the frame clock decay them, so the tail length doesn't
        // depend on how often the host latches. decay 0 is the exception:
        // "instant off" has to be instant, not up to a frame late.
        const bool was_running = running_;
        fade_follow(buf, params_[0] == 0);
        if (running_ && !was_running) next_ms_ = now_ms + FRAME_MS;
        return;
    }
    render(buf);
}

void EffectEngine::render(const PixelBuffer &buf) {
    switch (id_) {
    case SHIMMER:   render_shimmer(buf);   break;
    case CHASE:     render_chase(buf);     break;
    case HUE_SPLIT: render_hue_split(buf); break;
    default:
        memcpy(out_.leds, buf.leds, sizeof(out_.leds));
        break;
    }
}

bool EffectEngine::poll(const PixelBuffer &buf, uint32_t now_ms) {
    if (!running_) return false;
    if ((int32_t)(now_ms - next_ms_) < 0) return false;

    switch (id_) {
    case SHIMMER:
        // Reschedule from now, not from the missed deadline: a tile that went
        // deaf for a long WS2815 push shouldn't then burst to catch up.
        next_ms_ = now_ms + FRAME_MS;
        phase_ += phase_inc_;                    // wraps at 65536 = one cycle
        render_shimmer(buf);
        return true;

    case CHASE:
        // Accumulate rather than reschedule from now, so a period that isn't
        // a multiple of the loop's timing still averages out exactly. More
        // than a whole step behind (a long stall) resyncs instead of bursting.
        next_ms_ += chase_period_ms_;
        if ((int32_t)(now_ms - next_ms_) >= 0) next_ms_ = now_ms + chase_period_ms_;
        chase_step_ = (uint8_t)(chase_step_ + 1 == NUM_LEDS ? 0 : chase_step_ + 1);
        render_chase(buf);
        return true;

    case FADE:
        next_ms_ = now_ms + FRAME_MS;
        fade_follow(buf, true);
        return true;

    default:
        running_ = false;
        return false;
    }
}

// ---------------------------------------------------------------------------
// SHIMMER - raised-cosine brightness modulation, per-LED phase scatter
// ---------------------------------------------------------------------------

void EffectEngine::start_shimmer() {
    const uint8_t speed  = params_[0];
    const uint8_t spread = params_[2];
    const uint8_t seed   = params_[3];

    // 65536 phase units per cycle, EFFECT_HZ frames per second, speed in
    // cycles per minute: 65536 / (60 * 50) = 21.85, rounded to 22 (+0.7%).
    phase_inc_ = (uint16_t)speed * 22;
    phase_     = 0;

    // Seed 0 is xorshift's fixed point; map it to an arbitrary non-zero state
    // so that seed 0 is still a usable (and reproducible) scatter.
    uint16_t s = seed ? (uint16_t)(0x1000 | (seed << 4) | seed) : 0xACE1;
    for (uint8_t i = 0; i < NUM_LEDS; i++) {
        const uint8_t raw = (uint8_t)(xs16(s) >> 8);
        offset_[i] = (uint8_t)(((uint16_t)raw * spread) >> 8);
    }
}

void EffectEngine::render_shimmer(const PixelBuffer &buf) {
    const uint8_t depth = params_[1];
    const uint8_t theta = (uint8_t)(phase_ >> 8);

    for (uint8_t i = 0; i < NUM_LEDS; i++) {
        const uint8_t w = wave8((uint8_t)(theta + offset_[i]));
        // depth 0 -> scale pinned at 255 (buffer untouched);
        // depth 255 -> scale swings the full 255..0. The buffer is the
        // ceiling: SHIMMER only ever darkens, so it can't clip.
        const uint8_t scale =
            (uint8_t)(255 - (uint8_t)(((uint16_t)depth * (uint16_t)(255 - w)) >> 8));

        out_.leds[i].r = scale8(buf.leds[i].r, scale);
        out_.leds[i].g = scale8(buf.leds[i].g, scale);
        out_.leds[i].b = scale8(buf.leds[i].b, scale);
    }
}

// ---------------------------------------------------------------------------
// CHASE - evenly spaced lit LEDs stepping round the perimeter
// ---------------------------------------------------------------------------

void EffectEngine::start_chase() {
    const Pixel c = hue_to_rgb(params_[0]);
    const uint8_t bri = params_[1];
    chase_rgb_       = {scale8(c.r, bri), scale8(c.g, bri), scale8(c.b, bri)};
    chase_period_ms_ = (uint16_t)(CHASE_BASE_MS + (uint16_t)params_[2] * CHASE_STEP_MS);
    chase_step_      = 0;
}

void EffectEngine::render_chase(const PixelBuffer &buf) {
    memcpy(out_.leds, buf.leds, sizeof(out_.leds));

    // LED i is lit when (i - step) is a multiple of the period. Walking the
    // lit positions directly costs one modulo per render instead of one per
    // LED - the AVR has no hardware divide.
    const uint8_t period = (uint8_t)(params_[3] + 1);
    for (uint8_t i = (uint8_t)(chase_step_ % period); i < NUM_LEDS; i = (uint8_t)(i + period))
        out_.leds[i] = chase_rgb_;
}

// ---------------------------------------------------------------------------
// HUE_SPLIT - opposite hue shifts on two halves, in RGB space
// ---------------------------------------------------------------------------
//
// A hue rotation by a third of a turn (85 of 256) is exactly a cyclic
// rotation of the channels: red -> green -> blue -> red. Anything between two
// such rotations is approximated by a linear mix of them. That preserves
// r + g + b (so brightness, to within rounding), leaves greys and black
// exactly untouched, and costs two multiplies per channel with no divide -
// where a real RGB->HSV->RGB round trip would need several per LED.
//
// A negative shift is the same thing the other way round: -s == 256 - s.

// kSrc[k][j]: which input channel lands in output channel j after k
// one-third rotations toward increasing hue.
static const uint8_t kSrc[3][3] = {{0, 1, 2}, {2, 0, 1}, {1, 2, 0}};

void EffectEngine::start_hue_split() {
    const uint8_t shift = params_[0];
    const uint8_t amount[2] = {shift, (uint8_t)(256 - shift)};   // + half, - half
    for (uint8_t h = 0; h < 2; h++) {
        const uint8_t q   = amount[h] / 85;                      // 0..3
        const uint8_t rem = (uint8_t)(amount[h] - q * 85);       // 0..84
        split_rot_[h] = (uint8_t)(q % 3);
        split_mix_[h] = (uint8_t)(rem * 3);                      // /256 ~= /85.3
    }
}

void EffectEngine::render_hue_split(const PixelBuffer &buf) {
    const uint8_t offset = params_[1];
    for (uint8_t i = 0; i < NUM_LEDS; i++) {
        // + half is LEDs offset .. offset + NUM_LEDS/2 - 1, wrapping.
        const uint8_t j = (uint8_t)(i >= offset ? i - offset : i + NUM_LEDS - offset);
        const uint8_t h = j < NUM_LEDS / 2 ? 0 : 1;

        const uint8_t  k0 = split_rot_[h];
        const uint8_t  k1 = k0 == 2 ? 0 : (uint8_t)(k0 + 1);
        const uint16_t w  = split_mix_[h];
        const uint8_t  c[3] = {buf.leds[i].r, buf.leds[i].g, buf.leds[i].b};

        uint8_t o[3];
        for (uint8_t ch = 0; ch < 3; ch++)
            o[ch] = (uint8_t)(((uint16_t)c[kSrc[k0][ch]] * (256 - w) +
                               (uint16_t)c[kSrc[k1][ch]] * w) >> 8);
        out_.leds[i] = {o[0], o[1], o[2]};
    }
}

// ---------------------------------------------------------------------------
// FADE - released pixels decay instead of cutting
// ---------------------------------------------------------------------------
//
// A plain floor, v * k >> 8 with k <= 255, is strictly less than v for every
// v > 0, so a released pixel always reaches exactly black. Don't "improve" it
// with rounding: (v * k + 128) >> 8 stalls at 1 for high k.

void EffectEngine::fade_follow(const PixelBuffer &buf, bool decay_released) {
    const uint8_t k = params_[0];
    bool any_lit_released = false;

    for (uint8_t i = 0; i < NUM_LEDS; i++) {
        Pixel &o = out_.leds[i];
        if (!is_black(buf.leds[i])) {
            o = buf.leds[i];                     // host is driving it: follow exactly
            continue;
        }
        if (decay_released) {
            o.r = (uint8_t)(((uint16_t)o.r * k) >> 8);
            o.g = (uint8_t)(((uint16_t)o.g * k) >> 8);
            o.b = (uint8_t)(((uint16_t)o.b * k) >> 8);
        }
        if (!is_black(o)) any_lit_released = true;
    }

    // Nothing left to decay: stop the frame clock until a LATCH releases
    // something again, so an idle FADE tile doesn't push for nothing.
    running_ = any_lit_released;
}
