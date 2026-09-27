#pragma once
#include <stdint.h>
// Relative include for the same reason as command_handler.h: PlatformIO's LDF
// doesn't propagate lib_extra_dirs include paths into other lib/ folders.
#include "../../../common/tile_bus_protocol/protocol.h"
#include "pixel_buffer.h"

// The tile's effect register: a transform applied over the host's pixel
// buffer on its way to the LEDs. See docs/tile-effects.md for the wire-level
// definition of each effect and its params.
//
// The pixel buffer and the effect register are independent. SET_COLOR /
// SET_LEDS write only the buffer; SET_EFFECT writes only the register. The
// LEDs always show buffer -> effect -> output(), and the effect never writes
// the buffer, so the host can change content under a running effect without
// re-arming it.
//
// The point of rendering here rather than on the Pi is bandwidth, not compute:
// a full-pixel floor frame is ~11.6 kB and the Row Bus ceiling is 3.125 Mbps
// (docs/row-bus-protocol.md §1), so pixel frames alone run the bus at ~90%.
// A SET_EFFECT is 12 bytes once, after which the tile animates on its own and
// the bus goes quiet. That also decouples the animation rate from the frame
// rate - the tile renders on its own clock regardless of what the host does.
class EffectEngine {
public:
    // Local render rate for frame-clocked effects (SHIMMER, FADE).
    //
    // Every render the tile pushes itself is a WS2815 push with interrupts
    // off (TILE_LED_PUSH_US, ~1.8 ms), during which the tile cannot receive:
    // ~9% of this period, and anything the row sends in that window is lost.
    // Measured on the bench (#106): with CHASE running, 5% of small frames,
    // 20% of full SET_LEDS frames and ~9% of BLACKOUTs went missing. The row
    // cannot avoid these pushes - they are on the tile's clock, not after a
    // LATCH - so the tile stops making them while the host is streaming; see
    // host_streaming().
    static constexpr uint8_t  EFFECT_HZ = 50;
    static constexpr uint8_t  FRAME_MS  = 1000 / EFFECT_HZ;  // 20

    // How close together two LATCHes must be for the host to count as
    // streaming frames: 10 fps or faster. Below that, or once the host has
    // gone quiet this long, the tile renders on its own clock again.
    static constexpr uint8_t  STREAM_HOLD_MS = 100;

    static constexpr uint8_t  EFFECT_ID_MASK = 0x1F;  // bits 4:0
    static constexpr uint8_t  EFFECT_LEN     = 5;     // id + 4 params

    // CHASE step period: 40 ms + 10 ms per unit of `speed`.
    static constexpr uint16_t CHASE_BASE_MS  = 40;
    static constexpr uint8_t  CHASE_STEP_MS  = 10;

    static constexpr uint8_t  NUM_LEDS = PixelBuffer::NUM_LEDS;
    static_assert(NUM_LEDS % 2 == 0, "HUE_SPLIT splits the tile into equal halves");

    // Ids 1 and 2 (formerly SOLID and BREATHE) and 5 (SPARKLE, undefined)
    // are deliberately absent: stage() rejects them.
    enum Id : uint8_t {
        NONE      = 0,
        SHIMMER   = 3,
        CHASE     = 4,
        HUE_SPLIT = 6,
        FADE      = 7,
    };

    // SET_EFFECT handler. Validates and stages the effect; it takes over on
    // the next on_latch(), so a row's tiles can be armed one at a time and
    // started together by the broadcast LATCH. Returns false - leaving both
    // the running and any previously staged effect untouched - if the
    // payload is short, reserved bits are set, the id isn't implemented, or
    // a param is outside the range the effect defines.
    bool stage(const uint8_t *payload, uint8_t len);

    // LATCH handler. Commits a staged effect (restarting its clock), then
    // renders the buffer through the current effect into output(). Always
    // renders: new buffer contents must reach the LEDs on every LATCH.
    void on_latch(const PixelBuffer &buf, uint32_t now_ms);

    // Advances a time-varying effect. Returns true when output() changed and
    // the caller should push it to the strip; false when there is nothing to
    // do, which is always the case for NONE and HUE_SPLIT.
    bool poll(const PixelBuffer &buf, uint32_t now_ms);

    // What the LEDs should show.
    const PixelBuffer &output() const { return out_; }

    // True while the host is streaming frames: the last two LATCHes came
    // within STREAM_HOLD_MS of each other, and the latest within
    // STREAM_HOLD_MS of now. The caller should then push only on LATCH -
    // inside the window the row keeps the Tile Bus quiet (TILE_LATCH_QUIET_MS)
    // - and not when poll() reports a change. poll() still advances the
    // effect's clock meanwhile, so the next on_latch() renders it where it
    // should be and the animation keeps time; it just updates at the host's
    // frame rate instead of 50 Hz.
    //
    // One LATCH on its own is not streaming: SET_EFFECT + LATCH and then
    // silence - the usual way to start an effect - free-runs straight away.
    bool host_streaming(uint32_t now_ms) const;

    // Clears the register, anything staged, and all private state. Power-up
    // state; also used on re-addressing. Does not touch output(): the LEDs
    // change on the next LATCH, like any other register write.
    void reset();

    uint8_t id()     const { return id_; }
    bool    staged() const { return staged_; }

private:
    void commit(uint32_t now_ms);
    void render(const PixelBuffer &buf);   // current state, no clock advance

    void start_shimmer();
    void render_shimmer(const PixelBuffer &buf);
    void start_chase();
    void render_chase(const PixelBuffer &buf);
    void start_hue_split();
    void render_hue_split(const PixelBuffer &buf);
    void fade_follow(const PixelBuffer &buf, bool decay_released);

    // Staged by SET_EFFECT, committed by LATCH.
    bool     staged_           = false;
    uint8_t  staged_id_        = NONE;
    uint8_t  staged_params_[4] = {0, 0, 0, 0};

    // The register.
    uint8_t  id_        = NONE;
    uint8_t  params_[4] = {0, 0, 0, 0};

    // Output buffer. For FADE it is also the history the decay runs on.
    PixelBuffer out_ = {};

    // Frame clock (SHIMMER, FADE) and step clock (CHASE).
    uint32_t next_ms_ = 0;
    bool     running_ = false;   // poll() has work to do

    // SHIMMER
    uint8_t  offset_[NUM_LEDS] = {};  // per-LED phase, spread-scaled
    uint16_t phase_     = 0;
    uint16_t phase_inc_ = 0;

    // CHASE
    Pixel    chase_rgb_    = {0, 0, 0};
    uint16_t chase_period_ms_ = 0;
    uint8_t  chase_step_   = 0;      // 0 .. NUM_LEDS-1

    // HUE_SPLIT: per half, whole channel rotations (0-2) and mix weight /256.
    uint8_t  split_rot_[2] = {0, 0};
    uint8_t  split_mix_[2] = {0, 0};

    // The last two LATCHes, for host_streaming().
    uint8_t  latches_seen_  = 0;     // saturates at 2
    uint32_t last_latch_ms_ = 0;
    uint32_t prev_latch_ms_ = 0;
};
