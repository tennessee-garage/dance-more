#pragma once
#include <stdint.h>
// Relative include for the same reason as command_handler.h: PlatformIO's LDF
// doesn't propagate lib_extra_dirs include paths into other lib/ folders.
#include "../../../common/tile_bus_protocol/protocol.h"
#include "pixel_buffer.h"

// Tile-local pattern rendering. See docs/tile-patterns.md for the wire-level
// definition of each pattern and its params.
//
// The point of rendering here rather than on the Pi is bandwidth, not compute:
// a full-pixel floor frame is ~11.6 kB and the Row Bus ceiling is 3.125 Mbps
// (docs/row-bus-protocol.md §1), so pixel frames alone run the bus at ~90%.
// A SET_PATTERN is 12 bytes once, after which the tile animates on its own and
// the bus goes quiet. That also decouples the animation rate from the frame
// rate - the tile renders at PATTERN_HZ regardless of what the host is doing.
class PatternEngine {
public:
    // Local render rate. 20 ms leaves the WS2815 push (~1.8 ms, interrupts
    // disabled - see LedDriverAT::push) at ~9% of the period, so the tile
    // stays responsive to Tile Bus traffic while a pattern runs. The ceiling
    // is the push time itself; this is deliberately well short of it.
    static constexpr uint8_t  PATTERN_HZ = 50;
    static constexpr uint8_t  FRAME_MS   = 1000 / PATTERN_HZ;  // 20

    static constexpr uint8_t  PATTERN_ID_MASK  = 0x1F;  // bits 4:0
    static constexpr uint8_t  PATTERN_LEN      = 5;     // id + 4 params

    enum Id : uint8_t {
        OFF     = 0,
        SOLID   = 1,
        SHIMMER = 3,
    };

    // Stage a pattern from a SET_PATTERN payload. Returns false - leaving any
    // running pattern untouched - if the payload is short, the reserved bits
    // are set, or the id isn't implemented. The pattern does not begin until
    // the next on_latch(), so a row's tiles can be armed one at a time and
    // then started together by the broadcast LATCH.
    bool arm(const uint8_t *payload, uint8_t len);

    // LATCH handler. Starts whatever arm() staged, using the buffer's current
    // contents as the pattern's base image. No-op if nothing is armed, so the
    // host's normal per-frame LATCH doesn't restart a running pattern.
    void on_latch(PixelBuffer &buf, uint32_t now_ms);

    // Explicit pixel data (SET_COLOR / SET_LEDS) supersedes a pattern.
    void cancel() { armed_ = false; active_ = false; }

    bool active() const { return active_; }
    bool armed()  const { return armed_;  }

    // Renders the next frame into buf when FRAME_MS has elapsed. Returns true
    // when buf was updated and the caller should push it to the strip.
    bool poll(PixelBuffer &buf, uint32_t now_ms);

private:
    void start_shimmer(const PixelBuffer &buf, uint32_t now_ms);
    void render_shimmer(PixelBuffer &buf);

    uint8_t  id_       = OFF;
    uint8_t  params_[4] = {0, 0, 0, 0};
    bool     armed_    = false;
    bool     active_   = false;

    // Base image the pattern modulates - snapshotted once at on_latch() so
    // that rendering into buf doesn't feed back into the next frame.
    Pixel    base_[PixelBuffer::NUM_LEDS] = {};
    uint8_t  offset_[PixelBuffer::NUM_LEDS] = {};  // per-LED phase, spread-scaled

    uint16_t phase_        = 0;
    uint16_t phase_inc_    = 0;
    uint32_t next_frame_ms_ = 0;
};
