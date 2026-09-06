#pragma once
// protocol.h now lives in src/common/tile_bus_protocol/ (shared with src/row/);
// a relative include is needed here since PlatformIO's LDF doesn't propagate
// lib_extra_dirs include paths into other lib/ folders' own compile steps.
#include "../../../common/tile_bus_protocol/protocol.h"
#include "pixel_buffer.h"
#include "pattern.h"
#include "sense.h"

// Dispatch a received frame to the pixel buffer and/or sense control.
// my_addr is this tile's unicast address; used when building response frames.
// Caller must pre-filter: only call when frame.addr == my_addr or ADDR_BROADCAST.
// pattern is optional: pass the tile's PatternEngine to enable SET_PATTERN and
// the pattern-cancelling side effect of SET_COLOR/SET_LEDS. When it is nullptr
// (harnesses and tests that don't exercise patterns) SET_PATTERN stays the
// no-op it was before the pattern library existed.
// Returns a pointer to a statically-allocated response frame, or nullptr if no
// response is needed.
const Frame *handle_command(const Frame &in, PixelBuffer &buf,
                             ISenseControl &sense, uint8_t my_addr,
                             PatternEngine *pattern = nullptr);
