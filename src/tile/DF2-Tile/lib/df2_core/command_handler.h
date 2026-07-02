#pragma once
#include "protocol.h"
#include "pixel_buffer.h"
#include "sense.h"

// Dispatch a received frame to the pixel buffer and/or sense control.
// my_addr is this tile's unicast address; used when building response frames.
// Caller must pre-filter: only call when frame.addr == my_addr or ADDR_BROADCAST.
// Returns a pointer to a statically-allocated response frame, or nullptr if no
// response is needed.
const Frame *handle_command(const Frame &in, PixelBuffer &buf,
                             ISenseControl &sense, uint8_t my_addr);
