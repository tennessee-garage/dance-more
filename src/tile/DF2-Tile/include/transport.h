#pragma once
#include "protocol.h"

class ITransport {
public:
    virtual ~ITransport() = default;
    virtual void init() = 0;
    // Consume one available byte and feed it to parser. Returns true (and fills
    // *out) when a complete valid frame is received. Call in a loop until false
    // to drain all buffered input.
    virtual bool poll(FrameParser &parser, Frame *out) = 0;
    virtual void send(const Frame &frame) = 0;
};
