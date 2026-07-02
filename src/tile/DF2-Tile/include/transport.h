#pragma once
#include "protocol.h"

class ITransport {
public:
    virtual ~ITransport() = default;
    virtual void init() = 0;
    // Drain available RX bytes and feed each to parser.
    virtual void poll(FrameParser &parser) = 0;
    virtual void send(const Frame &frame) = 0;
};
