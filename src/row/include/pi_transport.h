#pragma once
#include "row_bus_protocol.h"

class IPiTransport {
public:
    virtual ~IPiTransport() = default;
    virtual void init() = 0;
    virtual bool poll(RowBusFrameParser &parser, RowBusFrame *out) = 0;
    virtual void send(const RowBusFrame &frame) = 0;
};
