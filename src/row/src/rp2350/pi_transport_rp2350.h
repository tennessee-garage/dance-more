#pragma once
#include "pi_transport.h"

class PiTransportRP2350 : public IPiTransport {
public:
    void init() override;
    bool poll(RowBusFrameParser &parser, RowBusFrame *out) override;
    void send(const RowBusFrame &frame) override;
};
