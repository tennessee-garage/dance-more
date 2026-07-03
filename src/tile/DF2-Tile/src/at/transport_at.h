#pragma once
#include "transport.h"

class TransportAT : public ITransport {
public:
    void init() override;
    bool poll(FrameParser &parser, Frame *out) override;
    void send(const Frame &frame) override;
};
