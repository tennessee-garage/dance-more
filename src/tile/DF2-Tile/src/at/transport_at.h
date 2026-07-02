#pragma once
#include "transport.h"

class TransportAT : public ITransport {
public:
    void init() override;
    void poll(FrameParser &parser) override;
    void send(const Frame &frame) override;
};
