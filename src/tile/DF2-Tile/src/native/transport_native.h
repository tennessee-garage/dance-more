#pragma once
#include "transport.h"

class TransportNative : public ITransport {
    int fd = -1;
    uint8_t tile_addr;
    uint8_t slot;
    int     row;
public:
    TransportNative(uint8_t tile_addr, uint8_t slot, int row);
    void init() override;
    void poll(FrameParser &parser) override;
    void send(const Frame &frame) override;
};
