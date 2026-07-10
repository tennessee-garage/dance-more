#pragma once
#include "transport.h"

class TileTransportNative : public ITransport {
    int fd = -1;
    int row;

    // Read state machine for the broker socket protocol.
    enum class RxState : uint8_t { TYPE, FRAME_LEN, FRAME_DATA };
    RxState rx_state          = RxState::TYPE;
    uint8_t rx_frame_remaining = 0;

public:
    explicit TileTransportNative(int row);

    int get_fd() const { return fd; }

    void init() override;   // connect to /tmp/df2-tile-bus-<row>.sock, IDENTIFY(addr=0x00, slot=0xFF)
    bool poll(FrameParser &parser, Frame *out) override;
    void send(const Frame &frame) override;
};
