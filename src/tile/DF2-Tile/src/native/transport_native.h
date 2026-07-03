#pragma once
#include "transport.h"
#include "broker_msg.h"
#include <stdint.h>

class SenseNative;

class TransportNative : public ITransport {
    int         fd        = -1;
    uint8_t     tile_addr;
    uint8_t     slot;
    int         row;
    SenseNative *sense    = nullptr;

    // Read state machine for the broker socket protocol.
    enum class RxState : uint8_t { TYPE, FRAME_LEN, FRAME_DATA };
    RxState rx_state          = RxState::TYPE;
    uint8_t rx_frame_remaining = 0;

public:
    TransportNative(uint8_t tile_addr, uint8_t slot, int row);

    void set_sense(SenseNative &s) { sense = &s; }
    int  get_fd() const { return fd; }

    void init() override;
    bool poll(FrameParser &parser, Frame *out) override;
    void send(const Frame &frame) override;
};
