#pragma once
#include "pi_transport.h"
#include <stdint.h>

// Connects to row-bus-broker (tools/row-bus-broker) as a client, modeling
// this row controller's slave role on the shared multidrop Row Bus. See
// row_broker_msg.h for the socket framing.
class PiTransportNative : public IPiTransport {
    int     fd = -1;
    uint8_t row_addr;

    // Read state machine for the broker socket protocol.
    enum class RxState : uint8_t { TYPE, FRAME_LEN_H, FRAME_LEN_L, FRAME_DATA };
    RxState  rx_state           = RxState::TYPE;
    uint16_t rx_frame_len       = 0;
    uint16_t rx_frame_remaining = 0;

public:
    explicit PiTransportNative(uint8_t row_addr); // 0x00-0x07

    void init() override;  // connect to /tmp/df2-row-bus.sock, IDENTIFY(row_addr)
    bool poll(RowBusFrameParser &parser, RowBusFrame *out) override;
    void send(const RowBusFrame &frame) override;
};
