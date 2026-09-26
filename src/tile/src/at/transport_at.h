#pragma once
#include "transport.h"

#include <stdint.h>

class TransportAT : public ITransport {
public:
    void init() override;
    bool poll(FrameParser &parser, Frame *out) override;
    void send(const Frame &frame) override;

private:
    // Silence after which a part-received frame is abandoned. A maximum-size
    // frame is MAX_FRAME_SIZE bytes (187) written back to back by the row,
    // 1.87 ms at 1 Mbps, so a 5 ms gap can only fall between frames. See
    // poll().
    static constexpr uint32_t RX_IDLE_RESET_MS = 5;

    uint32_t last_rx_ms_ = 0;
};
