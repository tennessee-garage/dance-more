#pragma once
#include <stdint.h>
#include "pi_transport.h"

class PiTransportRP2350 : public IPiTransport {
public:
    void init() override;
    bool poll(RowBusFrameParser &parser, RowBusFrame *out) override;
    void send(const RowBusFrame &frame) override;

private:
    // micros() timestamp of the last byte received from the Pi, so send()
    // can wait out the turnaround guard from the *incoming* frame's last
    // stop bit, not from our own transmission.
    uint32_t last_rx_byte_us_ = 0;
};
