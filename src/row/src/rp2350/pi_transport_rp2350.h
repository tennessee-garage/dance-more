#pragma once
#include <stdint.h>
#include "pi_transport.h"

class PiTransportRP2350 : public IPiTransport {
public:
    void init() override;
    bool poll(RowBusFrameParser &parser, RowBusFrame *out) override;
    void send(const RowBusFrame &frame) override;

    // Read-and-clear: true if Serial2's RX FIFO has overflowed since the
    // last call. An overflow means Row Bus bytes were silently dropped, so
    // some frame failed CRC and vanished. Surfacing it turns "the floor
    // glitched" into a specific, logged cause - see main.cpp, which hands
    // this to core 1 for the error log.
    bool take_rx_overflow();

    // Diagnostic counters, written by poll()/send() on core 0 and read by
    // main.cpp's ROW_DEBUG reporting on core 1. Separating "raw bytes seen"
    // from "frames that passed CRC" is what distinguishes a dead RX pin
    // from a framing/baud/CRC problem - the two failure modes look
    // identical from the Pi's end (silence either way).
    volatile uint32_t rx_bytes  = 0;
    volatile uint32_t rx_frames = 0;
    volatile uint32_t tx_frames = 0;

private:
    // micros() timestamp of the last byte received from the Pi, so send()
    // can wait out the turnaround guard from the *incoming* frame's last
    // stop bit, not from our own transmission.
    uint32_t last_rx_byte_us_ = 0;
};
