#pragma once
#include <stddef.h>
#include <stdint.h>
#include "pi_transport.h"
#include "rx_ring.h"

// See pi_transport_rp2350.cpp before changing this.
#ifndef ROW_BUS_BAUD
#define ROW_BUS_BAUD 3125000UL
#endif

class PiTransportRP2350 : public IPiTransport {
public:
    PiTransportRP2350();
    void init() override;
    bool poll(RowBusFrameParser &parser, RowBusFrame *out) override;
    void send(const RowBusFrame &frame) override;

    // Read-and-clear: true if Row Bus bytes may have been lost since the
    // last call - the receive ring was discarded because the reader fell too
    // far behind the DMA writing into it (see poll()). Some frame then failed
    // CRC or vanished. Surfacing it turns "the floor
    // glitched" into a specific, logged cause - see main.cpp, which hands
    // this to core 1 for the error log.
    bool take_rx_overflow();

    // Diagnostic counters, written by poll()/send() on core 0 and read by
    // main.cpp's ROW_DEBUG reporting on core 1. Separating "raw bytes seen"
    // from "frames that passed CRC" is what distinguishes a dead RX pin
    // from a framing/baud/CRC problem - the two failure modes look
    // identical from the Pi's end (silence either way).
    volatile uint32_t rx_bytes  = 0;
    volatile uint32_t rx_frames = 0;   // frames returned: this row's and broadcasts
    volatile uint32_t tx_frames = 0;

    // 16 KB of receive ring: ~52 ms of continuous traffic at 3.125 Mbps,
    // against a reader that is never more than one loop() pass behind. A
    // power of two, because the DMA ring wraps on one.
    static constexpr unsigned RX_RING_BITS = 14;
    static constexpr size_t   RX_RING_SIZE = (size_t)1 << RX_RING_BITS;
    static_assert(RX_RING_SIZE >= 4 * ROWBUS_MAX_FRAME, "ring must hold a chain's worth of maximum-size frames");

private:
    // How close the backlog may come to a full ring, and how long poll() may
    // go uncalled, before the ring can no longer be trusted not to have been
    // lapped. The time guard is the wire time of RX_RING_SIZE - HEADROOM
    // bytes: ~39 ms at 3.125 Mbps, orders of magnitude past any real gap
    // between polls.
    static constexpr size_t   RX_RING_HEADROOM = RX_RING_SIZE / 4;
    static constexpr uint32_t RX_LAP_GUARD_US  =
        (uint32_t)((RX_RING_SIZE - RX_RING_HEADROOM) * 10ULL * 1000000ULL / ROW_BUS_BAUD);

    size_t rx_write_index() const;

    RxRing<RX_RING_SIZE> ring_;
    unsigned             rx_dma_chan_  = 0;
    uint32_t             last_poll_us_ = 0;
    bool                 overflow_     = false;

    // Idle gap after which a part-received frame is abandoned. 20 ms is ~4x
    // the 4.7 ms a maximum-size frame takes on the wire, so it cannot fire
    // mid-frame, and it is short enough that a stranded row recovers within
    // one frame period rather than needing a power cycle. See poll().
    static constexpr uint32_t RX_IDLE_RESET_US = 20000;

    // micros() timestamp of the last byte received from the Pi, so send()
    // can wait out the turnaround guard from the *incoming* frame's last
    // stop bit, not from our own transmission.
    uint32_t last_rx_byte_us_ = 0;
};
