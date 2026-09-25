#include <Arduino.h>
#include <hardware/dma.h>
#include <hardware/gpio.h>
#include <hardware/uart.h>
#include "pi_transport_rp2350.h"
#include "pins.h"

// Row Bus baud rate (docs/row-bus-protocol.md §1). 3,125,000 is the fastest
// rate this link can actually run: the Raspberry Pi 5's RP1 UART is clocked
// at 50 MHz and a PL011 needs a divisor >= 1, so the Pi tops out at
// 50e6/16 = 3.125 Mbps. It divides exactly on both ends - Pi 50e6/16,
// RP2350 150e6/48 - so neither side accumulates baud error.
//
// Do not raise this without changing hardware. Asking for more fails
// silently rather than loudly: the Pi clamps to its maximum, the RP2350
// happily runs at whatever it was told, and the row then receives nothing
// but framing errors. (Serial2 dropped those uncounted; the DMA receive path
// passes them to the parser, where at most the odd one resembles a frame and
// fails its CRC.) The symptom is a mute bus, identical to a broken wire.
// Overridable via -DROW_BUS_BAUD for bring-up experiments.
// The value itself is in pi_transport_rp2350.h, which sizes the receive
// ring's lapping guard from it.

// Bus turnaround guard: hold the bus idle for >= 100 us after the last stop
// bit before releasing XDIR back to RX, same guard used on Tile Bus
// (docs/row-bus-protocol.md §9 "Bus turnaround timing").
static constexpr unsigned int TURNAROUND_GUARD_US = 100;

// The Row Bus runs on UART1, driven through the pico-sdk directly rather than
// arduino-pico's Serial2 - see init() for why. PIN_PI_TX/PIN_PI_RX (D9/D3)
// are not UART1's default pins, and a GPIO that UART1 cannot reach simply
// never carries the signal: gpio_set_function() does not fail, it just
// selects a function that isn't wired to this UART, so a bad pin produces a
// board that transmits and receives on unconnected pins with no error
// anywhere. Assert the valid sets (RP2350 datasheet GPIO function table,
// matching arduino-pico's SerialUART.cpp) at compile time.
static_assert(PIN_PI_TX == 4 || PIN_PI_TX == 8 || PIN_PI_TX == 20 || PIN_PI_TX == 24,
              "PIN_PI_TX must be a UART1 TX-capable GPIO (4, 8, 20, 24)");
static_assert(PIN_PI_RX == 5 || PIN_PI_RX == 9 || PIN_PI_RX == 21 || PIN_PI_RX == 25,
              "PIN_PI_RX must be a UART1 RX-capable GPIO (5, 9, 21, 25)");

static uart_inst_t *const PI_UART = uart1;

// The receive ring. The DMA channel's ring mode wraps the write address on a
// boundary of its own size, so the buffer must be aligned to that size.
alignas(PiTransportRP2350::RX_RING_SIZE) static uint8_t rx_ring_buf[PiTransportRP2350::RX_RING_SIZE];

PiTransportRP2350::PiTransportRP2350() : ring_(rx_ring_buf) {}

void PiTransportRP2350::init() {
    // Receive by DMA into a ring in RAM, not through arduino-pico's Serial2.
    //
    // Serial2 cost ~6.0 us per received byte against the 3.2 us/byte the
    // wire delivers them at. Its IRQ copies the UART FIFO into a software
    // queue, and every available() and read() then takes a mutex, disables
    // the UART IRQ, takes a second mutex and pumps the FIFO again - two such
    // round trips per byte, ~900 cycles. A row parses every byte on its
    // chain, so a chain of four maximum-size frames cost 34.9 ms of core-0
    // time per 33.3 ms frame and capped the floor at ~25 fps
    // (docs/measurements/2026-09-15-eight-row-bus-bringup.md, #99).
    //
    // Here the UART's RX DREQ paces a DMA channel that copies each byte into
    // rx_ring_buf as it arrives, wrapping forever (ENDLESS mode, RP2350
    // only). The CPU does nothing per byte to receive; poll() reads the
    // channel's write address to see how far the ring has filled and parses
    // straight out of it.
    //
    // One behaviour change from Serial2: it dropped characters with a
    // framing or parity error, whereas DMA takes the data byte regardless.
    // Such a byte now reaches the parser, fails that frame's CRC and is
    // counted - which is the more useful outcome on a noisy link.
    uart_init(PI_UART, ROW_BUS_BAUD);   // 8N1, FIFOs on, RX/TX DMA requests on
    uart_set_hw_flow(PI_UART, false, false);
    gpio_set_function(PIN_PI_TX, GPIO_FUNC_UART);
    gpio_set_function(PIN_PI_RX, GPIO_FUNC_UART);

    rx_dma_chan_ = dma_claim_unused_channel(true);
    dma_channel_config c = dma_channel_get_default_config(rx_dma_chan_);
    channel_config_set_transfer_data_size(&c, DMA_SIZE_8);
    channel_config_set_read_increment(&c, false);
    channel_config_set_write_increment(&c, true);
    channel_config_set_ring(&c, true, RX_RING_BITS);
    channel_config_set_dreq(&c, uart_get_dreq_num(PI_UART, false));
    // ENDLESS never decrements the count, but the count must still be
    // non-zero: "triggering a channel with a mode of 0xf and a count of 0x0
    // will result in the channel halting immediately" (RP2350 datasheet
    // §12.6.2.2.1). A zero count here produced a row that booted, ran both
    // cores and never received a byte.
    dma_channel_configure(rx_dma_chan_, &c, rx_ring_buf, &uart_get_hw(PI_UART)->dr,
                          (DMA_CH0_TRANS_COUNT_MODE_VALUE_ENDLESS << DMA_CH0_TRANS_COUNT_MODE_LSB) | 1u,
                          true);

    last_poll_us_ = micros();

    // XDIR starts low: RS-485 transceiver in RX mode.
    digitalWrite(PIN_PI_XDIR, LOW);
    pinMode(PIN_PI_XDIR, OUTPUT);
}

size_t PiTransportRP2350::rx_write_index() const {
    return (size_t)((uintptr_t)dma_hw->ch[rx_dma_chan_].write_addr - (uintptr_t)rx_ring_buf);
}

bool PiTransportRP2350::poll(RowBusFrameParser &parser, RowBusFrame *out) {
    const uint32_t now = micros();
    const size_t   w   = rx_write_index();

    // Lapping guard. The DMA never stops, so if the reader falls a whole ring
    // behind, the writer silently overwrites bytes not yet parsed and the
    // indices look like a nearly empty ring. Nothing on this core should
    // ever stall long enough - poll() runs every loop() pass and each pass is
    // bounded (see main.cpp) - but a guard that costs nothing beats trusting
    // that. Two signs the reader can no longer trust what is in the ring:
    //   - the backlog is within RX_RING_HEADROOM of the ring's size, or
    //   - long enough has passed since the last poll for the wire to have
    //     delivered that much, whatever the backlog now claims.
    // Either way, discard what is buffered, resync on the next frame, and
    // report it as a receive overflow. A frame or two is lost; the row is not.
    const size_t backlog = ring_.available(w);
    if (backlog > RX_RING_SIZE - RX_RING_HEADROOM ||
        (uint32_t)(now - last_poll_us_) >= RX_LAP_GUARD_US) {
        ring_.skip_to(w);
        parser.reset();
        overflow_ = true;
    }
    last_poll_us_ = now;

    if (ring_.available(w) != 0) {
        last_rx_byte_us_ = now;
    } else if (parser.in_progress() && (uint32_t)(now - last_rx_byte_us_) >= RX_IDLE_RESET_US) {
        // Recover from a truncated frame.
        //
        // The parser cannot resync on its own: mid-payload it takes every
        // byte as payload, so the next frame's SYNC1/SYNC2 is swallowed and
        // only the byte count in LEN can end the state. A frame cut short -
        // the host interrupted mid-write, or bytes lost to a brownout -
        // therefore leaves the row deaf until up to ROWBUS_MAX_PAYLOAD
        // further bytes have arrived to reach the CRC and fail it. That is
        // the cruel part: the Pi's response to a silent row is a scan, whose
        // 8-byte admin frames supply a couple of hundred bytes where 1,448
        // are owed, so the row looks permanently dead while both its cores
        // run normally and its watchdog sees nothing wrong. Observed on the
        // bench: a row unreachable across a full scan came back on the
        // second, having been fed just enough bytes by the first.
        //
        // A gap this long cannot occur inside a real frame - the Pi writes
        // one contiguously, 3.2 us per byte, and a whole maximum-size frame
        // is 4.7 ms - so this can only fire between frames.
        parser.reset();
    }

    // Parse whatever has arrived, returning at the first frame for this row.
    // Frames for other rows are CRC-checked but never returned (see
    // RowBusFrameParser::set_address()), so they are consumed here in one
    // call. A wrapped backlog takes two contiguous runs.
    const uint8_t *data;
    size_t n;
    while ((n = ring_.contiguous(w, &data)) != 0) {
        bool complete = false;
        size_t used = parser.feed(data, n, out, &complete);
        ring_.consume(used);
        rx_bytes += used;
        if (complete) {
            rx_frames++;
            return true;
        }
    }
    return false;
}

bool PiTransportRP2350::take_rx_overflow() {
    bool ovf  = overflow_;
    overflow_ = false;
    return ovf;
}

void PiTransportRP2350::send(const RowBusFrame &frame) {
    uint8_t buf[ROWBUS_MAX_FRAME];
    int len = row_bus_frame_encode(frame, buf, sizeof(buf));
    if (len <= 0) return;

    // Guard: don't key up until >= 100 us have passed since the Pi's last
    // byte, so its transceiver has fully released the line before we drive
    // it (docs/row-bus-protocol.md §9 "Bus turnaround timing"). Timed from
    // the incoming frame, not our own transmission - unsigned subtraction
    // wraps correctly across a micros() rollover.
    uint32_t elapsed = micros() - last_rx_byte_us_;
    if (elapsed < TURNAROUND_GUARD_US)
        delayMicroseconds(TURNAROUND_GUARD_US - elapsed);

    // Assert XDIR: switch transceiver to TX.
    digitalWrite(PIN_PI_XDIR, HIGH);

    uart_write_blocking(PI_UART, buf, (size_t)len);

    // Wait for the last stop bit to leave the wire, not just FIFO-empty:
    // BUSY stays set until the shift register is empty.
    uart_tx_wait_blocking(PI_UART);

    // Return transceiver to RX mode.
    digitalWrite(PIN_PI_XDIR, LOW);

    tx_frames++;
}
