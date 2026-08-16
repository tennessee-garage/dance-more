#include <Arduino.h>
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
// happily runs at whatever it was told, and every byte then dies as a
// framing error inside the core's UART IRQ handler, which drops bad chars
// without counting them. The symptom is a totally mute bus, identical to a
// broken wire. Overridable via -DROW_BUS_BAUD for bring-up experiments.
#ifndef ROW_BUS_BAUD
#define ROW_BUS_BAUD 3125000UL
#endif

// Bus turnaround guard: hold the bus idle for >= 100 us after the last stop
// bit before releasing XDIR back to RX, same guard used on Tile Bus
// (docs/row-bus-protocol.md §9 "Bus turnaround timing").
static constexpr unsigned int TURNAROUND_GUARD_US = 100;

// PIN_PI_TX/PIN_PI_RX (D9/D3) are not this board's default Serial2 (UART1)
// pins, so they need an explicit remap before begin() - unlike the Tile Bus
// side (PIN_ROW_TX/PIN_ROW_RX), which uses Serial1's defaults as-is.
//
// setTX()/setRX() return false for a GPIO that UART1 can't reach and
// begin() then quietly uses the variant defaults instead, so a bad remap
// produces a board that transmits and receives on unconnected pins with no
// error anywhere. Assert the valid sets (SerialUART.cpp) at compile time.
static_assert(PIN_PI_TX == 4 || PIN_PI_TX == 8 || PIN_PI_TX == 20 || PIN_PI_TX == 24,
              "PIN_PI_TX must be a UART1 TX-capable GPIO (4, 8, 20, 24)");
static_assert(PIN_PI_RX == 5 || PIN_PI_RX == 9 || PIN_PI_RX == 21 || PIN_PI_RX == 25,
              "PIN_PI_RX must be a UART1 RX-capable GPIO (5, 9, 21, 25)");

void PiTransportRP2350::init() {
    // Size the RX FIFO to hold a whole maximum-size frame.
    //
    // arduino-pico's SerialUART defaults to a 32-byte software FIFO - about
    // 100 us of buffering at 3.125 Mbps. A full SEND_DATA frame is 976 bytes
    // arriving back-to-back over 3.1 ms, some 30x longer than that window,
    // and poll() below costs two _pumpFIFO() round trips per byte (one in
    // available(), one in read()) against a 3.2 us/byte arrival rate. Core 0
    // cannot keep up, the FIFO overruns, and the core wedges hard enough to
    // need a power cycle - not merely dropped bytes.
    //
    // Measured on the bench: continuous frames died above ~170 bytes, while
    // the same 976-byte frame paced at 16 bytes per 400 us survived intact.
    // That is what makes this a receive-rate problem rather than a frame-size
    // one, and why the fix is buffer depth rather than anything in the parser.
    //
    // Sized for one max frame plus the LATCH and admin command that can
    // legitimately follow it back-to-back in the same burst. Must precede
    // begin(): setFIFOSize() returns false once the port is running, and it
    // is begin() that actually allocates the buffer.
    Serial2.setFIFOSize(2048);
    Serial2.setTX(PIN_PI_TX);
    Serial2.setRX(PIN_PI_RX);
    Serial2.begin(ROW_BUS_BAUD, SERIAL_8N1);

    // XDIR starts low: RS-485 transceiver in RX mode.
    digitalWrite(PIN_PI_XDIR, LOW);
    pinMode(PIN_PI_XDIR, OUTPUT);
}

bool PiTransportRP2350::poll(RowBusFrameParser &parser, RowBusFrame *out) {
    // Drain all available RX bytes. Return true on the first complete frame.
    while (Serial2.available()) {
        uint8_t byte = (uint8_t)Serial2.read();
        last_rx_byte_us_ = micros();
        rx_bytes++;
        if (parser.feed(byte, out)) {
            rx_frames++;
            return true;
        }
    }
    return false;
}

bool PiTransportRP2350::take_rx_overflow() {
    // SerialUART::overflow() is itself read-and-clear.
    return Serial2.overflow();
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

    Serial2.write(buf, (size_t)len);

    // Wait for the last stop bit to leave the wire, not just FIFO-empty.
    Serial2.flush();

    // Return transceiver to RX mode.
    digitalWrite(PIN_PI_XDIR, LOW);

    tx_frames++;
}
