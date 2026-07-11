#include <Arduino.h>
#include "pi_transport_rp2350.h"
#include "pins.h"

// Row Bus baud rate. 4 Mbps is the documented minimum for 30 FPS headroom
// (docs/row-bus-protocol.md §1); bump to 8_000_000 here if throughput needs
// it later - the driver shape doesn't change.
static constexpr unsigned long ROW_BUS_BAUD = 4000000;

// Bus turnaround guard: hold the bus idle for >= 100 us after the last stop
// bit before releasing XDIR back to RX, same guard used on Tile Bus
// (docs/row-bus-protocol.md §9 "Bus turnaround timing").
static constexpr unsigned int TURNAROUND_GUARD_US = 100;

// PIN_PI_TX/PIN_PI_RX (D10/D9) are not this board's default Serial2 (UART1)
// pins, so they need an explicit remap before begin() - unlike the Tile Bus
// side (PIN_ROW_TX/PIN_ROW_RX), which uses Serial1's defaults as-is.

void PiTransportRP2350::init() {
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
        if (parser.feed(byte, out)) return true;
    }
    return false;
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
}
