#include <Arduino.h>
#include "tile_transport_rp2350.h"
#include "pins.h"

// PIN_ROW_TX/PIN_ROW_RX (D6/D7) are this board's default Serial1 (UART0)
// pins, so no setTX()/setRX() remap is needed before begin() - unlike the
// Row Bus side (PIN_PI_TX/PIN_PI_RX), which will need one.

void TileTransportRP2350::init() {
    Serial1.begin(1000000, SERIAL_8N1);

    // XDIR starts low: RS-485 transceiver in RX mode.
    digitalWrite(PIN_ROW_XDIR, LOW);
    pinMode(PIN_ROW_XDIR, OUTPUT);
}

bool TileTransportRP2350::poll(FrameParser &parser, Frame *out) {
    // Drain all available RX bytes. Return true on the first complete frame.
    while (Serial1.available()) {
        uint8_t byte = (uint8_t)Serial1.read();
        if (parser.feed(byte, out)) return true;
    }
    return false;
}

void TileTransportRP2350::send(const Frame &frame) {
    uint8_t buf[MAX_FRAME_SIZE];
    int len = frame_encode(frame, buf, sizeof(buf));
    if (len <= 0) return;

    // Assert XDIR: switch transceiver to TX.
    digitalWrite(PIN_ROW_XDIR, HIGH);

    Serial1.write(buf, (size_t)len);

    // Wait for the last stop bit to leave the wire before releasing XDIR.
    Serial1.flush();

    // Return transceiver to RX mode.
    digitalWrite(PIN_ROW_XDIR, LOW);
}
