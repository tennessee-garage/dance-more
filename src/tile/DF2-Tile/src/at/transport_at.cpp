#include <Arduino.h>
#include "transport_at.h"
#include "pins.h"
#include "protocol.h"

void TransportAT::init() {
    // Select USART0 pin mapping before opening the port.
    Serial.swap(UART0_SWAP);
    Serial.begin(1000000, SERIAL_8N1);

    // DE starts low: RS-485 transceiver in RX mode.
    digitalWrite(PIN_DE, LOW);
    pinMode(PIN_DE, OUTPUT);
}

bool TransportAT::poll(FrameParser &parser, Frame *out) {
    // Drain all available RX bytes. Return true on the first complete frame.
    while (Serial.available()) {
        uint8_t byte = (uint8_t)Serial.read();
        if (parser.feed(byte, out)) return true;
    }
    return false;
}

void TransportAT::send(const Frame &frame) {
    uint8_t buf[MAX_FRAME_SIZE];
    int len = frame_encode(frame, buf, sizeof(buf));
    if (len <= 0) return;

    // Assert DE: switch transceiver to TX.
    digitalWrite(PIN_DE, HIGH);

    Serial.write(buf, (size_t)len);

    // Wait for the last stop bit to leave the wire before releasing DE.
    // megaTinyCore Serial::flush() polls USART_TXCIF_bm, not just DREIF.
    Serial.flush();

    // Return transceiver to RX mode.
    digitalWrite(PIN_DE, LOW);
}
