#include <Arduino.h>
#include "transport_at.h"
#include "pins.h"
#include "protocol.h"

void TransportAT::init() {
    Serial.begin(1000000, SERIAL_8N1);

    // DE starts low: RS-485 transceiver in RX mode.
    digitalWrite(PIN_DE, LOW);
    pinMode(PIN_DE, OUTPUT);
}

bool TransportAT::poll(FrameParser &parser, Frame *out) {
    const uint32_t now = millis();

    if (!Serial.available()) {
        // Recover from a frame cut short.
        //
        // Mid-payload the parser takes every byte as payload until LEN is
        // satisfied, so a frame that lost bytes leaves it swallowing the
        // frames that follow - and small frames like BLACKOUT's SET_COLORs
        // never add up to enough to get it out. That is what left tiles lit
        // after `df2-pi play` exited (#104): the row's BLACKOUT reached them
        // while they were pushing their LEDs with interrupts off
        // (TILE_LED_PUSH_US in protocol.h), and only a full-size frame, often
        // not coming, would clear it. The row now keeps the Tile Bus quiet in
        // that window; this makes any other loss cost one frame, not the
        // tile. Resetting an idle parser is a no-op, so no need to track
        // whether one is mid-frame.
        if ((uint32_t)(now - last_rx_ms_) >= RX_IDLE_RESET_MS) parser.reset();
        return false;
    }

    // Drain all available RX bytes. Return true on the first complete frame.
    last_rx_ms_ = now;
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
