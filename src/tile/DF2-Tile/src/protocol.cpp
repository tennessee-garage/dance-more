#include "protocol.h"

// CRC-16/CCITT: poly 0x1021, init 0xFFFF, computed over ADDR+CMD+LEN+PAYLOAD.
// Implemented in issue #3.
uint16_t crc16(const uint8_t *, uint8_t) {
    return 0;
}

// Encodes frame into buf. Returns byte count, or -1 on overflow.
// Implemented in issue #3.
int frame_encode(const Frame &, uint8_t *, uint8_t) {
    return -1;
}

bool FrameParser::feed(uint8_t, Frame *) {
    return false;
}

void FrameParser::reset() {
    state   = State::SYNC1;
    pay_idx = 0;
}
