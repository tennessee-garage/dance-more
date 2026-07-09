#include "protocol.h"

// CRC-16/CCITT: poly 0x1021, init 0xFFFF, no reflection, no final XOR.
// Known vector: crc16("123456789") == 0x29B1
static uint16_t crc16_update(uint16_t crc, uint8_t byte) {
    crc ^= (uint16_t)byte << 8;
    for (uint8_t i = 0; i < 8; i++)
        crc = (crc & 0x8000) ? (crc << 1) ^ 0x1021 : (crc << 1);
    return crc;
}

uint16_t crc16(const uint8_t *data, uint8_t len) {
    uint16_t crc = 0xFFFF;
    for (uint8_t i = 0; i < len; i++)
        crc = crc16_update(crc, data[i]);
    return crc;
}

int frame_encode(const Frame &frame, uint8_t *buf, uint8_t buf_len) {
    if (frame.len > MAX_PAYLOAD) return -1;
    uint8_t total = 7 + frame.len; // SYNC1+SYNC2+ADDR+CMD+LEN + payload + CRC_H+CRC_L
    if (total > buf_len) return -1;

    buf[0] = PROTO_SYNC1;
    buf[1] = PROTO_SYNC2;
    buf[2] = frame.addr;
    buf[3] = frame.cmd;
    buf[4] = frame.len;
    for (uint8_t i = 0; i < frame.len; i++)
        buf[5 + i] = frame.payload[i];

    // CRC over ADDR, CMD, LEN, PAYLOAD
    uint16_t crc = crc16(&buf[2], 3 + frame.len);
    buf[5 + frame.len]     = (uint8_t)(crc >> 8);
    buf[5 + frame.len + 1] = (uint8_t)(crc & 0xFF);
    return total;
}

// FrameParser feeds one byte at a time through a state machine.
// CRC is computed incrementally over ADDR+CMD+LEN+PAYLOAD so no
// extra buffer is needed on the stack.
bool FrameParser::feed(uint8_t byte, Frame *out) {
    switch (state) {
    case State::SYNC1:
        if (byte == PROTO_SYNC1) state = State::SYNC2;
        break;

    case State::SYNC2:
        if (byte == PROTO_SYNC2)      state = State::ADDR;
        else if (byte == PROTO_SYNC1) state = State::SYNC2; // 0xAA 0xAA: keep waiting for 0x55
        else                          state = State::SYNC1;
        break;

    case State::ADDR:
        current.addr = byte;
        running_crc  = crc16_update(0xFFFF, byte);
        state        = State::CMD;
        break;

    case State::CMD:
        current.cmd = byte;
        running_crc = crc16_update(running_crc, byte);
        state       = State::LEN;
        break;

    case State::LEN:
        current.len = byte;
        running_crc = crc16_update(running_crc, byte);
        pay_idx     = 0;
        if (byte > MAX_PAYLOAD)  state = State::SYNC1;
        else if (byte == 0)      state = State::CRC_H;
        else                     state = State::PAYLOAD;
        break;

    case State::PAYLOAD:
        current.payload[pay_idx] = byte;
        running_crc              = crc16_update(running_crc, byte);
        if (++pay_idx >= current.len) state = State::CRC_H;
        break;

    case State::CRC_H:
        crc_high = byte;
        state    = State::CRC_L;
        break;

    case State::CRC_L: {
        uint16_t received = ((uint16_t)crc_high << 8) | byte;
        state = State::SYNC1;
        if (received == running_crc) {
            *out = current;
            return true;
        }
        break;
    }
    }
    return false;
}

void FrameParser::reset() {
    state       = State::SYNC1;
    pay_idx     = 0;
    running_crc = 0;
}
