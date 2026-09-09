#include "row_bus_protocol.h"

// CRC-16/CCITT: poly 0x1021, init 0xFFFF, no reflection, no final XOR.
// Known vector: row_bus_crc16("123456789") == 0x29B1
// Identical algorithm to Tile Bus's crc16 - just a uint16_t length here
// since Row Bus payloads can be far larger than a uint8_t can index.
// Table-driven CRC-16/CCITT, matching the Tile Bus side in
// src/common/tile_bus_protocol/protocol.cpp. Every Row Bus byte passes
// through here on the way through the parser, and the bitwise version's 8
// shift/xor iterations per byte sat directly in the receive hot path that
// poll() is trying to keep under the 3.2 us/byte wire rate. Worth far less
// here than on the ATtiny - roughly 5% of the per-byte cost at 150 MHz
// against a large fraction at 20 MHz - but it is 512 bytes of flash on a
// 2 MB part, and the margin it buys is in the one loop that has none.
static const uint16_t row_bus_crc16_table[256] = {
    0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50A5, 0x60C6, 0x70E7,
    0x8108, 0x9129, 0xA14A, 0xB16B, 0xC18C, 0xD1AD, 0xE1CE, 0xF1EF,
    0x1231, 0x0210, 0x3273, 0x2252, 0x52B5, 0x4294, 0x72F7, 0x62D6,
    0x9339, 0x8318, 0xB37B, 0xA35A, 0xD3BD, 0xC39C, 0xF3FF, 0xE3DE,
    0x2462, 0x3443, 0x0420, 0x1401, 0x64E6, 0x74C7, 0x44A4, 0x5485,
    0xA56A, 0xB54B, 0x8528, 0x9509, 0xE5EE, 0xF5CF, 0xC5AC, 0xD58D,
    0x3653, 0x2672, 0x1611, 0x0630, 0x76D7, 0x66F6, 0x5695, 0x46B4,
    0xB75B, 0xA77A, 0x9719, 0x8738, 0xF7DF, 0xE7FE, 0xD79D, 0xC7BC,
    0x48C4, 0x58E5, 0x6886, 0x78A7, 0x0840, 0x1861, 0x2802, 0x3823,
    0xC9CC, 0xD9ED, 0xE98E, 0xF9AF, 0x8948, 0x9969, 0xA90A, 0xB92B,
    0x5AF5, 0x4AD4, 0x7AB7, 0x6A96, 0x1A71, 0x0A50, 0x3A33, 0x2A12,
    0xDBFD, 0xCBDC, 0xFBBF, 0xEB9E, 0x9B79, 0x8B58, 0xBB3B, 0xAB1A,
    0x6CA6, 0x7C87, 0x4CE4, 0x5CC5, 0x2C22, 0x3C03, 0x0C60, 0x1C41,
    0xEDAE, 0xFD8F, 0xCDEC, 0xDDCD, 0xAD2A, 0xBD0B, 0x8D68, 0x9D49,
    0x7E97, 0x6EB6, 0x5ED5, 0x4EF4, 0x3E13, 0x2E32, 0x1E51, 0x0E70,
    0xFF9F, 0xEFBE, 0xDFDD, 0xCFFC, 0xBF1B, 0xAF3A, 0x9F59, 0x8F78,
    0x9188, 0x81A9, 0xB1CA, 0xA1EB, 0xD10C, 0xC12D, 0xF14E, 0xE16F,
    0x1080, 0x00A1, 0x30C2, 0x20E3, 0x5004, 0x4025, 0x7046, 0x6067,
    0x83B9, 0x9398, 0xA3FB, 0xB3DA, 0xC33D, 0xD31C, 0xE37F, 0xF35E,
    0x02B1, 0x1290, 0x22F3, 0x32D2, 0x4235, 0x5214, 0x6277, 0x7256,
    0xB5EA, 0xA5CB, 0x95A8, 0x8589, 0xF56E, 0xE54F, 0xD52C, 0xC50D,
    0x34E2, 0x24C3, 0x14A0, 0x0481, 0x7466, 0x6447, 0x5424, 0x4405,
    0xA7DB, 0xB7FA, 0x8799, 0x97B8, 0xE75F, 0xF77E, 0xC71D, 0xD73C,
    0x26D3, 0x36F2, 0x0691, 0x16B0, 0x6657, 0x7676, 0x4615, 0x5634,
    0xD94C, 0xC96D, 0xF90E, 0xE92F, 0x99C8, 0x89E9, 0xB98A, 0xA9AB,
    0x5844, 0x4865, 0x7806, 0x6827, 0x18C0, 0x08E1, 0x3882, 0x28A3,
    0xCB7D, 0xDB5C, 0xEB3F, 0xFB1E, 0x8BF9, 0x9BD8, 0xABBB, 0xBB9A,
    0x4A75, 0x5A54, 0x6A37, 0x7A16, 0x0AF1, 0x1AD0, 0x2AB3, 0x3A92,
    0xFD2E, 0xED0F, 0xDD6C, 0xCD4D, 0xBDAA, 0xAD8B, 0x9DE8, 0x8DC9,
    0x7C26, 0x6C07, 0x5C64, 0x4C45, 0x3CA2, 0x2C83, 0x1CE0, 0x0CC1,
    0xEF1F, 0xFF3E, 0xCF5D, 0xDF7C, 0xAF9B, 0xBFBA, 0x8FD9, 0x9FF8,
    0x6E17, 0x7E36, 0x4E55, 0x5E74, 0x2E93, 0x3EB2, 0x0ED1, 0x1EF0,
};

static inline uint16_t row_bus_crc16_update(uint16_t crc, uint8_t byte) {
    return (uint16_t)((crc << 8) ^ row_bus_crc16_table[(uint8_t)((crc >> 8) ^ byte)]);
}

uint16_t row_bus_crc16(const uint8_t *data, uint16_t len) {
    uint16_t crc = 0xFFFF;
    for (uint16_t i = 0; i < len; i++)
        crc = row_bus_crc16_update(crc, data[i]);
    return crc;
}

int row_bus_frame_encode(const RowBusFrame &frame, uint8_t *buf, uint16_t buf_len) {
    if (frame.len > ROWBUS_MAX_PAYLOAD) return -1;
    uint16_t total = 8 + frame.len; // SYNC1+SYNC2+ADDR+CMD+LEN_H+LEN_L + payload + CRC_H+CRC_L
    if (total > buf_len) return -1;

    buf[0] = ROWBUS_SYNC1;
    buf[1] = ROWBUS_SYNC2;
    buf[2] = frame.addr;
    buf[3] = frame.cmd;
    buf[4] = (uint8_t)(frame.len >> 8);
    buf[5] = (uint8_t)(frame.len & 0xFF);
    for (uint16_t i = 0; i < frame.len; i++)
        buf[6 + i] = frame.payload[i];

    // CRC over ADDR, CMD, LEN_H, LEN_L, PAYLOAD
    uint16_t crc = row_bus_crc16(&buf[2], (uint16_t)(4 + frame.len));
    buf[6 + frame.len]     = (uint8_t)(crc >> 8);
    buf[6 + frame.len + 1] = (uint8_t)(crc & 0xFF);
    return (int)total;
}

// RowBusFrameParser feeds one byte at a time through a state machine.
// CRC is computed incrementally over ADDR+CMD+LEN_H+LEN_L+PAYLOAD so no
// extra buffer is needed on the stack.
bool RowBusFrameParser::feed(uint8_t byte, RowBusFrame *out) {
    switch (state) {
    case State::SYNC1:
        if (byte == ROWBUS_SYNC1) state = State::SYNC2;
        break;

    case State::SYNC2:
        if (byte == ROWBUS_SYNC2)        state = State::ADDR;
        else if (byte == ROWBUS_SYNC1)   state = State::SYNC2; // 0xAA 0xAA: keep waiting for 0x55
        else                             state = State::SYNC1;
        break;

    case State::ADDR:
        current.addr = byte;
        running_crc  = row_bus_crc16_update(0xFFFF, byte);
        state        = State::CMD;
        break;

    case State::CMD:
        current.cmd = byte;
        running_crc = row_bus_crc16_update(running_crc, byte);
        state       = State::LEN_H;
        break;

    case State::LEN_H:
        current.len = (uint16_t)((uint16_t)byte << 8);
        running_crc = row_bus_crc16_update(running_crc, byte);
        state       = State::LEN_L;
        break;

    case State::LEN_L:
        current.len = (uint16_t)(current.len | byte);
        running_crc = row_bus_crc16_update(running_crc, byte);
        pay_idx     = 0;
        if (current.len > ROWBUS_MAX_PAYLOAD) state = State::SYNC1;
        else if (current.len == 0)            state = State::CRC_H;
        else                                   state = State::PAYLOAD;
        break;

    case State::PAYLOAD:
        current.payload[pay_idx] = byte;
        running_crc               = row_bus_crc16_update(running_crc, byte);
        if (++pay_idx >= current.len) state = State::CRC_H;
        break;

    case State::CRC_H:
        crc_high = byte;
        state    = State::CRC_L;
        break;

    case State::CRC_L: {
        uint16_t received = (uint16_t)(((uint16_t)crc_high << 8) | byte);
        state = State::SYNC1;
        if (received != running_crc) crc_failures_++;
        if (received == running_crc) {
            *out = current;
            return true;
        }
        break;
    }
    }
    return false;
}

void RowBusFrameParser::reset() {
    state       = State::SYNC1;
    pay_idx     = 0;
    running_crc = 0;
}
