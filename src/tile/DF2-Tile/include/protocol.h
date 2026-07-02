#pragma once
#include <stdint.h>

static constexpr uint8_t PROTO_SYNC1    = 0xAA;
static constexpr uint8_t PROTO_SYNC2    = 0x55;
static constexpr uint8_t ADDR_BROADCAST = 0xFF;
static constexpr uint8_t MAX_PAYLOAD    = 120;
static constexpr uint8_t MAX_FRAME_SIZE = 127;

enum class Cmd : uint8_t {
    // Commands: row controller → tile
    ACTIVATE_SENSE = 0x01,
    DETECT_SENSE   = 0x02,
    CLEAR_SENSE    = 0x03,
    TEST           = 0x04,
    SET_COLOR      = 0x10,
    SET_PATTERN    = 0x11,
    SET_LEDS       = 0x12,
    LATCH          = 0x13,
    // Responses: tile → row controller
    ACK            = 0x80,
    DETECT_RESP    = 0x82,
};

struct Frame {
    uint8_t addr;
    uint8_t cmd;
    uint8_t len;
    uint8_t payload[MAX_PAYLOAD];
};

uint16_t crc16(const uint8_t *data, uint8_t len);

// Returns number of bytes written to buf, or -1 on overflow.
int frame_encode(const Frame &frame, uint8_t *buf, uint8_t buf_len);

class FrameParser {
public:
    // Feed one byte; returns true (and populates *out) when a valid frame arrives.
    bool feed(uint8_t byte, Frame *out);
    void reset();

private:
    enum class State : uint8_t {
        SYNC1, SYNC2, ADDR, CMD, LEN, PAYLOAD, CRC_H, CRC_L
    };
    State   state    = State::SYNC1;
    Frame   current  = {};
    uint8_t pay_idx  = 0;
    uint8_t crc_high = 0;
};
