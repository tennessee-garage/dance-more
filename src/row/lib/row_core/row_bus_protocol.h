#pragma once
#include <stdint.h>

static constexpr uint8_t  ROWBUS_SYNC1          = 0xAA;
static constexpr uint8_t  ROWBUS_SYNC2          = 0x55;
static constexpr uint8_t  ROWBUS_ADDR_BROADCAST = 0xFF;
static constexpr uint16_t ROWBUS_MAX_PAYLOAD    = 968;
static constexpr uint16_t ROWBUS_MAX_FRAME      = 976;

enum class RowBusCmd : uint8_t {
    TEST           = 0x01,
    STATUS         = 0x02,
    POWER          = 0x03,
    RE_DISCOVER    = 0x04,
    ERROR_LOG      = 0x05,
    SEND_DATA      = 0x10,
    LATCH          = 0x11,
    BLACKOUT       = 0x12,
    TEST_RESP        = 0x81,
    STATUS_RESP      = 0x82,
    POWER_RESP       = 0x83,
    RE_DISCOVER_RESP = 0x84,
    ERROR_LOG_RESP   = 0x85,
};

struct RowBusFrame {
    uint8_t  addr;
    uint8_t  cmd;
    uint16_t len;
    uint8_t  payload[ROWBUS_MAX_PAYLOAD];
};

uint16_t row_bus_crc16(const uint8_t *data, uint16_t len);

// Returns number of bytes written to buf, or -1 on overflow.
int row_bus_frame_encode(const RowBusFrame &frame, uint8_t *buf, uint16_t buf_len);

class RowBusFrameParser {
public:
    // Feed one byte; returns true (and populates *out) when a valid frame arrives.
    bool feed(uint8_t byte, RowBusFrame *out);
    void reset();

private:
    enum class State : uint8_t {
        SYNC1, SYNC2, ADDR, CMD, LEN_H, LEN_L, PAYLOAD, CRC_H, CRC_L
    };
    State       state       = State::SYNC1;
    RowBusFrame current     = {};
    uint16_t    pay_idx     = 0;
    uint8_t     crc_high    = 0;
    uint16_t    running_crc = 0;
};
