#pragma once
#include <stdint.h>
#include "protocol.h"   // src/common/tile_bus_protocol/, via lib_extra_dirs

static constexpr uint8_t  ROWBUS_SYNC1          = 0xAA;
static constexpr uint8_t  ROWBUS_SYNC2          = 0x55;
static constexpr uint8_t  ROWBUS_ADDR_BROADCAST = 0xFF;

// A SEND_DATA payload is 8 tile entries back to back, each a 1-byte tile_cmd
// followed by that command's Tile Bus payload. SET_LEDS is the largest, so it
// sets the ceiling - and it tracks LEDS_PER_TILE, which is why this derives
// from Tile Bus's MAX_PAYLOAD rather than restating a number.
static constexpr uint8_t  ROWBUS_SLOTS          = 8;
static constexpr uint16_t ROWBUS_TILE_ENTRY_MAX = 1 + MAX_PAYLOAD;                     // 181
static constexpr uint16_t ROWBUS_MAX_PAYLOAD    = ROWBUS_SLOTS * ROWBUS_TILE_ENTRY_MAX; // 1448
// SYNC1 SYNC2 ADDR CMD LEN_H LEN_L + payload + CRC_H CRC_L
static constexpr uint16_t ROWBUS_FRAME_OVERHEAD = 8;
static constexpr uint16_t ROWBUS_MAX_FRAME      = ROWBUS_FRAME_OVERHEAD + ROWBUS_MAX_PAYLOAD; // 1456

// Row Bus's LEN is 2 bytes, so there's room to spare here - but the RX FIFO in
// src/rp2350/pi_transport_rp2350.cpp is sized against ROWBUS_MAX_FRAME and an
// overrun there wedges the core (see main.cpp's watchdog note). Keep them in step.
static_assert(ROWBUS_MAX_FRAME <= 65535, "Row Bus frame no longer fits a uint16_t length");

enum class RowBusCmd : uint8_t {
    TEST           = 0x01,
    STATUS         = 0x02,
    POWER          = 0x03,
    RE_DISCOVER    = 0x04,
    ERROR_LOG      = 0x05,
    VERSION        = 0x06,
    SEND_DATA      = 0x10,
    LATCH          = 0x11,
    BLACKOUT       = 0x12,
    TEST_RESP        = 0x81,
    STATUS_RESP      = 0x82,
    POWER_RESP       = 0x83,
    RE_DISCOVER_RESP = 0x84,
    ERROR_LOG_RESP   = 0x85,
    VERSION_RESP     = 0x86,
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

    // True while a frame is only partly received.
    //
    // Mid-payload the parser consumes bytes unconditionally until it has as
    // many as LEN promised, so a following frame's SYNC1/SYNC2 is taken as
    // payload rather than recognised. A frame truncated on the wire therefore
    // leaves the row deaf until enough further bytes arrive to reach the CRC
    // and fail it - up to ROWBUS_MAX_PAYLOAD of them, which is far more than
    // the Pi's small admin frames supply while it is probing an unresponsive
    // row. Transports pair this with an idle timeout so a gap no legitimate
    // frame can contain resets the parser instead.
    bool in_progress() const { return state != State::SYNC1; }

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
