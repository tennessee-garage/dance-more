#pragma once
#include <stdint.h>

static constexpr uint8_t PROTO_SYNC1    = 0xAA;
static constexpr uint8_t PROTO_SYNC2    = 0x55;
static constexpr uint8_t ADDR_BROADCAST = 0xFF;
// A tile holds this until the row assigns it one during the SENSE walk. It is
// the "Reserved" address of docs/tile-bus-protocol.md §3, so no frame is ever
// legitimately sent to it and an unaddressed tile answers only broadcasts -
// which is all it needs, since DETECT_SENSE and SET_ADDRESS are both
// broadcast. Addresses are assigned by position and need only be unique on
// one Tile Bus: the floor's 8 rows are 8 separate buses that never join.
static constexpr uint8_t ADDR_UNASSIGNED = 0x00;
// ---- LED geometry ----
// A build specification, not a preference: the strip is 300 LED/5m and the
// wooden frame's ledge gives ~10" of the tile's 15" side to lay it on, so
// 15 per side is what physically fits. See docs/hardware-tile.md's "LED
// layout and chain order". Both wire protocols size themselves from here -
// Tile Bus's MAX_PAYLOAD below, Row Bus's ROWBUS_MAX_PAYLOAD in
// src/row/lib/row_core/row_bus_protocol.h.
static constexpr uint8_t LEDS_PER_SIDE = 15;
static constexpr uint8_t LEDS_PER_TILE = 4 * LEDS_PER_SIDE;  // 60, corners dark
static constexpr uint8_t BYTES_PER_LED = 3;                  // RGB888

// SYNC1 SYNC2 ADDR CMD LEN + payload + CRC_H CRC_L
static constexpr uint8_t FRAME_OVERHEAD = 7;
static constexpr uint8_t MAX_PAYLOAD    = LEDS_PER_TILE * BYTES_PER_LED;   // 180
static constexpr uint8_t MAX_FRAME_SIZE = FRAME_OVERHEAD + MAX_PAYLOAD;    // 187

// LEN is one byte on Tile Bus, and frame_encode()/crc16() take uint8_t
// lengths - so the whole frame, not just the payload, has to stay under 256.
// Written against LEDS_PER_SIDE rather than MAX_PAYLOAD because the latter is
// itself a uint8_t and would silently wrap before the check ran. At 3 bytes
// per LED this caps a side at 20; going past that means a 2-byte LEN on Tile
// Bus, or fewer bytes per LED (RGB565).
static_assert(FRAME_OVERHEAD + 4 * LEDS_PER_SIDE * BYTES_PER_LED <= 255,
              "Tile Bus frame no longer fits a uint8_t length");

enum class Cmd : uint8_t {
    // Commands: row controller → tile
    ACTIVATE_SENSE = 0x01,
    DETECT_SENSE   = 0x02,
    CLEAR_SENSE    = 0x03,
    TEST           = 0x04,
    VERSION        = 0x05,
    SET_ADDRESS    = 0x06,
    SET_COLOR      = 0x10,
    SET_PATTERN    = 0x11,
    SET_LEDS       = 0x12,
    LATCH          = 0x13,
    // Responses: tile → row controller
    ACK            = 0x80,
    DETECT_RESP    = 0x82,
    VERSION_RESP   = 0x85,
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
    State    state       = State::SYNC1;
    Frame    current     = {};
    uint8_t  pay_idx     = 0;
    uint8_t  crc_high    = 0;
    uint16_t running_crc = 0;
};
