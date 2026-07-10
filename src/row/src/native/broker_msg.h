#pragma once
#include <stdint.h>

// Socket framing between tiles/row-controller and the bus broker
// (src/tile/tools/tile-bus-broker). This is a deliberate copy, not a shared
// header, per issue #34: it's a native-testing-only wire contract with a
// separate, unmodified binary, not the real Tile Bus protocol (which *is*
// shared - see src/common/tile_bus_protocol/). Keep byte-for-byte identical
// to src/tile/src/native/broker_msg.h.
//
// Every message on the socket starts with a 1-byte type, followed by a
// type-specific payload.
//
//  FRAME             0x01  [len:1] [frame bytes:len]  — raw tile-bus frame
//  IDENTIFY          0x02  [tile_addr:1] [slot:1]     — sent once on connect
//  SENSE_ASSERT      0x03  []                         — tile N → broker (sense out low)
//  SENSE_RELEASE     0x04  []                         — tile N → broker (sense out released)
//  SENSE_IS_ASSERTED 0x05  []                         — broker → tile N+1 (sense in low)
//  SENSE_DEASSERTED  0x06  []                         — broker → tile N+1 (sense in released)
//
// SENSE routing: slot N SENSE_ASSERT → slot (uint8_t)(N+1) gets SENSE_IS_ASSERTED.
// Row controller uses slot 0xFF; uint8_t wrap means its assertion reaches slot 0.

enum class BrokerMsg : uint8_t {
    FRAME             = 0x01,
    IDENTIFY          = 0x02,
    SENSE_ASSERT      = 0x03,
    SENSE_RELEASE     = 0x04,
    SENSE_IS_ASSERTED = 0x05,
    SENSE_DEASSERTED  = 0x06,
};
