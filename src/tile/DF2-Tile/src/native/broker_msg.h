#pragma once
#include <stdint.h>

// Socket framing between tiles/row-controller and the bus broker.
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
