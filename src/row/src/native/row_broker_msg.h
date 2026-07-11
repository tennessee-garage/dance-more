#pragma once
#include <stdint.h>

// Socket framing between row controllers/mock-Pi and row-bus-broker.
// Every message on the socket starts with a 1-byte type, followed by a
// type-specific payload.
//
//  FRAME     0x01  [len_h:1] [len_l:1] [frame bytes:len]  — raw Row Bus frame
//  IDENTIFY  0x02  [row_addr:1]                           — sent once on connect
//
// RowBusFrame is up to ROWBUS_MAX_FRAME (976) bytes, so FRAME needs a 2-byte
// length prefix - unlike tile_bus_protocol's broker_msg.h, which fits its
// 127-byte MAX_FRAME_SIZE in one byte.
//
// Row Bus has no SENSE sideband (row addresses are static, no auto-mapping),
// so unlike broker_msg.h there are no SENSE_* message types here.
//
// row_addr identifies the client: 0x00-0x07 for a row controller, 0xFF for
// the Pi (mock-Pi test driver). The broker itself doesn't interpret this
// value - it's carried only for logging - since routing is full broadcast
// and each client filters by ADDR itself, exactly as real RS-485 hardware
// would.

enum class RowBrokerMsg : uint8_t {
    FRAME    = 0x01,
    IDENTIFY = 0x02,
};
