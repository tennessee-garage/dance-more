#include "command_handler.h"
#include "fw_version_info.h"

const Frame *handle_command(const Frame &in, PixelBuffer &buf,
                             ISenseControl &sense, uint8_t &my_addr,
                             PatternEngine *pattern) {
    static Frame response;

    switch (static_cast<Cmd>(in.cmd)) {

    case Cmd::SET_COLOR:
        // Explicit pixel data always wins - the host must be able to take a
        // tile back from a running pattern.
        if (pattern) pattern->cancel();
        if (in.len >= 3) {
            for (uint8_t i = 0; i < PixelBuffer::NUM_LEDS; i++)
                buf.leds[i] = {in.payload[0], in.payload[1], in.payload[2]};
        }
        return nullptr;

    case Cmd::SET_LEDS:
        if (pattern) pattern->cancel();
        if (in.len >= PixelBuffer::NUM_LEDS * 3) {
            for (uint8_t i = 0; i < PixelBuffer::NUM_LEDS; i++) {
                buf.leds[i].r = in.payload[i * 3];
                buf.leds[i].g = in.payload[i * 3 + 1];
                buf.leds[i].b = in.payload[i * 3 + 2];
            }
        }
        return nullptr;

    case Cmd::SET_PATTERN:
        // Staged, not started: the pattern begins on the next LATCH so a row's
        // tiles can be armed one at a time and started together. Malformed or
        // unimplemented ids are ignored - display commands carry no ACK, so
        // there is nothing to report and the tile keeps what it has.
        // See docs/tile-patterns.md.
        if (pattern) pattern->arm(in.payload, in.len);
        return nullptr;

    case Cmd::LATCH:
        buf.latch_pending = true;
        return nullptr;

    case Cmd::ACTIVATE_SENSE:
        sense.assert_sense_out();
        response.addr       = my_addr;
        response.cmd        = (uint8_t)Cmd::ACK | in.cmd; // 0x81
        response.len        = 1;
        response.payload[0] = 0x00; // success
        return &response;

    case Cmd::CLEAR_SENSE:
        sense.release_sense_out();
        return nullptr;

    case Cmd::DETECT_SENSE:
        if (sense.sense_is_asserted()) {
            response.addr = my_addr;
            response.cmd  = (uint8_t)Cmd::DETECT_RESP;
            response.len  = 0;
            return &response;
        }
        return nullptr;

    case Cmd::SET_ADDRESS:
        // Broadcast, but answered by exactly one tile: the SENSE walk has
        // asserted precisely one tile's incoming line at this point, so the
        // row can hand an address to a tile that does not yet have one - the
        // bootstrap that address-based unicast cannot do for itself. A tile
        // whose SENSE_IN is not asserted must stay silent, or every tile on
        // the bus would take the same address at once.
        if (!sense.sense_is_asserted()) return nullptr;
        if (in.len < 1) return nullptr;
        if (in.payload[0] == ADDR_UNASSIGNED || in.payload[0] == ADDR_BROADCAST)
            return nullptr;  // neither is a usable unicast address (§3)

        my_addr = in.payload[0];
        // Answering *from* the new address is the acknowledgement that
        // matters: it proves the assignment took, rather than only that the
        // command was received.
        response.addr       = my_addr;
        response.cmd        = (uint8_t)Cmd::ACK | in.cmd; // 0x86
        response.len        = 1;
        response.payload[0] = 0x00; // success
        return &response;

    case Cmd::TEST:
        response.addr       = my_addr;
        response.cmd        = (uint8_t)Cmd::ACK | in.cmd; // 0x84
        response.len        = 1;
        response.payload[0] = 0x00; // all tests passed
        return &response;

    case Cmd::VERSION:
        response.addr = my_addr;
        response.cmd  = (uint8_t)Cmd::VERSION_RESP;
        response.len  = FW_VERSION_WIRE_SIZE;
        fw_version_encode(tile_fw_version(), response.payload);
        return &response;

    default:
        return nullptr;
    }
}
