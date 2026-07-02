#include "command_handler.h"

const Frame *handle_command(const Frame &in, PixelBuffer &buf,
                             ISenseControl &sense, uint8_t my_addr) {
    static Frame response;

    switch (static_cast<Cmd>(in.cmd)) {

    case Cmd::SET_COLOR:
        if (in.len >= 3) {
            for (uint8_t i = 0; i < PixelBuffer::NUM_LEDS; i++)
                buf.leds[i] = {in.payload[0], in.payload[1], in.payload[2]};
        }
        return nullptr;

    case Cmd::SET_LEDS:
        if (in.len >= PixelBuffer::NUM_LEDS * 3) {
            for (uint8_t i = 0; i < PixelBuffer::NUM_LEDS; i++) {
                buf.leds[i].r = in.payload[i * 3];
                buf.leds[i].g = in.payload[i * 3 + 1];
                buf.leds[i].b = in.payload[i * 3 + 2];
            }
        }
        return nullptr;

    case Cmd::SET_PATTERN:
        // Pattern rendering deferred; payload ignored until pattern library is defined.
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

    case Cmd::TEST:
        response.addr       = my_addr;
        response.cmd        = (uint8_t)Cmd::ACK | in.cmd; // 0x84
        response.len        = 1;
        response.payload[0] = 0x00; // all tests passed
        return &response;

    default:
        return nullptr;
    }
}
