#include "row_command_handler.h"

RowCommandHandler::RowCommandHandler(ITransport &tile_transport, SenseMapper &sense,
                                      IPowerMonitor &power, uint8_t my_row_addr)
    : tile_transport_(tile_transport), sense_(sense), power_(power), my_row_addr_(my_row_addr) {}

void RowCommandHandler::send_tile_frame(uint8_t addr, Cmd cmd, const uint8_t *payload, uint8_t len) {
    Frame f{};
    f.addr = addr;
    f.cmd  = (uint8_t)cmd;
    f.len  = len;
    for (uint8_t i = 0; i < len; i++) f.payload[i] = payload[i];
    tile_transport_.send(f);
}

void RowCommandHandler::broadcast_tile_latch() {
    Frame f{};
    f.addr = ADDR_BROADCAST;
    f.cmd  = (uint8_t)Cmd::LATCH;
    f.len  = 0;
    tile_transport_.send(f);
}

// Stub: real self-test (Row Bus UART loopback, Tile Bus transceiver, SRAM)
// needs platform hooks that don't exist yet.
void RowCommandHandler::handle_test() {
    response_.addr = my_row_addr_;
    response_.cmd  = (uint8_t)RowBusCmd::TEST_RESP;
    response_.len  = 2;
    response_.payload[0] = 0x00; // result: all pass
    response_.payload[1] = 0x00; // fault_flags: none
}

void RowCommandHandler::handle_status() {
    const TileMap &map = sense_.result();

    response_.addr = my_row_addr_;
    response_.cmd  = (uint8_t)RowBusCmd::STATUS_RESP;
    response_.len  = 10;
    // SenseMapState's declaration order (IDLE, DISCOVERING, DONE, ERROR)
    // already matches the wire's 0x00-0x03 (idle/discovering/running/error).
    response_.payload[0] = (uint8_t)sense_.state();
    response_.payload[1] = map.discovered_count();
    for (uint8_t slot = 0; slot < TileMap::NUM_SLOTS; slot++)
        response_.payload[2 + slot] = (uint8_t)map.status_for(slot);
}

void RowCommandHandler::handle_power() {
    PowerReading reading = power_.read();

    response_.addr = my_row_addr_;
    response_.cmd  = (uint8_t)RowBusCmd::POWER_RESP;
    response_.len  = 6;
    response_.payload[0] = (uint8_t)(reading.voltage_mV >> 8);
    response_.payload[1] = (uint8_t)(reading.voltage_mV & 0xFF);
    response_.payload[2] = (uint8_t)(reading.current_mA >> 8);
    response_.payload[3] = (uint8_t)(reading.current_mA & 0xFF);
    response_.payload[4] = (uint8_t)(reading.power_mW >> 8);
    response_.payload[5] = (uint8_t)(reading.power_mW & 0xFF);
}

void RowCommandHandler::handle_re_discover() {
    sense_.start(); // proceeds asynchronously via the caller's regular sense_.poll() loop

    response_.addr = my_row_addr_;
    response_.cmd  = (uint8_t)RowBusCmd::RE_DISCOVER_RESP;
    response_.len  = 1;
    response_.payload[0] = 0x00; // started
}

// Stub: the 32-entry ring buffer this needs doesn't have a producer yet -
// wire this up once Tile Bus retry logic exists in concrete transports.
void RowCommandHandler::handle_error_log() {
    response_.addr = my_row_addr_;
    response_.cmd  = (uint8_t)RowBusCmd::ERROR_LOG_RESP;
    response_.len  = 1;
    response_.payload[0] = 0x00; // entry_count
}

void RowCommandHandler::handle_send_data(const RowBusFrame &in) {
    const TileMap &map = sense_.result();
    uint16_t offset = 0;

    for (uint8_t slot = 0; slot < TileMap::NUM_SLOTS; slot++) {
        if (offset >= in.len) break; // short/malformed frame - stop rather than read garbage

        uint8_t tile_cmd = in.payload[offset];
        uint8_t data_len;
        switch ((Cmd)tile_cmd) {
        case Cmd::SET_COLOR:   data_len = 3;   break;
        case Cmd::SET_PATTERN: data_len = 5;   break;
        case Cmd::SET_LEDS:    data_len = 120; break;
        default: return; // unrecognized tile_cmd - can't know its size, stop parsing
        }

        if (map.is_discovered(slot))
            send_tile_frame(map.address_for(slot), (Cmd)tile_cmd, &in.payload[offset + 1], data_len);
        // else: slot not discovered - skip forwarding, still advance offset below

        offset = (uint16_t)(offset + 1 + data_len);
    }
}

// TODO(#46): overrun buffering - defer this LATCH if still forwarding a
// SEND_DATA that hasn't finished, per docs/row-bus-protocol.md's
// LATCH_OVERRUN handling. Not tracked yet; needs real Tile Bus forwarding
// timing to design against.
void RowCommandHandler::handle_latch() {
    broadcast_tile_latch();
}

void RowCommandHandler::handle_blackout() {
    const TileMap &map = sense_.result();
    const uint8_t black[3] = {0, 0, 0};

    for (uint8_t slot = 0; slot < TileMap::NUM_SLOTS; slot++) {
        if (!map.is_discovered(slot)) continue;
        send_tile_frame(map.address_for(slot), Cmd::SET_COLOR, black, sizeof(black));
    }
    broadcast_tile_latch();
}

const RowBusFrame *RowCommandHandler::handle(const RowBusFrame &in) {
    switch ((RowBusCmd)in.cmd) {
    case RowBusCmd::TEST:         handle_test();         return &response_;
    case RowBusCmd::STATUS:       handle_status();       return &response_;
    case RowBusCmd::POWER:        handle_power();        return &response_;
    case RowBusCmd::RE_DISCOVER:  handle_re_discover();  return &response_;
    case RowBusCmd::ERROR_LOG:    handle_error_log();    return &response_;
    case RowBusCmd::SEND_DATA:    handle_send_data(in);  return nullptr;
    case RowBusCmd::LATCH:        handle_latch();        return nullptr;
    case RowBusCmd::BLACKOUT:     handle_blackout();     return nullptr;
    default:                                              return nullptr;
    }
}
