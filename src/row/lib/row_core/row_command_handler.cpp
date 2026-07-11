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

void RowCommandHandler::handle_error_log() {
    response_.addr = my_row_addr_;
    response_.cmd  = (uint8_t)RowBusCmd::ERROR_LOG_RESP;
    response_.len  = (uint16_t)(1 + error_log_count_ * 5);
    response_.payload[0] = error_log_count_;

    // Oldest first: if the buffer hasn't wrapped yet, entry 0 is oldest and
    // sits at index 0; once full, the oldest is whatever error_log_next_ is
    // about to overwrite.
    uint8_t start = (error_log_count_ == ERROR_LOG_CAPACITY) ? error_log_next_ : 0;
    for (uint8_t i = 0; i < error_log_count_; i++) {
        const ErrorLogEntry &e = error_log_[(start + i) % ERROR_LOG_CAPACITY];
        uint8_t *p = &response_.payload[1 + i * 5];
        p[0] = e.slot;
        p[1] = e.tile_bus_cmd;
        p[2] = e.error_type;
        p[3] = (uint8_t)(e.timestamp_s >> 8);
        p[4] = (uint8_t)(e.timestamp_s & 0xFF);
    }
}

void RowCommandHandler::log_error(uint8_t slot, uint8_t tile_bus_cmd, uint8_t error_type, uint32_t now_ms) {
    ErrorLogEntry &e = error_log_[error_log_next_];
    e.slot         = slot;
    e.tile_bus_cmd = tile_bus_cmd;
    e.error_type   = error_type;
    e.timestamp_s  = (uint16_t)(now_ms / 1000);

    error_log_next_ = (uint8_t)((error_log_next_ + 1) % ERROR_LOG_CAPACITY);
    if (error_log_count_ < ERROR_LOG_CAPACITY) error_log_count_++;
}

bool RowCommandHandler::parse_entry(uint16_t offset, uint8_t *tile_cmd_out, uint8_t *data_len_out) const {
    if (offset >= forwarding_frame_.len) return false; // short/malformed frame

    uint8_t tile_cmd = forwarding_frame_.payload[offset];
    uint8_t data_len;
    switch ((Cmd)tile_cmd) {
    case Cmd::SET_COLOR:   data_len = 3;   break;
    case Cmd::SET_PATTERN: data_len = 5;   break;
    case Cmd::SET_LEDS:    data_len = 120; break;
    default: return false; // unrecognized tile_cmd - can't know its size
    }

    *tile_cmd_out = tile_cmd;
    *data_len_out = data_len;
    return true;
}

void RowCommandHandler::handle_send_data(const RowBusFrame &in) {
    // Starts (or restarts) incremental forwarding; poll() advances it one
    // slot per call. The protocol doesn't allow a second SEND_DATA to the
    // same row before the frame's LATCH, so re-entry here just restarts.
    forwarding_frame_ = in;
    forward_offset_   = 0;
    forward_slot_     = 0;
    forwarding_       = true;
    latch_deferred_   = false;
}

void RowCommandHandler::advance_forwarding() {
    uint8_t tile_cmd, data_len;
    if (forward_slot_ >= TileMap::NUM_SLOTS || !parse_entry(forward_offset_, &tile_cmd, &data_len)) {
        finish_forwarding();
        return;
    }

    const TileMap &map = sense_.result();
    if (map.is_discovered(forward_slot_))
        send_tile_frame(map.address_for(forward_slot_), (Cmd)tile_cmd,
                         &forwarding_frame_.payload[forward_offset_ + 1], data_len);
    // else: slot not discovered - skip forwarding, still advance below

    forward_offset_ = (uint16_t)(forward_offset_ + 1 + data_len);
    forward_slot_++;

    if (forward_slot_ >= TileMap::NUM_SLOTS || forward_offset_ >= forwarding_frame_.len) finish_forwarding();
}

void RowCommandHandler::finish_forwarding() {
    forwarding_ = false;
    if (!latch_deferred_) return;

    latch_deferred_ = false;
    broadcast_tile_latch();
    log_error(overrun_slot_, overrun_tile_cmd_, ERROR_TYPE_LATCH_OVERRUN, last_now_ms_);
}

void RowCommandHandler::handle_latch() {
    if (!forwarding_) {
        broadcast_tile_latch();
        return;
    }

    // Still forwarding SEND_DATA: buffer this LATCH per docs/row-bus-protocol.md's
    // overrun handling - finish forwarding (poll() drives that), then fire it.
    latch_deferred_ = true;
    overrun_slot_   = forward_slot_;

    uint8_t tile_cmd, data_len;
    overrun_tile_cmd_ = parse_entry(forward_offset_, &tile_cmd, &data_len) ? tile_cmd : 0x00;
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

void RowCommandHandler::poll(uint32_t now_ms) {
    last_now_ms_ = now_ms;
    if (forwarding_) advance_forwarding();
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
