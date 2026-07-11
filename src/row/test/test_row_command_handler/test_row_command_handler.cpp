#include <unity.h>
#include <vector>
#include "row_command_handler.h"

void setUp() {}
void tearDown() {}

// ---------------------------------------------------------------------------
// Fakes
// ---------------------------------------------------------------------------

class FakeTileTransport : public ITransport {
public:
    void init() override {}
    bool poll(FrameParser &, Frame *) override { return false; }
    void send(const Frame &frame) override { sent.push_back(frame); }
    std::vector<Frame> sent;
};

class FakeRowSense : public IRowSenseControl {
public:
    void assert_out() override {}
    void release_out() override {}
};

class FakePowerMonitor : public IPowerMonitor {
public:
    PowerReading reading{};
    void init() override {}
    PowerReading read() override { return reading; }
};

static RowBusFrame make_frame(uint8_t addr, RowBusCmd cmd, const uint8_t *payload, uint16_t len) {
    RowBusFrame f{};
    f.addr = addr;
    f.cmd  = (uint8_t)cmd;
    f.len  = len;
    for (uint16_t i = 0; i < len; i++) f.payload[i] = payload[i];
    return f;
}

// ---------------------------------------------------------------------------
// STATUS
// ---------------------------------------------------------------------------

void test_status_reports_discovered_tiles_and_state() {
    FakeTileTransport transport;
    FakeRowSense      row_sense;
    TileMap           map;
    SenseMapper       sense(transport, row_sense, map);
    FakePowerMonitor  power;

    map.set_discovered(0, 0x01);
    map.set_discovered(1, 0x02);
    map.set_discovered(2, 0x03);
    map.set_discovered(3, 0x04);
    map.set_discovered(4, 0x05);
    // slots 5,6,7 remain not-discovered

    RowCommandHandler handler(transport, sense, power, 0x00);
    RowBusFrame req = make_frame(0x00, RowBusCmd::STATUS, nullptr, 0);
    const RowBusFrame *resp = handler.handle(req);

    TEST_ASSERT_NOT_NULL(resp);
    TEST_ASSERT_EQUAL_HEX8((uint8_t)RowBusCmd::STATUS_RESP, resp->cmd);
    TEST_ASSERT_EQUAL(10, resp->len);
    TEST_ASSERT_EQUAL_HEX8(0x00, resp->payload[0]); // state: IDLE (discovery never started)
    TEST_ASSERT_EQUAL(5, resp->payload[1]);          // tiles_found
    for (uint8_t i = 0; i < 5; i++)
        TEST_ASSERT_EQUAL_HEX8((uint8_t)TileStatus::OK, resp->payload[2 + i]);
    for (uint8_t i = 5; i < 8; i++)
        TEST_ASSERT_EQUAL_HEX8((uint8_t)TileStatus::NOT_DISCOVERED, resp->payload[2 + i]);
}

// ---------------------------------------------------------------------------
// POWER
// ---------------------------------------------------------------------------

void test_power_reports_reading_fields() {
    FakeTileTransport transport;
    FakeRowSense      row_sense;
    TileMap           map;
    SenseMapper       sense(transport, row_sense, map);
    FakePowerMonitor  power;
    power.reading = {12043, 812, 9779};

    RowCommandHandler handler(transport, sense, power, 0x00);
    RowBusFrame req = make_frame(0x00, RowBusCmd::POWER, nullptr, 0);
    const RowBusFrame *resp = handler.handle(req);

    TEST_ASSERT_NOT_NULL(resp);
    TEST_ASSERT_EQUAL_HEX8((uint8_t)RowBusCmd::POWER_RESP, resp->cmd);
    TEST_ASSERT_EQUAL(6, resp->len);
    TEST_ASSERT_EQUAL_HEX8(12043 >> 8,   resp->payload[0]);
    TEST_ASSERT_EQUAL_HEX8(12043 & 0xFF, resp->payload[1]);
    TEST_ASSERT_EQUAL_HEX8(812 >> 8,     resp->payload[2]);
    TEST_ASSERT_EQUAL_HEX8(812 & 0xFF,   resp->payload[3]);
    TEST_ASSERT_EQUAL_HEX8(9779 >> 8,    resp->payload[4]);
    TEST_ASSERT_EQUAL_HEX8(9779 & 0xFF,  resp->payload[5]);
}

// ---------------------------------------------------------------------------
// SEND_DATA
// ---------------------------------------------------------------------------

void test_send_data_forwards_mixed_commands_to_correct_addresses() {
    FakeTileTransport transport;
    FakeRowSense      row_sense;
    TileMap           map;
    SenseMapper       sense(transport, row_sense, map);
    FakePowerMonitor  power;

    for (uint8_t i = 0; i < 8; i++) map.set_discovered(i, (uint8_t)(i + 1));

    // slots 0-6: SET_COLOR: slot 7: SET_LEDS.
    uint8_t payload[8 * 121] = {};
    uint16_t offset = 0;
    for (uint8_t slot = 0; slot < 7; slot++) {
        payload[offset++] = (uint8_t)Cmd::SET_COLOR;
        payload[offset++] = (uint8_t)(slot * 10);
        payload[offset++] = (uint8_t)(slot * 10 + 1);
        payload[offset++] = (uint8_t)(slot * 10 + 2);
    }
    payload[offset++] = (uint8_t)Cmd::SET_LEDS;
    for (uint16_t i = 0; i < 120; i++) payload[offset++] = (uint8_t)i;

    RowCommandHandler handler(transport, sense, power, 0x00);
    RowBusFrame req = make_frame(0x00, RowBusCmd::SEND_DATA, payload, offset);
    const RowBusFrame *resp = handler.handle(req);

    TEST_ASSERT_NULL(resp); // fire-and-forget, no response
    // handle() only starts forwarding now (#46) - poll() advances one slot
    // per call, so 8 calls are needed to forward all 8 slots.
    for (int i = 0; i < 8; i++) handler.poll(0);
    TEST_ASSERT_EQUAL(8, transport.sent.size());

    for (uint8_t slot = 0; slot < 7; slot++) {
        const Frame &f = transport.sent[slot];
        TEST_ASSERT_EQUAL_HEX8(slot + 1, f.addr);
        TEST_ASSERT_EQUAL_HEX8((uint8_t)Cmd::SET_COLOR, f.cmd);
        TEST_ASSERT_EQUAL(3, f.len);
        TEST_ASSERT_EQUAL_HEX8(slot * 10,     f.payload[0]);
        TEST_ASSERT_EQUAL_HEX8(slot * 10 + 1, f.payload[1]);
        TEST_ASSERT_EQUAL_HEX8(slot * 10 + 2, f.payload[2]);
    }

    const Frame &last = transport.sent[7];
    TEST_ASSERT_EQUAL_HEX8(8, last.addr);
    TEST_ASSERT_EQUAL_HEX8((uint8_t)Cmd::SET_LEDS, last.cmd);
    TEST_ASSERT_EQUAL(120, last.len);
    for (uint16_t i = 0; i < 120; i++)
        TEST_ASSERT_EQUAL_HEX8((uint8_t)i, last.payload[i]);
}

void test_send_data_skips_undiscovered_slots() {
    FakeTileTransport transport;
    FakeRowSense      row_sense;
    TileMap           map;
    SenseMapper       sense(transport, row_sense, map);
    FakePowerMonitor  power;

    map.set_discovered(0, 0x01);
    // slots 1-7 not discovered

    uint8_t payload[8 * 4] = {};
    uint16_t offset = 0;
    for (uint8_t slot = 0; slot < 8; slot++) {
        payload[offset++] = (uint8_t)Cmd::SET_COLOR;
        payload[offset++] = 1; payload[offset++] = 2; payload[offset++] = 3;
    }

    RowCommandHandler handler(transport, sense, power, 0x00);
    RowBusFrame req = make_frame(0x00, RowBusCmd::SEND_DATA, payload, offset);
    handler.handle(req);
    for (int i = 0; i < 8; i++) handler.poll(0);

    // Only slot 0 is discovered, but the parser must still walk all 8
    // fixed-size entries to stay aligned - if offset tracking were wrong,
    // this would either crash or miscount.
    TEST_ASSERT_EQUAL(1, transport.sent.size());
    TEST_ASSERT_EQUAL_HEX8(0x01, transport.sent[0].addr);
}

// ---------------------------------------------------------------------------
// LATCH / BLACKOUT
// ---------------------------------------------------------------------------

void test_latch_broadcasts_tile_latch() {
    FakeTileTransport transport;
    FakeRowSense      row_sense;
    TileMap           map;
    SenseMapper       sense(transport, row_sense, map);
    FakePowerMonitor  power;

    RowCommandHandler handler(transport, sense, power, 0x00);
    RowBusFrame req = make_frame(ROWBUS_ADDR_BROADCAST, RowBusCmd::LATCH, nullptr, 0);
    const RowBusFrame *resp = handler.handle(req);

    TEST_ASSERT_NULL(resp);
    TEST_ASSERT_EQUAL(1, transport.sent.size());
    TEST_ASSERT_EQUAL_HEX8(ADDR_BROADCAST, transport.sent[0].addr);
    TEST_ASSERT_EQUAL_HEX8((uint8_t)Cmd::LATCH, transport.sent[0].cmd);
}

void test_blackout_sends_black_to_each_discovered_tile_then_latch() {
    FakeTileTransport transport;
    FakeRowSense      row_sense;
    TileMap           map;
    SenseMapper       sense(transport, row_sense, map);
    FakePowerMonitor  power;

    for (uint8_t i = 0; i < 8; i++) map.set_discovered(i, (uint8_t)(i + 1));

    RowCommandHandler handler(transport, sense, power, 0x00);
    RowBusFrame req = make_frame(ROWBUS_ADDR_BROADCAST, RowBusCmd::BLACKOUT, nullptr, 0);
    const RowBusFrame *resp = handler.handle(req);

    TEST_ASSERT_NULL(resp);
    TEST_ASSERT_EQUAL(9, transport.sent.size()); // 8 SET_COLOR + 1 LATCH

    for (uint8_t i = 0; i < 8; i++) {
        const Frame &f = transport.sent[i];
        TEST_ASSERT_EQUAL_HEX8(i + 1, f.addr);
        TEST_ASSERT_EQUAL_HEX8((uint8_t)Cmd::SET_COLOR, f.cmd);
        TEST_ASSERT_EQUAL(3, f.len);
        TEST_ASSERT_EQUAL_HEX8(0, f.payload[0]);
        TEST_ASSERT_EQUAL_HEX8(0, f.payload[1]);
        TEST_ASSERT_EQUAL_HEX8(0, f.payload[2]);
    }

    const Frame &latch = transport.sent[8];
    TEST_ASSERT_EQUAL_HEX8(ADDR_BROADCAST, latch.addr);
    TEST_ASSERT_EQUAL_HEX8((uint8_t)Cmd::LATCH, latch.cmd);
    TEST_ASSERT_EQUAL(0, latch.len);
}

// ---------------------------------------------------------------------------
// LATCH overrun (#46)
// ---------------------------------------------------------------------------

void test_latch_defers_when_send_data_still_forwarding() {
    FakeTileTransport transport;
    FakeRowSense      row_sense;
    TileMap           map;
    SenseMapper       sense(transport, row_sense, map);
    FakePowerMonitor  power;

    for (uint8_t i = 0; i < 8; i++) map.set_discovered(i, (uint8_t)(i + 1));

    uint8_t payload[8 * 4] = {};
    uint16_t offset = 0;
    for (uint8_t slot = 0; slot < 8; slot++) {
        payload[offset++] = (uint8_t)Cmd::SET_COLOR;
        payload[offset++] = 1; payload[offset++] = 2; payload[offset++] = 3;
    }

    RowCommandHandler handler(transport, sense, power, 0x00);
    RowBusFrame send_req = make_frame(0x00, RowBusCmd::SEND_DATA, payload, offset);
    handler.handle(send_req);

    // Advance 3 of 8 slots before LATCH arrives.
    handler.poll(1000);
    handler.poll(1000);
    handler.poll(1000);
    TEST_ASSERT_EQUAL(3, transport.sent.size());

    RowBusFrame latch_req = make_frame(ROWBUS_ADDR_BROADCAST, RowBusCmd::LATCH, nullptr, 0);
    const RowBusFrame *latch_resp = handler.handle(latch_req);
    TEST_ASSERT_NULL(latch_resp);

    // Deferred: no LATCH frame sent yet, forwarding continues untouched.
    TEST_ASSERT_EQUAL(3, transport.sent.size());

    // Drain the remaining 5 slots.
    for (int i = 0; i < 5; i++) handler.poll(2000);

    // All 8 tiles forwarded, then exactly one deferred LATCH broadcast.
    TEST_ASSERT_EQUAL(9, transport.sent.size());
    const Frame &latch = transport.sent[8];
    TEST_ASSERT_EQUAL_HEX8(ADDR_BROADCAST, latch.addr);
    TEST_ASSERT_EQUAL_HEX8((uint8_t)Cmd::LATCH, latch.cmd);

    // Overrun logged: 3 slots had been forwarded, slot 3's SET_COLOR was
    // the in-flight command, timestamped at the poll() call that finished
    // forwarding (2000ms -> 2s).
    RowBusFrame log_req = make_frame(0x00, RowBusCmd::ERROR_LOG, nullptr, 0);
    const RowBusFrame *log_resp = handler.handle(log_req);
    TEST_ASSERT_NOT_NULL(log_resp);
    TEST_ASSERT_EQUAL(1 + 5, log_resp->len);
    TEST_ASSERT_EQUAL(1, log_resp->payload[0]); // entry_count
    TEST_ASSERT_EQUAL_HEX8(3,                        log_resp->payload[1]); // slot
    TEST_ASSERT_EQUAL_HEX8((uint8_t)Cmd::SET_COLOR,  log_resp->payload[2]); // tile_bus_cmd
    TEST_ASSERT_EQUAL_HEX8(0x04,                     log_resp->payload[3]); // error_type: LATCH_OVERRUN
    TEST_ASSERT_EQUAL_HEX8(0, log_resp->payload[4]); // timestamp high byte
    TEST_ASSERT_EQUAL_HEX8(2, log_resp->payload[5]); // timestamp low byte: 2000ms -> 2s
}

void test_latch_without_forwarding_is_unaffected_by_overrun_logic() {
    FakeTileTransport transport;
    FakeRowSense      row_sense;
    TileMap           map;
    SenseMapper       sense(transport, row_sense, map);
    FakePowerMonitor  power;

    RowCommandHandler handler(transport, sense, power, 0x00);
    handler.poll(0); // no forwarding in flight - must be a no-op

    RowBusFrame req = make_frame(ROWBUS_ADDR_BROADCAST, RowBusCmd::LATCH, nullptr, 0);
    const RowBusFrame *resp = handler.handle(req);

    TEST_ASSERT_NULL(resp);
    TEST_ASSERT_EQUAL(1, transport.sent.size()); // immediate broadcast, same as before #46

    RowBusFrame log_req = make_frame(0x00, RowBusCmd::ERROR_LOG, nullptr, 0);
    const RowBusFrame *log_resp = handler.handle(log_req);
    TEST_ASSERT_EQUAL(0, log_resp->payload[0]); // no overrun logged
}

void test_error_log_ring_buffer_caps_at_32_and_drops_oldest() {
    FakeTileTransport transport;
    FakeRowSense      row_sense;
    TileMap           map;
    SenseMapper       sense(transport, row_sense, map);
    FakePowerMonitor  power;
    map.set_discovered(0, 0x01); // slots 1-7 undiscovered, keeps each cycle cheap

    uint8_t payload[8 * 4] = {};
    uint16_t offset = 0;
    for (uint8_t slot = 0; slot < 8; slot++) {
        payload[offset++] = (uint8_t)Cmd::SET_COLOR;
        payload[offset++] = 0; payload[offset++] = 0; payload[offset++] = 0;
    }

    RowCommandHandler handler(transport, sense, power, 0x00);
    RowBusFrame send_req  = make_frame(0x00, RowBusCmd::SEND_DATA, payload, offset);
    RowBusFrame latch_req = make_frame(ROWBUS_ADDR_BROADCAST, RowBusCmd::LATCH, nullptr, 0);

    // Trigger 33 overruns (one more than the 32-entry capacity), each with a
    // distinct timestamp, to verify the oldest entry gets dropped.
    for (uint32_t i = 0; i < 33; i++) {
        uint32_t now_ms = i * 1000;
        handler.handle(send_req);
        handler.poll(now_ms);                            // advance 1 slot, then defer
        handler.handle(latch_req);
        for (int s = 0; s < 7; s++) handler.poll(now_ms); // finish the remaining 7 slots
    }

    RowBusFrame log_req = make_frame(0x00, RowBusCmd::ERROR_LOG, nullptr, 0);
    const RowBusFrame *resp = handler.handle(log_req);
    TEST_ASSERT_EQUAL(32, resp->payload[0]);
    TEST_ASSERT_EQUAL(1 + 32 * 5, resp->len);

    // Oldest (timestamp 0, from i=0) was overwritten; entries are timestamps
    // 1..32, oldest first.
    for (int i = 0; i < 32; i++) {
        uint16_t ts = (uint16_t)((resp->payload[1 + i * 5 + 3] << 8) | resp->payload[1 + i * 5 + 4]);
        TEST_ASSERT_EQUAL(i + 1, ts);
    }
}

// ---------------------------------------------------------------------------
// RE_DISCOVER / TEST / ERROR_LOG
// ---------------------------------------------------------------------------

void test_re_discover_starts_sense_mapping_and_acks() {
    FakeTileTransport transport;
    FakeRowSense      row_sense;
    TileMap           map;
    SenseMapper       sense(transport, row_sense, map);
    FakePowerMonitor  power;

    RowCommandHandler handler(transport, sense, power, 0x00);
    RowBusFrame req = make_frame(0x00, RowBusCmd::RE_DISCOVER, nullptr, 0);
    const RowBusFrame *resp = handler.handle(req);

    TEST_ASSERT_NOT_NULL(resp);
    TEST_ASSERT_EQUAL_HEX8((uint8_t)RowBusCmd::RE_DISCOVER_RESP, resp->cmd);
    TEST_ASSERT_EQUAL(1, resp->len);
    TEST_ASSERT_EQUAL_HEX8(0x00, resp->payload[0]);
    TEST_ASSERT_EQUAL(SenseMapState::DISCOVERING, sense.state());
}

void test_test_command_returns_always_pass_stub() {
    FakeTileTransport transport;
    FakeRowSense      row_sense;
    TileMap           map;
    SenseMapper       sense(transport, row_sense, map);
    FakePowerMonitor  power;

    RowCommandHandler handler(transport, sense, power, 0x00);
    RowBusFrame req = make_frame(0x00, RowBusCmd::TEST, nullptr, 0);
    const RowBusFrame *resp = handler.handle(req);

    TEST_ASSERT_NOT_NULL(resp);
    TEST_ASSERT_EQUAL_HEX8((uint8_t)RowBusCmd::TEST_RESP, resp->cmd);
    TEST_ASSERT_EQUAL(2, resp->len);
    TEST_ASSERT_EQUAL_HEX8(0x00, resp->payload[0]);
    TEST_ASSERT_EQUAL_HEX8(0x00, resp->payload[1]);
}

void test_error_log_returns_empty_when_no_errors_logged() {
    FakeTileTransport transport;
    FakeRowSense      row_sense;
    TileMap           map;
    SenseMapper       sense(transport, row_sense, map);
    FakePowerMonitor  power;

    RowCommandHandler handler(transport, sense, power, 0x00);
    RowBusFrame req = make_frame(0x00, RowBusCmd::ERROR_LOG, nullptr, 0);
    const RowBusFrame *resp = handler.handle(req);

    TEST_ASSERT_NOT_NULL(resp);
    TEST_ASSERT_EQUAL_HEX8((uint8_t)RowBusCmd::ERROR_LOG_RESP, resp->cmd);
    TEST_ASSERT_EQUAL(1, resp->len);
    TEST_ASSERT_EQUAL_HEX8(0x00, resp->payload[0]);
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int main(int, char **) {
    UNITY_BEGIN();

    RUN_TEST(test_status_reports_discovered_tiles_and_state);
    RUN_TEST(test_power_reports_reading_fields);
    RUN_TEST(test_send_data_forwards_mixed_commands_to_correct_addresses);
    RUN_TEST(test_send_data_skips_undiscovered_slots);
    RUN_TEST(test_latch_broadcasts_tile_latch);
    RUN_TEST(test_latch_defers_when_send_data_still_forwarding);
    RUN_TEST(test_latch_without_forwarding_is_unaffected_by_overrun_logic);
    RUN_TEST(test_error_log_ring_buffer_caps_at_32_and_drops_oldest);
    RUN_TEST(test_blackout_sends_black_to_each_discovered_tile_then_latch);
    RUN_TEST(test_re_discover_starts_sense_mapping_and_acks);
    RUN_TEST(test_test_command_returns_always_pass_stub);
    RUN_TEST(test_error_log_returns_empty_when_no_errors_logged);

    return UNITY_END();
}
