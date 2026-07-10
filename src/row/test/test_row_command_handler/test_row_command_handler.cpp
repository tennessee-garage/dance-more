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

void test_error_log_returns_empty_stub() {
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
    RUN_TEST(test_blackout_sends_black_to_each_discovered_tile_then_latch);
    RUN_TEST(test_re_discover_starts_sense_mapping_and_acks);
    RUN_TEST(test_test_command_returns_always_pass_stub);
    RUN_TEST(test_error_log_returns_empty_stub);

    return UNITY_END();
}
