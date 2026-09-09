#include <unity.h>
#include <string.h>
#include "command_handler.h"
#include "fw_version_info.h"
#include "../../include/fw_version.h"

class MockSense : public ISenseControl {
public:
    bool asserted      = false;
    bool assert_called = false;
    bool release_called = false;

    void assert_sense_out() override  { asserted = true;  assert_called  = true; }
    void release_sense_out() override { asserted = false; release_called = true; }
    bool sense_is_asserted() const override { return asserted; }

    void reset() { asserted = false; assert_called = false; release_called = false; }
};

static MockSense mock_sense;
static PixelBuffer buf;
// Mutable because SET_ADDRESS writes it, and restored in setUp() so a test
// that reassigns the address cannot leak into the next one.
static constexpr uint8_t DEFAULT_ADDR = 0x05;
static uint8_t MY_ADDR = DEFAULT_ADDR;

void setUp() {
    buf = PixelBuffer{};
    mock_sense.reset();
    MY_ADDR = DEFAULT_ADDR;
}
void tearDown() {}

// ---------------------------------------------------------------------------
// Unknown command
// ---------------------------------------------------------------------------

void test_unknown_command_returns_null() {
    Frame in = {};
    in.cmd = 0x42;
    in.len = 0;
    TEST_ASSERT_NULL(handle_command(in, buf, mock_sense, MY_ADDR));
}

// ---------------------------------------------------------------------------
// SET_COLOR
// ---------------------------------------------------------------------------

void test_set_color_fills_all_leds() {
    Frame in = {};
    in.cmd        = (uint8_t)Cmd::SET_COLOR;
    in.len        = 3;
    in.payload[0] = 0xFF;
    in.payload[1] = 0x80;
    in.payload[2] = 0x10;

    TEST_ASSERT_NULL(handle_command(in, buf, mock_sense, MY_ADDR));

    for (uint8_t i = 0; i < PixelBuffer::NUM_LEDS; i++) {
        TEST_ASSERT_EQUAL_HEX8(0xFF, buf.leds[i].r);
        TEST_ASSERT_EQUAL_HEX8(0x80, buf.leds[i].g);
        TEST_ASSERT_EQUAL_HEX8(0x10, buf.leds[i].b);
    }
}

void test_set_color_noop_when_len_too_short() {
    buf.leds[0] = {0x01, 0x02, 0x03};
    Frame in = {};
    in.cmd = (uint8_t)Cmd::SET_COLOR;
    in.len = 2;

    TEST_ASSERT_NULL(handle_command(in, buf, mock_sense, MY_ADDR));
    TEST_ASSERT_EQUAL_HEX8(0x01, buf.leds[0].r);
}

// ---------------------------------------------------------------------------
// SET_LEDS
// ---------------------------------------------------------------------------

void test_set_leds_copies_all_pixels() {
    Frame in = {};
    in.cmd = (uint8_t)Cmd::SET_LEDS;
    in.len = PixelBuffer::NUM_LEDS * 3;
    for (uint8_t i = 0; i < PixelBuffer::NUM_LEDS; i++) {
        in.payload[i * 3]     = i;
        in.payload[i * 3 + 1] = (uint8_t)(i + 1);
        in.payload[i * 3 + 2] = (uint8_t)(i + 2);
    }

    TEST_ASSERT_NULL(handle_command(in, buf, mock_sense, MY_ADDR));

    for (uint8_t i = 0; i < PixelBuffer::NUM_LEDS; i++) {
        TEST_ASSERT_EQUAL_HEX8(i,       buf.leds[i].r);
        TEST_ASSERT_EQUAL_HEX8(i + 1,   buf.leds[i].g);
        TEST_ASSERT_EQUAL_HEX8(i + 2,   buf.leds[i].b);
    }
}

void test_set_leds_noop_when_len_too_short() {
    buf.leds[0] = {0xAA, 0xBB, 0xCC};
    Frame in = {};
    in.cmd = (uint8_t)Cmd::SET_LEDS;
    in.len = PixelBuffer::NUM_LEDS * 3 - 1;

    TEST_ASSERT_NULL(handle_command(in, buf, mock_sense, MY_ADDR));
    TEST_ASSERT_EQUAL_HEX8(0xAA, buf.leds[0].r);
}

// ---------------------------------------------------------------------------
// SET_PATTERN
// ---------------------------------------------------------------------------

void test_set_pattern_returns_null() {
    Frame in = {};
    in.cmd = (uint8_t)Cmd::SET_PATTERN;
    in.len = 0;
    TEST_ASSERT_NULL(handle_command(in, buf, mock_sense, MY_ADDR));
}

// ---------------------------------------------------------------------------
// LATCH
// ---------------------------------------------------------------------------

void test_latch_sets_pending_flag() {
    TEST_ASSERT_FALSE(buf.latch_pending);
    Frame in = {};
    in.cmd = (uint8_t)Cmd::LATCH;
    in.len = 0;

    TEST_ASSERT_NULL(handle_command(in, buf, mock_sense, MY_ADDR));
    TEST_ASSERT_TRUE(buf.latch_pending);
}

// ---------------------------------------------------------------------------
// ACTIVATE_SENSE
// ---------------------------------------------------------------------------

void test_activate_sense_asserts_and_returns_ack() {
    Frame in = {};
    in.addr = 0xFF;
    in.cmd  = (uint8_t)Cmd::ACTIVATE_SENSE;
    in.len  = 0;

    const Frame *resp = handle_command(in, buf, mock_sense, MY_ADDR);

    TEST_ASSERT_NOT_NULL(resp);
    TEST_ASSERT_TRUE(mock_sense.assert_called);
    TEST_ASSERT_EQUAL_HEX8(MY_ADDR, resp->addr);
    TEST_ASSERT_EQUAL_HEX8(0x81, resp->cmd);
    TEST_ASSERT_EQUAL_HEX8(1, resp->len);
    TEST_ASSERT_EQUAL_HEX8(0x00, resp->payload[0]);
}

// ---------------------------------------------------------------------------
// CLEAR_SENSE
// ---------------------------------------------------------------------------

void test_clear_sense_releases_and_returns_null() {
    mock_sense.asserted = true;
    Frame in = {};
    in.cmd = (uint8_t)Cmd::CLEAR_SENSE;
    in.len = 0;

    TEST_ASSERT_NULL(handle_command(in, buf, mock_sense, MY_ADDR));
    TEST_ASSERT_TRUE(mock_sense.release_called);
    TEST_ASSERT_FALSE(mock_sense.asserted);
}

// ---------------------------------------------------------------------------
// DETECT_SENSE
// ---------------------------------------------------------------------------

void test_detect_sense_returns_resp_when_asserted() {
    mock_sense.asserted = true;
    Frame in = {};
    in.addr = 0xFF;
    in.cmd  = (uint8_t)Cmd::DETECT_SENSE;
    in.len  = 0;

    const Frame *resp = handle_command(in, buf, mock_sense, MY_ADDR);

    TEST_ASSERT_NOT_NULL(resp);
    TEST_ASSERT_EQUAL_HEX8(MY_ADDR, resp->addr);
    TEST_ASSERT_EQUAL_HEX8((uint8_t)Cmd::DETECT_RESP, resp->cmd);
    TEST_ASSERT_EQUAL_HEX8(0, resp->len);
}

void test_detect_sense_returns_null_when_not_asserted() {
    mock_sense.asserted = false;
    Frame in = {};
    in.cmd = (uint8_t)Cmd::DETECT_SENSE;
    in.len = 0;

    TEST_ASSERT_NULL(handle_command(in, buf, mock_sense, MY_ADDR));
}

// ---------------------------------------------------------------------------
// TEST
// ---------------------------------------------------------------------------

void test_test_command_returns_ack() {
    Frame in = {};
    in.addr = 0xFF;
    in.cmd  = (uint8_t)Cmd::TEST;
    in.len  = 0;

    const Frame *resp = handle_command(in, buf, mock_sense, MY_ADDR);

    TEST_ASSERT_NOT_NULL(resp);
    TEST_ASSERT_EQUAL_HEX8(MY_ADDR, resp->addr);
    TEST_ASSERT_EQUAL_HEX8(0x84, resp->cmd);
    TEST_ASSERT_EQUAL_HEX8(1, resp->len);
    TEST_ASSERT_EQUAL_HEX8(0x00, resp->payload[0]);
}

// ---------------------------------------------------------------------------
// VERSION
// ---------------------------------------------------------------------------

void test_version_command_returns_version_resp() {
    Frame in = {};
    in.cmd = (uint8_t)Cmd::VERSION;
    in.len = 0;

    const Frame *resp = handle_command(in, buf, mock_sense, MY_ADDR);

    TEST_ASSERT_NOT_NULL(resp);
    TEST_ASSERT_EQUAL_HEX8(MY_ADDR, resp->addr);
    TEST_ASSERT_EQUAL_HEX8((uint8_t)Cmd::VERSION_RESP, resp->cmd);
    TEST_ASSERT_EQUAL_HEX8(FW_VERSION_WIRE_SIZE, resp->len);

    FirmwareVersion decoded{};
    TEST_ASSERT_TRUE(fw_version_decode(resp->payload, &decoded));
    TEST_ASSERT_EQUAL_UINT16(TILE_FW_VERSION, decoded.version);
}

// ---------------------------------------------------------------------------
// SET_ADDRESS
// ---------------------------------------------------------------------------

static Frame set_address_frame(uint8_t addr) {
    Frame in = {};
    in.addr       = ADDR_BROADCAST;   // always broadcast: the target may have no address
    in.cmd        = (uint8_t)Cmd::SET_ADDRESS;
    in.len        = 1;
    in.payload[0] = addr;
    return in;
}

void test_set_address_assigns_and_acks_from_the_new_address() {
    mock_sense.asserted = true;       // this tile is the one the walk has selected
    MY_ADDR = ADDR_UNASSIGNED;

    const Frame *resp = handle_command(set_address_frame(0x03), buf, mock_sense, MY_ADDR);

    TEST_ASSERT_EQUAL_HEX8(0x03, MY_ADDR);
    TEST_ASSERT_NOT_NULL(resp);
    // Answering from the new address is what proves the assignment took.
    TEST_ASSERT_EQUAL_HEX8(0x03, resp->addr);
    TEST_ASSERT_EQUAL_HEX8(0x86, resp->cmd);
    TEST_ASSERT_EQUAL_HEX8(1, resp->len);
    TEST_ASSERT_EQUAL_HEX8(0x00, resp->payload[0]);
}

// The whole scheme rests on this: SET_ADDRESS is a broadcast, so if a tile
// without its SENSE line asserted acted on it, every tile on the bus would
// take the same address at once.
void test_set_address_ignored_when_sense_not_asserted() {
    mock_sense.asserted = false;
    MY_ADDR = ADDR_UNASSIGNED;

    TEST_ASSERT_NULL(handle_command(set_address_frame(0x03), buf, mock_sense, MY_ADDR));
    TEST_ASSERT_EQUAL_HEX8(ADDR_UNASSIGNED, MY_ADDR);
}

void test_set_address_rejects_reserved_and_broadcast_addresses() {
    mock_sense.asserted = true;

    TEST_ASSERT_NULL(handle_command(set_address_frame(ADDR_UNASSIGNED), buf, mock_sense, MY_ADDR));
    TEST_ASSERT_EQUAL_HEX8(DEFAULT_ADDR, MY_ADDR);

    TEST_ASSERT_NULL(handle_command(set_address_frame(ADDR_BROADCAST), buf, mock_sense, MY_ADDR));
    TEST_ASSERT_EQUAL_HEX8(DEFAULT_ADDR, MY_ADDR);
}

void test_set_address_ignores_empty_payload() {
    mock_sense.asserted = true;
    Frame in = {};
    in.addr = ADDR_BROADCAST;
    in.cmd  = (uint8_t)Cmd::SET_ADDRESS;
    in.len  = 0;

    TEST_ASSERT_NULL(handle_command(in, buf, mock_sense, MY_ADDR));
    TEST_ASSERT_EQUAL_HEX8(DEFAULT_ADDR, MY_ADDR);
}

// A row that resets mid-show re-walks the chain while the tiles keep whatever
// they were given. Reassignment has to be plain overwrite, with no notion of
// "already addressed", or the second walk would find tiles it cannot rename.
void test_set_address_overwrites_an_existing_address() {
    mock_sense.asserted = true;
    MY_ADDR = 0x07;

    const Frame *resp = handle_command(set_address_frame(0x01), buf, mock_sense, MY_ADDR);

    TEST_ASSERT_EQUAL_HEX8(0x01, MY_ADDR);
    TEST_ASSERT_NOT_NULL(resp);
    TEST_ASSERT_EQUAL_HEX8(0x01, resp->addr);
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int main(int, char **) {
    UNITY_BEGIN();

    RUN_TEST(test_unknown_command_returns_null);

    RUN_TEST(test_set_color_fills_all_leds);
    RUN_TEST(test_set_color_noop_when_len_too_short);

    RUN_TEST(test_set_leds_copies_all_pixels);
    RUN_TEST(test_set_leds_noop_when_len_too_short);

    RUN_TEST(test_set_pattern_returns_null);

    RUN_TEST(test_latch_sets_pending_flag);

    RUN_TEST(test_activate_sense_asserts_and_returns_ack);

    RUN_TEST(test_clear_sense_releases_and_returns_null);

    RUN_TEST(test_detect_sense_returns_resp_when_asserted);
    RUN_TEST(test_detect_sense_returns_null_when_not_asserted);

    RUN_TEST(test_test_command_returns_ack);

    RUN_TEST(test_version_command_returns_version_resp);

    RUN_TEST(test_set_address_assigns_and_acks_from_the_new_address);
    RUN_TEST(test_set_address_ignored_when_sense_not_asserted);
    RUN_TEST(test_set_address_rejects_reserved_and_broadcast_addresses);
    RUN_TEST(test_set_address_ignores_empty_payload);
    RUN_TEST(test_set_address_overwrites_an_existing_address);

    return UNITY_END();
}
