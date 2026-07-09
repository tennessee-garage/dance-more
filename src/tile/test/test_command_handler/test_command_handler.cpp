#include <unity.h>
#include <string.h>
#include "command_handler.h"

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
static constexpr uint8_t MY_ADDR = 0x05;

void setUp() {
    buf = PixelBuffer{};
    mock_sense.reset();
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

    return UNITY_END();
}
