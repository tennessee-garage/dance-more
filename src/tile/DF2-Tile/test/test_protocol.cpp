#include <unity.h>
#include <string.h>
#include "protocol.h"

void setUp() {}
void tearDown() {}

// ---------------------------------------------------------------------------
// CRC-16/CCITT
// ---------------------------------------------------------------------------

void test_crc16_known_vector() {
    // Standard CRC-16/CCITT check value: "123456789" == 0x29B1
    const uint8_t data[] = {'1','2','3','4','5','6','7','8','9'};
    TEST_ASSERT_EQUAL_HEX16(0x29B1, crc16(data, sizeof(data)));
}

void test_crc16_empty() {
    const uint8_t dummy = 0;
    TEST_ASSERT_EQUAL_HEX16(0xFFFF, crc16(&dummy, 0)); // init value, no bytes processed
}

void test_crc16_differs_by_content() {
    const uint8_t a[] = {0x01, 0x02, 0x03};
    const uint8_t b[] = {0x01, 0x02, 0x04};
    TEST_ASSERT_NOT_EQUAL(crc16(a, 3), crc16(b, 3));
}

// ---------------------------------------------------------------------------
// frame_encode
// ---------------------------------------------------------------------------

void test_encode_minimal_frame() {
    Frame f = {};
    f.addr = 0x0A;
    f.cmd  = (uint8_t)Cmd::TEST;
    f.len  = 0;

    uint8_t buf[MAX_FRAME_SIZE];
    int n = frame_encode(f, buf, sizeof(buf));

    TEST_ASSERT_EQUAL(7, n);
    TEST_ASSERT_EQUAL_HEX8(0xAA, buf[0]);  // SYNC1
    TEST_ASSERT_EQUAL_HEX8(0x55, buf[1]);  // SYNC2
    TEST_ASSERT_EQUAL_HEX8(0x0A, buf[2]);  // ADDR
    TEST_ASSERT_EQUAL_HEX8((uint8_t)Cmd::TEST, buf[3]); // CMD
    TEST_ASSERT_EQUAL_HEX8(0x00, buf[4]);  // LEN

    // CRC must cover ADDR+CMD+LEN
    const uint8_t crc_in[] = {0x0A, (uint8_t)Cmd::TEST, 0x00};
    uint16_t expected = crc16(crc_in, 3);
    TEST_ASSERT_EQUAL_HEX8(expected >> 8,   buf[5]); // CRC_H
    TEST_ASSERT_EQUAL_HEX8(expected & 0xFF, buf[6]); // CRC_L
}

void test_encode_with_payload() {
    Frame f = {};
    f.addr = 0x03;
    f.cmd  = (uint8_t)Cmd::SET_COLOR;
    f.len  = 3;
    f.payload[0] = 0xFF;
    f.payload[1] = 0x00;
    f.payload[2] = 0x80;

    uint8_t buf[MAX_FRAME_SIZE];
    int n = frame_encode(f, buf, sizeof(buf));

    TEST_ASSERT_EQUAL(10, n); // 7 + 3 payload bytes
    TEST_ASSERT_EQUAL_HEX8(0xFF, buf[5]);
    TEST_ASSERT_EQUAL_HEX8(0x00, buf[6]);
    TEST_ASSERT_EQUAL_HEX8(0x80, buf[7]);

    // CRC covers ADDR+CMD+LEN+PAYLOAD
    const uint8_t crc_in[] = {0x03, (uint8_t)Cmd::SET_COLOR, 0x03, 0xFF, 0x00, 0x80};
    uint16_t expected = crc16(crc_in, sizeof(crc_in));
    TEST_ASSERT_EQUAL_HEX8(expected >> 8,   buf[8]);
    TEST_ASSERT_EQUAL_HEX8(expected & 0xFF, buf[9]);
}

void test_encode_max_payload_frame() {
    Frame f = {};
    f.addr = 0x01;
    f.cmd  = (uint8_t)Cmd::SET_LEDS;
    f.len  = MAX_PAYLOAD; // 120
    for (uint8_t i = 0; i < MAX_PAYLOAD; i++) f.payload[i] = i;

    uint8_t buf[MAX_FRAME_SIZE];
    int n = frame_encode(f, buf, sizeof(buf));
    TEST_ASSERT_EQUAL(MAX_FRAME_SIZE, n); // 7 + 120 = 127
}

void test_encode_returns_minus1_when_buf_too_small() {
    Frame f = {};
    f.addr = 0x01;
    f.cmd  = (uint8_t)Cmd::TEST;
    f.len  = 0;

    uint8_t buf[6]; // needs 7
    TEST_ASSERT_EQUAL(-1, frame_encode(f, buf, sizeof(buf)));
}

void test_encode_returns_minus1_when_len_too_large() {
    Frame f = {};
    f.len = MAX_PAYLOAD + 1;
    uint8_t buf[MAX_FRAME_SIZE];
    TEST_ASSERT_EQUAL(-1, frame_encode(f, buf, sizeof(buf)));
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool parse_all(FrameParser &parser, const uint8_t *buf, int len, Frame *out) {
    for (int i = 0; i < len; i++) {
        if (parser.feed(buf[i], out)) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// FrameParser round-trips
// ---------------------------------------------------------------------------

void test_parser_roundtrip_no_payload() {
    Frame tx = {};
    tx.addr = 0x07;
    tx.cmd  = (uint8_t)Cmd::LATCH;
    tx.len  = 0;

    uint8_t buf[MAX_FRAME_SIZE];
    int n = frame_encode(tx, buf, sizeof(buf));

    FrameParser parser;
    Frame rx = {};
    TEST_ASSERT_TRUE(parse_all(parser, buf, n, &rx));
    TEST_ASSERT_EQUAL_HEX8(tx.addr, rx.addr);
    TEST_ASSERT_EQUAL_HEX8(tx.cmd,  rx.cmd);
    TEST_ASSERT_EQUAL_HEX8(0,       rx.len);
}

void test_parser_roundtrip_with_payload() {
    Frame tx = {};
    tx.addr       = 0x0B;
    tx.cmd        = (uint8_t)Cmd::SET_COLOR;
    tx.len        = 3;
    tx.payload[0] = 0x10;
    tx.payload[1] = 0x20;
    tx.payload[2] = 0x30;

    uint8_t buf[MAX_FRAME_SIZE];
    int n = frame_encode(tx, buf, sizeof(buf));

    FrameParser parser;
    Frame rx = {};
    TEST_ASSERT_TRUE(parse_all(parser, buf, n, &rx));
    TEST_ASSERT_EQUAL_HEX8(tx.addr,       rx.addr);
    TEST_ASSERT_EQUAL_HEX8(tx.cmd,        rx.cmd);
    TEST_ASSERT_EQUAL_HEX8(tx.len,        rx.len);
    TEST_ASSERT_EQUAL_HEX8(tx.payload[0], rx.payload[0]);
    TEST_ASSERT_EQUAL_HEX8(tx.payload[1], rx.payload[1]);
    TEST_ASSERT_EQUAL_HEX8(tx.payload[2], rx.payload[2]);
}

void test_parser_roundtrip_set_leds() {
    Frame tx = {};
    tx.addr = 0x04;
    tx.cmd  = (uint8_t)Cmd::SET_LEDS;
    tx.len  = MAX_PAYLOAD;
    for (uint8_t i = 0; i < MAX_PAYLOAD; i++) tx.payload[i] = (uint8_t)(i * 2);

    uint8_t buf[MAX_FRAME_SIZE];
    int n = frame_encode(tx, buf, sizeof(buf));

    FrameParser parser;
    Frame rx = {};
    TEST_ASSERT_TRUE(parse_all(parser, buf, n, &rx));
    TEST_ASSERT_EQUAL_HEX8(MAX_PAYLOAD, rx.len);
    for (uint8_t i = 0; i < MAX_PAYLOAD; i++)
        TEST_ASSERT_EQUAL_HEX8(tx.payload[i], rx.payload[i]);
}

// ---------------------------------------------------------------------------
// FrameParser error handling
// ---------------------------------------------------------------------------

void test_parser_rejects_corrupted_crc() {
    Frame tx = {};
    tx.addr = 0x01;
    tx.cmd  = (uint8_t)Cmd::TEST;
    tx.len  = 0;

    uint8_t buf[MAX_FRAME_SIZE];
    int n = frame_encode(tx, buf, sizeof(buf));
    buf[n - 1] ^= 0xFF; // flip all bits in CRC_L

    FrameParser parser;
    Frame rx = {};
    TEST_ASSERT_FALSE(parse_all(parser, buf, n, &rx));
}

void test_parser_rejects_corrupted_payload() {
    Frame tx = {};
    tx.addr       = 0x01;
    tx.cmd        = (uint8_t)Cmd::SET_COLOR;
    tx.len        = 3;
    tx.payload[0] = 0xFF;
    tx.payload[1] = 0x00;
    tx.payload[2] = 0x00;

    uint8_t buf[MAX_FRAME_SIZE];
    int n = frame_encode(tx, buf, sizeof(buf));
    buf[5] ^= 0x01; // corrupt first payload byte, CRC now wrong

    FrameParser parser;
    Frame rx = {};
    TEST_ASSERT_FALSE(parse_all(parser, buf, n, &rx));
}

void test_parser_rejects_invalid_len() {
    // Hand-craft a frame with LEN = MAX_PAYLOAD + 1 (121)
    const uint8_t stream[] = {
        0xAA, 0x55,
        0x01,                   // ADDR
        (uint8_t)Cmd::SET_LEDS, // CMD
        MAX_PAYLOAD + 1,        // LEN — invalid
        0x00, 0x00              // dummy CRC (never reached)
    };
    FrameParser parser;
    Frame rx = {};
    TEST_ASSERT_FALSE(parse_all(parser, stream, sizeof(stream), &rx));
}

// ---------------------------------------------------------------------------
// FrameParser sync / framing edge cases
// ---------------------------------------------------------------------------

void test_parser_ignores_garbage_prefix() {
    Frame tx = {};
    tx.addr = 0x01;
    tx.cmd  = (uint8_t)Cmd::CLEAR_SENSE;
    tx.len  = 0;

    uint8_t frame_buf[MAX_FRAME_SIZE];
    int n = frame_encode(tx, frame_buf, sizeof(frame_buf));

    // Build stream: garbage + valid frame
    const uint8_t garbage[] = {0x00, 0x11, 0xAA, 0x00, 0xFF, 0x42};
    uint8_t stream[sizeof(garbage) + MAX_FRAME_SIZE];
    memcpy(stream, garbage, sizeof(garbage));
    memcpy(stream + sizeof(garbage), frame_buf, n);

    FrameParser parser;
    Frame rx = {};
    TEST_ASSERT_TRUE(parse_all(parser, stream, sizeof(garbage) + n, &rx));
    TEST_ASSERT_EQUAL_HEX8(tx.addr, rx.addr);
    TEST_ASSERT_EQUAL_HEX8(tx.cmd,  rx.cmd);
}

void test_parser_handles_double_aa_sync() {
    // Stream: 0xAA | 0xAA 0x55 ADDR CMD LEN CRC_H CRC_L
    // First 0xAA enters SYNC2; second 0xAA stays in SYNC2; 0x55 advances to ADDR.
    Frame tx = {};
    tx.addr = 0x02;
    tx.cmd  = (uint8_t)Cmd::DETECT_SENSE;
    tx.len  = 0;

    uint8_t frame_buf[MAX_FRAME_SIZE];
    int n = frame_encode(tx, frame_buf, sizeof(frame_buf));

    uint8_t stream[MAX_FRAME_SIZE + 1];
    stream[0] = 0xAA; // extra SYNC1 byte before the real frame
    memcpy(stream + 1, frame_buf, n);

    FrameParser parser;
    Frame rx = {};
    TEST_ASSERT_TRUE(parse_all(parser, stream, n + 1, &rx));
    TEST_ASSERT_EQUAL_HEX8(tx.addr, rx.addr);
    TEST_ASSERT_EQUAL_HEX8(tx.cmd,  rx.cmd);
}

void test_parser_consecutive_frames() {
    Frame tx1 = {};
    tx1.addr = 0x01; tx1.cmd = (uint8_t)Cmd::SET_COLOR; tx1.len = 3;
    tx1.payload[0] = 0xFF; tx1.payload[1] = 0x00; tx1.payload[2] = 0x00;

    Frame tx2 = {};
    tx2.addr = 0xFF; tx2.cmd = (uint8_t)Cmd::LATCH; tx2.len = 0;

    uint8_t buf[MAX_FRAME_SIZE * 2];
    int n1 = frame_encode(tx1, buf,      MAX_FRAME_SIZE);
    int n2 = frame_encode(tx2, buf + n1, MAX_FRAME_SIZE);

    FrameParser parser;
    Frame rx1 = {}, rx2 = {};
    bool got1 = false, got2 = false;
    for (int i = 0; i < n1 + n2; i++) {
        Frame tmp = {};
        if (parser.feed(buf[i], &tmp)) {
            if (!got1) { rx1 = tmp; got1 = true; }
            else        { rx2 = tmp; got2 = true; }
        }
    }
    TEST_ASSERT_TRUE(got1);
    TEST_ASSERT_TRUE(got2);
    TEST_ASSERT_EQUAL_HEX8(tx1.cmd,  rx1.cmd);
    TEST_ASSERT_EQUAL_HEX8(tx1.payload[0], rx1.payload[0]);
    TEST_ASSERT_EQUAL_HEX8(tx2.cmd,  rx2.cmd);
    TEST_ASSERT_EQUAL_HEX8(tx2.addr, rx2.addr);
}

void test_parser_reset_clears_state() {
    Frame tx = {};
    tx.addr = 0x01; tx.cmd = (uint8_t)Cmd::TEST; tx.len = 0;

    uint8_t buf[MAX_FRAME_SIZE];
    int n = frame_encode(tx, buf, sizeof(buf));

    FrameParser parser;
    Frame rx = {};
    // Feed half a frame, then reset
    for (int i = 0; i < n / 2; i++) parser.feed(buf[i], &rx);
    parser.reset();
    // Feed the full frame — should parse cleanly from scratch
    TEST_ASSERT_TRUE(parse_all(parser, buf, n, &rx));
    TEST_ASSERT_EQUAL_HEX8(tx.addr, rx.addr);
    TEST_ASSERT_EQUAL_HEX8(tx.cmd,  rx.cmd);
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int main(int, char **) {
    UNITY_BEGIN();

    RUN_TEST(test_crc16_known_vector);
    RUN_TEST(test_crc16_empty);
    RUN_TEST(test_crc16_differs_by_content);

    RUN_TEST(test_encode_minimal_frame);
    RUN_TEST(test_encode_with_payload);
    RUN_TEST(test_encode_max_payload_frame);
    RUN_TEST(test_encode_returns_minus1_when_buf_too_small);
    RUN_TEST(test_encode_returns_minus1_when_len_too_large);

    RUN_TEST(test_parser_roundtrip_no_payload);
    RUN_TEST(test_parser_roundtrip_with_payload);
    RUN_TEST(test_parser_roundtrip_set_leds);

    RUN_TEST(test_parser_rejects_corrupted_crc);
    RUN_TEST(test_parser_rejects_corrupted_payload);
    RUN_TEST(test_parser_rejects_invalid_len);

    RUN_TEST(test_parser_ignores_garbage_prefix);
    RUN_TEST(test_parser_handles_double_aa_sync);
    RUN_TEST(test_parser_consecutive_frames);
    RUN_TEST(test_parser_reset_clears_state);

    return UNITY_END();
}
