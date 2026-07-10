#include <unity.h>
#include <string.h>
#include "row_bus_protocol.h"

void setUp() {}
void tearDown() {}

// ---------------------------------------------------------------------------
// CRC-16/CCITT
// ---------------------------------------------------------------------------

void test_crc16_known_vector() {
    // Standard CRC-16/CCITT check value: "123456789" == 0x29B1
    const uint8_t data[] = {'1','2','3','4','5','6','7','8','9'};
    TEST_ASSERT_EQUAL_HEX16(0x29B1, row_bus_crc16(data, sizeof(data)));
}

void test_crc16_empty() {
    const uint8_t dummy = 0;
    TEST_ASSERT_EQUAL_HEX16(0xFFFF, row_bus_crc16(&dummy, 0)); // init value, no bytes processed
}

void test_crc16_differs_by_content() {
    const uint8_t a[] = {0x01, 0x02, 0x03};
    const uint8_t b[] = {0x01, 0x02, 0x04};
    TEST_ASSERT_NOT_EQUAL(row_bus_crc16(a, 3), row_bus_crc16(b, 3));
}

// ---------------------------------------------------------------------------
// row_bus_frame_encode
// ---------------------------------------------------------------------------

void test_encode_minimal_frame() {
    RowBusFrame f = {};
    f.addr = 0x03;
    f.cmd  = (uint8_t)RowBusCmd::STATUS;
    f.len  = 0;

    uint8_t buf[ROWBUS_MAX_FRAME];
    int n = row_bus_frame_encode(f, buf, sizeof(buf));

    TEST_ASSERT_EQUAL(8, n);
    TEST_ASSERT_EQUAL_HEX8(0xAA, buf[0]);  // SYNC1
    TEST_ASSERT_EQUAL_HEX8(0x55, buf[1]);  // SYNC2
    TEST_ASSERT_EQUAL_HEX8(0x03, buf[2]);  // ADDR
    TEST_ASSERT_EQUAL_HEX8((uint8_t)RowBusCmd::STATUS, buf[3]); // CMD
    TEST_ASSERT_EQUAL_HEX8(0x00, buf[4]);  // LEN_H
    TEST_ASSERT_EQUAL_HEX8(0x00, buf[5]);  // LEN_L

    // CRC must cover ADDR+CMD+LEN_H+LEN_L
    const uint8_t crc_in[] = {0x03, (uint8_t)RowBusCmd::STATUS, 0x00, 0x00};
    uint16_t expected = row_bus_crc16(crc_in, 4);
    TEST_ASSERT_EQUAL_HEX8(expected >> 8,   buf[6]); // CRC_H
    TEST_ASSERT_EQUAL_HEX8(expected & 0xFF, buf[7]); // CRC_L
}

void test_encode_with_payload() {
    RowBusFrame f = {};
    f.addr = 0x01;
    f.cmd  = (uint8_t)RowBusCmd::POWER_RESP;
    f.len  = 6;
    f.payload[0] = 0x2E; f.payload[1] = 0xEB; // voltage_mV
    f.payload[2] = 0x03; f.payload[3] = 0x2C; // current_mA
    f.payload[4] = 0x26; f.payload[5] = 0x33; // power_mW

    uint8_t buf[ROWBUS_MAX_FRAME];
    int n = row_bus_frame_encode(f, buf, sizeof(buf));

    TEST_ASSERT_EQUAL(14, n); // 8 + 6 payload bytes
    TEST_ASSERT_EQUAL_HEX8(0x2E, buf[6]);
    TEST_ASSERT_EQUAL_HEX8(0x33, buf[11]);

    const uint8_t crc_in[] = {0x01, (uint8_t)RowBusCmd::POWER_RESP, 0x00, 0x06,
                               0x2E, 0xEB, 0x03, 0x2C, 0x26, 0x33};
    uint16_t expected = row_bus_crc16(crc_in, sizeof(crc_in));
    TEST_ASSERT_EQUAL_HEX8(expected >> 8,   buf[12]);
    TEST_ASSERT_EQUAL_HEX8(expected & 0xFF, buf[13]);
}

void test_encode_max_payload_frame() {
    RowBusFrame f = {};
    f.addr = 0x02;
    f.cmd  = (uint8_t)RowBusCmd::SEND_DATA;
    f.len  = ROWBUS_MAX_PAYLOAD; // 968, a full SEND_DATA frame
    for (uint16_t i = 0; i < ROWBUS_MAX_PAYLOAD; i++) f.payload[i] = (uint8_t)i;

    uint8_t buf[ROWBUS_MAX_FRAME];
    int n = row_bus_frame_encode(f, buf, sizeof(buf));
    TEST_ASSERT_EQUAL(ROWBUS_MAX_FRAME, n); // 8 + 968 = 976
    TEST_ASSERT_EQUAL_HEX8(0x03, buf[4]); // LEN_H: 968 = 0x03C8
    TEST_ASSERT_EQUAL_HEX8(0xC8, buf[5]); // LEN_L
}

void test_encode_returns_minus1_when_buf_too_small() {
    RowBusFrame f = {};
    f.addr = 0x01;
    f.cmd  = (uint8_t)RowBusCmd::TEST;
    f.len  = 0;

    uint8_t buf[7]; // needs 8
    TEST_ASSERT_EQUAL(-1, row_bus_frame_encode(f, buf, sizeof(buf)));
}

void test_encode_returns_minus1_when_len_too_large() {
    RowBusFrame f = {};
    f.len = ROWBUS_MAX_PAYLOAD + 1;
    uint8_t buf[ROWBUS_MAX_FRAME];
    TEST_ASSERT_EQUAL(-1, row_bus_frame_encode(f, buf, sizeof(buf)));
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool parse_all(RowBusFrameParser &parser, const uint8_t *buf, int len, RowBusFrame *out) {
    for (int i = 0; i < len; i++) {
        if (parser.feed(buf[i], out)) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// RowBusFrameParser round-trips
// ---------------------------------------------------------------------------

void test_parser_roundtrip_no_payload() {
    RowBusFrame tx = {};
    tx.addr = 0x04;
    tx.cmd  = (uint8_t)RowBusCmd::LATCH;
    tx.len  = 0;

    uint8_t buf[ROWBUS_MAX_FRAME];
    int n = row_bus_frame_encode(tx, buf, sizeof(buf));

    RowBusFrameParser parser;
    RowBusFrame rx = {};
    TEST_ASSERT_TRUE(parse_all(parser, buf, n, &rx));
    TEST_ASSERT_EQUAL_HEX8(tx.addr, rx.addr);
    TEST_ASSERT_EQUAL_HEX8(tx.cmd,  rx.cmd);
    TEST_ASSERT_EQUAL(0, rx.len);
}

void test_parser_roundtrip_with_payload() {
    RowBusFrame tx = {};
    tx.addr       = 0x05;
    tx.cmd        = (uint8_t)RowBusCmd::STATUS_RESP;
    tx.len        = 10;
    for (uint16_t i = 0; i < tx.len; i++) tx.payload[i] = (uint8_t)(i * 3);

    uint8_t buf[ROWBUS_MAX_FRAME];
    int n = row_bus_frame_encode(tx, buf, sizeof(buf));

    RowBusFrameParser parser;
    RowBusFrame rx = {};
    TEST_ASSERT_TRUE(parse_all(parser, buf, n, &rx));
    TEST_ASSERT_EQUAL_HEX8(tx.addr, rx.addr);
    TEST_ASSERT_EQUAL_HEX8(tx.cmd,  rx.cmd);
    TEST_ASSERT_EQUAL(tx.len, rx.len);
    for (uint16_t i = 0; i < tx.len; i++)
        TEST_ASSERT_EQUAL_HEX8(tx.payload[i], rx.payload[i]);
}

// Per #27's acceptance criteria: round-trip a full SEND_DATA-sized (968-byte)
// frame specifically, not just a small payload.
void test_parser_roundtrip_send_data_max_payload() {
    RowBusFrame tx = {};
    tx.addr = 0x07;
    tx.cmd  = (uint8_t)RowBusCmd::SEND_DATA;
    tx.len  = ROWBUS_MAX_PAYLOAD;
    for (uint16_t i = 0; i < ROWBUS_MAX_PAYLOAD; i++) tx.payload[i] = (uint8_t)(i ^ 0xA5);

    uint8_t buf[ROWBUS_MAX_FRAME];
    int n = row_bus_frame_encode(tx, buf, sizeof(buf));
    TEST_ASSERT_EQUAL(ROWBUS_MAX_FRAME, n);

    RowBusFrameParser parser;
    RowBusFrame rx = {};
    TEST_ASSERT_TRUE(parse_all(parser, buf, n, &rx));
    TEST_ASSERT_EQUAL_HEX8(tx.addr, rx.addr);
    TEST_ASSERT_EQUAL_HEX8(tx.cmd,  rx.cmd);
    TEST_ASSERT_EQUAL(ROWBUS_MAX_PAYLOAD, rx.len);
    for (uint16_t i = 0; i < ROWBUS_MAX_PAYLOAD; i++)
        TEST_ASSERT_EQUAL_HEX8(tx.payload[i], rx.payload[i]);
}

// ---------------------------------------------------------------------------
// RowBusFrameParser error handling
// ---------------------------------------------------------------------------

void test_parser_rejects_corrupted_crc() {
    RowBusFrame tx = {};
    tx.addr = 0x01;
    tx.cmd  = (uint8_t)RowBusCmd::TEST;
    tx.len  = 0;

    uint8_t buf[ROWBUS_MAX_FRAME];
    int n = row_bus_frame_encode(tx, buf, sizeof(buf));
    buf[n - 1] ^= 0xFF; // flip all bits in CRC_L

    RowBusFrameParser parser;
    RowBusFrame rx = {};
    TEST_ASSERT_FALSE(parse_all(parser, buf, n, &rx));
}

void test_parser_rejects_corrupted_payload() {
    RowBusFrame tx = {};
    tx.addr       = 0x01;
    tx.cmd        = (uint8_t)RowBusCmd::STATUS_RESP;
    tx.len        = 3;
    tx.payload[0] = 0xFF;
    tx.payload[1] = 0x00;
    tx.payload[2] = 0x00;

    uint8_t buf[ROWBUS_MAX_FRAME];
    int n = row_bus_frame_encode(tx, buf, sizeof(buf));
    buf[6] ^= 0x01; // corrupt first payload byte, CRC now wrong

    RowBusFrameParser parser;
    RowBusFrame rx = {};
    TEST_ASSERT_FALSE(parse_all(parser, buf, n, &rx));
}

void test_parser_rejects_invalid_len() {
    // Hand-craft a frame with LEN = ROWBUS_MAX_PAYLOAD + 1 (969 = 0x03C9)
    const uint8_t stream[] = {
        0xAA, 0x55,
        0x01,                       // ADDR
        (uint8_t)RowBusCmd::SEND_DATA, // CMD
        0x03, 0xC9,                 // LEN — invalid (969)
        0x00, 0x00                  // dummy CRC (never reached)
    };
    RowBusFrameParser parser;
    RowBusFrame rx = {};
    TEST_ASSERT_FALSE(parse_all(parser, stream, sizeof(stream), &rx));
}

// ---------------------------------------------------------------------------
// RowBusFrameParser sync / framing edge cases
// ---------------------------------------------------------------------------

void test_parser_ignores_garbage_prefix() {
    RowBusFrame tx = {};
    tx.addr = 0x01;
    tx.cmd  = (uint8_t)RowBusCmd::BLACKOUT;
    tx.len  = 0;

    uint8_t frame_buf[ROWBUS_MAX_FRAME];
    int n = row_bus_frame_encode(tx, frame_buf, sizeof(frame_buf));

    const uint8_t garbage[] = {0x00, 0x11, 0xAA, 0x00, 0xFF, 0x42};
    uint8_t stream[sizeof(garbage) + ROWBUS_MAX_FRAME];
    memcpy(stream, garbage, sizeof(garbage));
    memcpy(stream + sizeof(garbage), frame_buf, n);

    RowBusFrameParser parser;
    RowBusFrame rx = {};
    TEST_ASSERT_TRUE(parse_all(parser, stream, sizeof(garbage) + n, &rx));
    TEST_ASSERT_EQUAL_HEX8(tx.addr, rx.addr);
    TEST_ASSERT_EQUAL_HEX8(tx.cmd,  rx.cmd);
}

void test_parser_handles_double_aa_sync() {
    // Stream: 0xAA | 0xAA 0x55 ADDR CMD LEN_H LEN_L CRC_H CRC_L
    // First 0xAA enters SYNC2; second 0xAA stays in SYNC2; 0x55 advances to ADDR.
    RowBusFrame tx = {};
    tx.addr = 0x02;
    tx.cmd  = (uint8_t)RowBusCmd::RE_DISCOVER;
    tx.len  = 0;

    uint8_t frame_buf[ROWBUS_MAX_FRAME];
    int n = row_bus_frame_encode(tx, frame_buf, sizeof(frame_buf));

    uint8_t stream[ROWBUS_MAX_FRAME + 1];
    stream[0] = 0xAA; // extra SYNC1 byte before the real frame
    memcpy(stream + 1, frame_buf, n);

    RowBusFrameParser parser;
    RowBusFrame rx = {};
    TEST_ASSERT_TRUE(parse_all(parser, stream, n + 1, &rx));
    TEST_ASSERT_EQUAL_HEX8(tx.addr, rx.addr);
    TEST_ASSERT_EQUAL_HEX8(tx.cmd,  rx.cmd);
}

void test_parser_consecutive_frames() {
    RowBusFrame tx1 = {};
    tx1.addr = 0x01; tx1.cmd = (uint8_t)RowBusCmd::TEST; tx1.len = 0;

    RowBusFrame tx2 = {};
    tx2.addr = 0xFF; tx2.cmd = (uint8_t)RowBusCmd::LATCH; tx2.len = 0;

    uint8_t buf[ROWBUS_MAX_FRAME * 2];
    int n1 = row_bus_frame_encode(tx1, buf,      ROWBUS_MAX_FRAME);
    int n2 = row_bus_frame_encode(tx2, buf + n1, ROWBUS_MAX_FRAME);

    RowBusFrameParser parser;
    RowBusFrame rx1 = {}, rx2 = {};
    bool got1 = false, got2 = false;
    for (int i = 0; i < n1 + n2; i++) {
        RowBusFrame tmp = {};
        if (parser.feed(buf[i], &tmp)) {
            if (!got1) { rx1 = tmp; got1 = true; }
            else        { rx2 = tmp; got2 = true; }
        }
    }
    TEST_ASSERT_TRUE(got1);
    TEST_ASSERT_TRUE(got2);
    TEST_ASSERT_EQUAL_HEX8(tx1.cmd,  rx1.cmd);
    TEST_ASSERT_EQUAL_HEX8(tx2.cmd,  rx2.cmd);
    TEST_ASSERT_EQUAL_HEX8(tx2.addr, rx2.addr);
}

void test_parser_reset_clears_state() {
    RowBusFrame tx = {};
    tx.addr = 0x01; tx.cmd = (uint8_t)RowBusCmd::TEST; tx.len = 0;

    uint8_t buf[ROWBUS_MAX_FRAME];
    int n = row_bus_frame_encode(tx, buf, sizeof(buf));

    RowBusFrameParser parser;
    RowBusFrame rx = {};
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
    RUN_TEST(test_parser_roundtrip_send_data_max_payload);

    RUN_TEST(test_parser_rejects_corrupted_crc);
    RUN_TEST(test_parser_rejects_corrupted_payload);
    RUN_TEST(test_parser_rejects_invalid_len);

    RUN_TEST(test_parser_ignores_garbage_prefix);
    RUN_TEST(test_parser_handles_double_aa_sync);
    RUN_TEST(test_parser_consecutive_frames);
    RUN_TEST(test_parser_reset_clears_state);

    return UNITY_END();
}
