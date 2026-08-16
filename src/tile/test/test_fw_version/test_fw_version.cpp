#include <unity.h>
#include <string.h>
#include "firmware_version.h"
#include "fw_version_info.h"
#include "../../include/fw_version.h"

void setUp() {}
void tearDown() {}

void test_encode_wire_layout() {
    FirmwareVersion v{0x0102, 0x11223344, FW_VERSION_FLAG_DIRTY};
    uint8_t buf[FW_VERSION_WIRE_SIZE];
    fw_version_encode(v, buf);

    const uint8_t expected[FW_VERSION_WIRE_SIZE] = {
        0x01, 0x02,             // version, big-endian
        0x11, 0x22, 0x33, 0x44, // git_sha, big-endian
        0x01,                   // flags
    };
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expected, buf, FW_VERSION_WIRE_SIZE);
}

void test_decode_roundtrip() {
    FirmwareVersion in{12, 0x2b5c293c, 0};
    uint8_t buf[FW_VERSION_WIRE_SIZE];
    fw_version_encode(in, buf);

    FirmwareVersion out{};
    TEST_ASSERT_TRUE(fw_version_decode(buf, &out));
    TEST_ASSERT_EQUAL_UINT16(in.version, out.version);
    TEST_ASSERT_EQUAL_UINT32(in.git_sha, out.git_sha);
    TEST_ASSERT_EQUAL_UINT8(in.flags, out.flags);
}

void test_decode_zero_entry() {
    // An undiscovered/non-responsive tile slot is transmitted as all zeros
    // (docs/row-bus-protocol.md's VERSION_RESP) and must decode cleanly.
    const uint8_t buf[FW_VERSION_WIRE_SIZE] = {0};
    FirmwareVersion out{};
    TEST_ASSERT_TRUE(fw_version_decode(buf, &out));
    TEST_ASSERT_EQUAL_UINT16(0, out.version);
    TEST_ASSERT_EQUAL_UINT32(0, out.git_sha);
    TEST_ASSERT_EQUAL_UINT8(0, out.flags);
}

void test_decode_rejects_reserved_flag_bits() {
    uint8_t buf[FW_VERSION_WIRE_SIZE] = {0x00, 0x01, 0, 0, 0, 0, 0x02}; // bit 1 set
    FirmwareVersion out{};
    TEST_ASSERT_FALSE(fw_version_decode(buf, &out));
}

// tile_fw_version() pulls TILE_FW_VERSION and the build-injected FW_GIT_SHA/
// FW_DIRTY (tools/fw_version_env.py) together; native/test_native builds
// still see real values since the script runs there too.
void test_tile_fw_version_reports_build_identity() {
    FirmwareVersion v = tile_fw_version();
    TEST_ASSERT_EQUAL_UINT16(TILE_FW_VERSION, v.version);
    TEST_ASSERT_EQUAL_UINT8(0, v.flags & ~FW_VERSION_FLAG_DIRTY); // no reserved bits set
}

int main(int, char **) {
    UNITY_BEGIN();

    RUN_TEST(test_encode_wire_layout);
    RUN_TEST(test_decode_roundtrip);
    RUN_TEST(test_decode_zero_entry);
    RUN_TEST(test_decode_rejects_reserved_flag_bits);
    RUN_TEST(test_tile_fw_version_reports_build_identity);

    return UNITY_END();
}
