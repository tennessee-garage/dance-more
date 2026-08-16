#pragma once
#include <stdint.h>

// Firmware version identity carried on both buses: docs/row-bus-protocol.md's
// VERSION_RESP embeds one for the row plus one per tile slot;
// docs/tile-bus-protocol.md's VERSION_RESP embeds one for the tile.
//
// version   - ROW_FW_VERSION or TILE_FW_VERSION (src/row or src/tile
//             include/fw_version.h), hand-bumped per side.
// git_sha   - first 4 bytes of the build's commit SHA (FW_GIT_SHA, injected
//             by platformio.ini's extra_scripts).
// flags     - bit 0 = built from a dirty tree (FW_DIRTY); bits 1-7
//             reserved, must be 0.
struct FirmwareVersion {
    uint16_t version;
    uint32_t git_sha;
    uint8_t  flags;
};

static constexpr uint8_t FW_VERSION_FLAG_DIRTY = 0x01;

// Wire size: version(2) + git_sha(4) + flags(1), big-endian.
static constexpr uint8_t FW_VERSION_WIRE_SIZE = 7;

void fw_version_encode(const FirmwareVersion &v, uint8_t *buf);

// Returns false (and leaves *out unchanged) if a reserved flag bit is set -
// the one way this fixed-size, always-in-range encoding can be malformed.
bool fw_version_decode(const uint8_t *buf, FirmwareVersion *out);
