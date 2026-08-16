#pragma once
#include <stdint.h>

// Row firmware version. Bump by 1 in any PR that changes src/row (non-doc).
// A PR touching src/common/tile_bus_protocol bumps this AND
// src/tile/include/fw_version.h's TILE_FW_VERSION. Plain incrementing
// integer - gaps are fine, going backwards is not.
static constexpr uint16_t ROW_FW_VERSION = 1;
