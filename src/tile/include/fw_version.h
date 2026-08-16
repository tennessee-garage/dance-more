#pragma once
#include <stdint.h>

// Tile firmware version. Bump by 1 in any PR that changes src/tile (non-doc).
// A PR touching src/common/tile_bus_protocol bumps this AND
// src/row/include/fw_version.h's ROW_FW_VERSION. Plain incrementing
// integer - gaps are fine, going backwards is not.
static constexpr uint16_t TILE_FW_VERSION = 1;
