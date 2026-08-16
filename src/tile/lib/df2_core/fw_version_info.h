#pragma once
// See command_handler.h: PlatformIO's LDF doesn't propagate lib_extra_dirs
// include paths into other lib/ folders' own compile steps, so this needs a
// relative include rather than the bare "firmware_version.h" that files
// directly under src/ can use.
#include "../../../common/tile_bus_protocol/firmware_version.h"

// This tile's identity: TILE_FW_VERSION (include/fw_version.h) plus the
// git SHA/dirty flag injected by tools/fw_version_env.py at build time.
FirmwareVersion tile_fw_version();
