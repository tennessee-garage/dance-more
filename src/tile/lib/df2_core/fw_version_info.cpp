#include "fw_version_info.h"
#include "../../include/fw_version.h" // TILE_FW_VERSION

// tools/fw_version_env.py sets these for every real build; the fallback
// here only matters for a tool that compiles this file without going
// through platformio.ini's extra_scripts (e.g. a bare g++ invocation).
#ifndef FW_GIT_SHA
#define FW_GIT_SHA 0
#endif
#ifndef FW_DIRTY
#define FW_DIRTY 1
#endif

FirmwareVersion tile_fw_version() {
    FirmwareVersion v{};
    v.version = TILE_FW_VERSION;
    v.git_sha = (uint32_t)(FW_GIT_SHA);
    v.flags   = FW_DIRTY ? FW_VERSION_FLAG_DIRTY : 0;
    return v;
}
