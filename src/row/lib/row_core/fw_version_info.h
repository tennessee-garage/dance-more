#pragma once
#include "firmware_version.h"

// This board's identity: ROW_FW_VERSION (include/fw_version.h) plus the
// git SHA/dirty flag injected by tools/fw_version_env.py at build time.
FirmwareVersion row_fw_version();
