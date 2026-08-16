#include "firmware_version.h"

void fw_version_encode(const FirmwareVersion &v, uint8_t *buf) {
    buf[0] = (uint8_t)(v.version >> 8);
    buf[1] = (uint8_t)(v.version & 0xFF);
    buf[2] = (uint8_t)(v.git_sha >> 24);
    buf[3] = (uint8_t)(v.git_sha >> 16);
    buf[4] = (uint8_t)(v.git_sha >> 8);
    buf[5] = (uint8_t)(v.git_sha & 0xFF);
    buf[6] = v.flags;
}

bool fw_version_decode(const uint8_t *buf, FirmwareVersion *out) {
    uint8_t flags = buf[6];
    if (flags & ~FW_VERSION_FLAG_DIRTY) return false; // reserved bits must be 0

    out->version = (uint16_t)((buf[0] << 8) | buf[1]);
    out->git_sha = ((uint32_t)buf[2] << 24) | ((uint32_t)buf[3] << 16) |
                   ((uint32_t)buf[4] << 8) | buf[5];
    out->flags   = flags;
    return true;
}
