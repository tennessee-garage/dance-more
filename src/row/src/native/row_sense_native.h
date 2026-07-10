#pragma once
#include "row_sense_control.h"

// Shares TileTransportNative's socket fd - SENSE and FRAME messages
// interleave on the same broker connection.
class RowSenseNative : public IRowSenseControl {
    int fd;
public:
    explicit RowSenseNative(int fd) : fd(fd) {}

    void assert_out() override;    // write SENSE_ASSERT (0x03)
    void release_out() override;   // write SENSE_RELEASE (0x04)
};
