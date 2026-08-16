#pragma once
// MY_ROW_ADDR (0x00-0x07) comes from the build, not this file - each row's
// PlatformIO environment (row0..row7 in platformio.ini) defines it via
// build_flags, chosen by whichever env you build/upload with.
//
// This used to be generated per-board from the RP2350's CHIPID
// (tools/assign_row_address.py), read via `picotool otp get` before any
// firmware was on the chip. That required the board to already be in
// BOOTSEL mode at the moment the script ran, which conflicted with
// PlatformIO's own upload flow (a 1200-baud touch reset into the
// bootloader) - the two were fighting over who puts the board in BOOTSEL
// and when, and the script never got past that far enough to be used.
// Address selection is now just "which environment did you build."
#ifndef MY_ROW_ADDR
#error "No row address defined. Build/upload with one of the row0..row7 " \
       "environments (platformio.ini), not seeed_xiao_rp2350 directly."
#endif
