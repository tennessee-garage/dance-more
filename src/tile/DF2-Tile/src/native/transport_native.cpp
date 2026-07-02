#include "transport_native.h"
#include <cstdio>

// Implemented in issue #6.
TransportNative::TransportNative(uint8_t addr, uint8_t s, int r)
    : tile_addr(addr), slot(s), row(r) {}

void TransportNative::init() {}
void TransportNative::poll(FrameParser &) {}
void TransportNative::send(const Frame &) {}
