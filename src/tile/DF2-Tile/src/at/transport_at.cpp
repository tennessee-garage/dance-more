#include <Arduino.h>
#include "transport_at.h"

// Implemented in issue #5.
void TransportAT::init() {}
bool TransportAT::poll(FrameParser &, Frame *) { return false; }
void TransportAT::send(const Frame &) {}
