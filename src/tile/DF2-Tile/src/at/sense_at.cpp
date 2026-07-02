#include <Arduino.h>
#include "sense_at.h"

// Implemented in issue #9.
void SenseAT::init() {}
void SenseAT::assert_sense_out() {}
void SenseAT::release_sense_out() {}
bool SenseAT::sense_is_asserted() const { return false; }
