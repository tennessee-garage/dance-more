#include <Arduino.h>
#include "row_sense_rp2350.h"
#include "pins.h"

void RowSenseRP2350::init() {
    // Start released, with pull-up so the line stays high until asserted.
    pinMode(PIN_ROW_SENSE, INPUT_PULLUP);
}

void RowSenseRP2350::assert_out() {
    // Drive the line low first, then switch to output so the transition is
    // clean (avoids a brief HIGH glitch from the port register) - same
    // technique as the tile project's SenseAT::assert_sense_out().
    digitalWrite(PIN_ROW_SENSE, LOW);
    pinMode(PIN_ROW_SENSE, OUTPUT);
}

void RowSenseRP2350::release_out() {
    // Return to input with pull-up so the line is held high.
    pinMode(PIN_ROW_SENSE, INPUT_PULLUP);
}
