#include <Arduino.h>
#include "sense_at.h"
#include "pins.h"

void SenseAT::init() {
    // SENSE_IN: plain input; the upstream tile's pull-up holds the line high.
    pinMode(PIN_SENSE_IN, INPUT);

    // SENSE_OUT: start released, with pull-up so the line stays high.
    pinMode(PIN_SENSE_OUT, INPUT_PULLUP);
}

void SenseAT::assert_sense_out() {
    // Drive the line low first, then switch to output so the transition is
    // clean (avoids a brief HIGH glitch from the port register).
    digitalWrite(PIN_SENSE_OUT, LOW);
    pinMode(PIN_SENSE_OUT, OUTPUT);
}

void SenseAT::release_sense_out() {
    // Return to input with pull-up so this tile holds the line high.
    pinMode(PIN_SENSE_OUT, INPUT_PULLUP);
}

bool SenseAT::sense_is_asserted() const {
    return digitalRead(PIN_SENSE_IN) == LOW;
}
