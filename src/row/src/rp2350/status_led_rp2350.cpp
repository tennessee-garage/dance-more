#include <Arduino.h>
#include "status_led_rp2350.h"
#include "pins.h"

// Both LEDs are active-low: the GPIO sinks current through the LED to
// light it. LOW = on, HIGH = off.

void StatusLedRP2350::init() {
    pinMode(PIN_LED_READY, OUTPUT);
    pinMode(PIN_LED_DATA, OUTPUT);
    digitalWrite(PIN_LED_READY, HIGH);
    digitalWrite(PIN_LED_DATA, HIGH);
}

void StatusLedRP2350::set_ready(bool on) {
    digitalWrite(PIN_LED_READY, on ? LOW : HIGH);
}

void StatusLedRP2350::set_data(bool on) {
    digitalWrite(PIN_LED_DATA, on ? LOW : HIGH);
}
