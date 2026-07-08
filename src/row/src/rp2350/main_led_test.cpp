#include <Arduino.h>
#include "status_led_rp2350.h"

// Standalone status-LED connectivity test — no protocol, no RS-485.
// Blinks PIN_LED_READY and PIN_LED_DATA independently, then together,
// so each LED's (active-low) wiring can be visually confirmed.

static constexpr uint16_t STEP_MS = 300;
static constexpr uint8_t  REPS    = 5;

static StatusLedRP2350 status_led;

static void blink(bool ready, bool data, uint8_t reps) {
    for (uint8_t i = 0; i < reps; i++) {
        status_led.set_ready(ready);
        status_led.set_data(data);
        delay(STEP_MS);
        status_led.set_ready(false);
        status_led.set_data(false);
        delay(STEP_MS);
    }
}

void setup() {
    status_led.init();
}

void loop() {
    blink(true, false, REPS);   // READY only
    blink(false, true, REPS);   // DATA only
    for (uint8_t i = 0; i < REPS; i++) {   // alternate READY/DATA
        status_led.set_ready(true);
        status_led.set_data(false);
        delay(STEP_MS);
        status_led.set_ready(false);
        status_led.set_data(true);
        delay(STEP_MS);
    }
    status_led.set_ready(false);
    status_led.set_data(false);
    blink(true, true, REPS);    // both together
    status_led.set_ready(false);
    status_led.set_data(false);
    delay(1000);
}
