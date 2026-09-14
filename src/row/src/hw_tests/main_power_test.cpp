#include <Arduino.h>
#include "rp2350/power_monitor_rp2350.h"
#include "rp2350/status_led_rp2350.h"

// Standalone INA226 connectivity test — polls the power monitor once a
// second and prints readings to Serial.

static PowerMonitorRP2350 power_monitor;
static StatusLedRP2350 status_led;

static constexpr uint16_t STEP_MS = 300;
static constexpr uint8_t  REPS    = 2;

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
    Serial.begin(115200);
    power_monitor.init();
    status_led.init();
}

void loop() {
    PowerReading r = power_monitor.read();
    Serial.print("voltage=");
    Serial.print(r.voltage_mV);
    Serial.print("mV current=");
    Serial.print(r.current_mA);
    Serial.print("mA power=");
    Serial.print(r.power_mW);
    Serial.println("mW");
    blink(true, false, REPS);   // READY only
    blink(false, true, REPS);   // DATA only
    delay(1000);
}
