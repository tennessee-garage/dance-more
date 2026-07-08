#include <Arduino.h>
#include "rp2350/power_monitor_rp2350.h"

// Standalone INA220 connectivity test — polls the power monitor once a
// second and prints readings to Serial.

static PowerMonitorRP2350 power_monitor;

void setup() {
    Serial.begin(115200);
    power_monitor.init();
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
    delay(1000);
}
