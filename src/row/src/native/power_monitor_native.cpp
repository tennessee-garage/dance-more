#include "power_monitor_native.h"
#include <cmath>
#include <cstdio>

void PowerMonitorNative::init() {
    printf("[power] monitor ready (native/mock mode)\n");
    fflush(stdout);
}

PowerReading PowerMonitorNative::read() {
    ++tick;

    // Plausible 12V-rail reading with current riding a slow sine wave,
    // so common code polling this can exercise changing load without hardware.
    float current_a = 2.0f + 1.5f * sinf((float)tick * 0.1f);
    float voltage_v = 12.0f;

    PowerReading r;
    r.voltage_mV = (uint16_t)(voltage_v * 1000.0f);
    r.current_mA = (uint16_t)(current_a * 1000.0f);
    r.power_mW   = (uint16_t)(voltage_v * current_a * 1000.0f);
    return r;
}
