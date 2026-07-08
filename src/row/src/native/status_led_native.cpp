#include "status_led_native.h"
#include <cstdio>

void StatusLedNative::init() {
    printf("[status] LED driver ready (native/log mode)\n");
    fflush(stdout);
}

void StatusLedNative::set_ready(bool on) {
    if (on == ready_on) return;
    ready_on = on;
    printf("[status] READY=%s\n", ready_on ? "ON" : "OFF");
    fflush(stdout);
}

void StatusLedNative::set_data(bool on) {
    if (on == data_on) return;
    data_on = on;
    printf("[status] DATA=%s\n", data_on ? "ON" : "OFF");
    fflush(stdout);
}
