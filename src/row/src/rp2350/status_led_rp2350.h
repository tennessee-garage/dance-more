#pragma once
#include "status_led.h"

class StatusLedRP2350 : public IStatusLed {
public:
    void init() override;
    void set_ready(bool on) override;
    void set_data(bool on) override;
};
