#pragma once
#include "status_led.h"

class StatusLedNative : public IStatusLed {
    bool ready_on = false;
    bool data_on = false;
public:
    void init() override;
    void set_ready(bool on) override;
    void set_data(bool on) override;
};
