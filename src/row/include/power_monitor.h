#pragma once
#include <cstdint>

struct PowerReading {
    uint16_t voltage_mV;
    uint16_t current_mA;
    uint16_t power_mW;
};

class IPowerMonitor {
public:
    virtual void init() = 0;
    virtual PowerReading read() = 0;
};
