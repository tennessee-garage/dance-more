#pragma once
#include "power_monitor.h"
#include <cstdint>

class PowerMonitorNative : public IPowerMonitor {
    uint32_t tick = 0;
public:
    void init() override;
    PowerReading read() override;
};
