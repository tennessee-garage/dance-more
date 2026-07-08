#pragma once
#include "power_monitor.h"

class PowerMonitorRP2350 : public IPowerMonitor {
public:
    void init() override;
    PowerReading read() override;
};
