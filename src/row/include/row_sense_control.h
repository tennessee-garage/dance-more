#pragma once

class IRowSenseControl {
public:
    virtual ~IRowSenseControl() = default;
    virtual void assert_out() = 0;    // drive SENSE toward slot 0 low
    virtual void release_out() = 0;   // release (high-Z)
};
