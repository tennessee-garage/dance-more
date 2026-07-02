#pragma once

class ISenseControl {
public:
    virtual ~ISenseControl() = default;
    virtual void assert_sense_out() = 0;
    virtual void release_sense_out() = 0;
    virtual bool sense_is_asserted() const = 0;
};
