#pragma once
#include "sense.h"

class SenseAT : public ISenseControl {
public:
    void init();
    void assert_sense_out() override;
    void release_sense_out() override;
    bool sense_is_asserted() const override;
};
