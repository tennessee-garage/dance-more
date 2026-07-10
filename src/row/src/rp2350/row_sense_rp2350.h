#pragma once
#include "row_sense_control.h"

class RowSenseRP2350 : public IRowSenseControl {
public:
    void init();                  // configure PIN_ROW_SENSE, start released
    void assert_out() override;   // drive low
    void release_out() override;  // release (input, pulled up)
};
