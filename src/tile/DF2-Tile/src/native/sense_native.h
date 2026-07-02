#pragma once
#include "sense.h"

class SenseNative : public ISenseControl {
    int  fd;
    bool sense_in_state = false;
public:
    explicit SenseNative(int socket_fd);
    void on_sense_in_update(bool low);
    void assert_sense_out() override;
    void release_sense_out() override;
    bool sense_is_asserted() const override;
};
