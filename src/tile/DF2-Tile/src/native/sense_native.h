#pragma once
#include "sense.h"
#include <stdint.h>

class SenseNative : public ISenseControl {
    int  fd             = -1;
    bool sense_in_state = false;
public:
    SenseNative() = default;
    explicit SenseNative(int socket_fd);

    void set_fd(int socket_fd) { fd = socket_fd; }

    // Called by TransportNative::poll() when a SENSE sideband message arrives.
    void on_sense_in_update(bool asserted);

    void assert_sense_out() override;
    void release_sense_out() override;
    bool sense_is_asserted() const override;
};
