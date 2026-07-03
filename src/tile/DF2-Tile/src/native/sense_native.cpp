#include "sense_native.h"
#include "broker_msg.h"
#include <unistd.h>

SenseNative::SenseNative(int socket_fd) : fd(socket_fd) {}

void SenseNative::on_sense_in_update(bool asserted) {
    sense_in_state = asserted;
}

void SenseNative::assert_sense_out() {
    if (fd < 0) return;
    uint8_t msg = (uint8_t)BrokerMsg::SENSE_ASSERT;
    write(fd, &msg, 1);
}

void SenseNative::release_sense_out() {
    if (fd < 0) return;
    uint8_t msg = (uint8_t)BrokerMsg::SENSE_RELEASE;
    write(fd, &msg, 1);
}

bool SenseNative::sense_is_asserted() const {
    return sense_in_state;
}
