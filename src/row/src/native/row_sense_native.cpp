#include "row_sense_native.h"
#include "broker_msg.h"
#include <unistd.h>

void RowSenseNative::assert_out() {
    uint8_t byte = (uint8_t)BrokerMsg::SENSE_ASSERT;
    write(fd, &byte, 1);
}

void RowSenseNative::release_out() {
    uint8_t byte = (uint8_t)BrokerMsg::SENSE_RELEASE;
    write(fd, &byte, 1);
}
