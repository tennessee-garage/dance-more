#include "sense_native.h"

// Implemented in issue #10.
SenseNative::SenseNative(int socket_fd) : fd(socket_fd) {}

void SenseNative::on_sense_in_update(bool low) { sense_in_state = low; }
void SenseNative::assert_sense_out() {}
void SenseNative::release_sense_out() {}
bool SenseNative::sense_is_asserted() const { return sense_in_state; }
