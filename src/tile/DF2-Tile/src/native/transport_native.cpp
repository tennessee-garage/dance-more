#include "transport_native.h"
#include "sense_native.h"
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <cstdlib>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>

TransportNative::TransportNative(uint8_t addr, uint8_t s, int r)
    : tile_addr(addr), slot(s), row(r) {}

void TransportNative::init() {
    fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); exit(1); }

    char path[64];
    snprintf(path, sizeof(path), "/tmp/df2-tile-bus-%d.sock", row);

    struct sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("connect");
        exit(1);
    }

    uint8_t msg[3] = {(uint8_t)BrokerMsg::IDENTIFY, tile_addr, slot};
    write(fd, msg, sizeof(msg));
}

// Reads one byte from the socket (non-blocking). Drives the broker-protocol
// state machine; feeds frame bytes to parser. Returns true exactly once per
// complete tile-bus frame received, filling *out.
bool TransportNative::poll(FrameParser &parser, Frame *out) {
    uint8_t byte;
    if (recv(fd, &byte, 1, MSG_DONTWAIT) != 1) return false;

    switch (rx_state) {
    case RxState::TYPE:
        switch (static_cast<BrokerMsg>(byte)) {
        case BrokerMsg::FRAME:
            rx_state = RxState::FRAME_LEN;
            break;
        case BrokerMsg::SENSE_IS_ASSERTED:
            if (sense) sense->on_sense_in_update(true);
            break;
        case BrokerMsg::SENSE_DEASSERTED:
            if (sense) sense->on_sense_in_update(false);
            break;
        default:
            break;
        }
        return false;

    case RxState::FRAME_LEN:
        rx_frame_remaining = byte;
        rx_state = (byte == 0) ? RxState::TYPE : RxState::FRAME_DATA;
        return false;

    case RxState::FRAME_DATA: {
        bool complete = parser.feed(byte, out);
        if (--rx_frame_remaining == 0) rx_state = RxState::TYPE;
        return complete;
    }
    }
    return false;
}

void TransportNative::send(const Frame &frame) {
    uint8_t raw[MAX_FRAME_SIZE];
    int n = frame_encode(frame, raw, sizeof(raw));
    if (n < 0) return;

    uint8_t header[2] = {(uint8_t)BrokerMsg::FRAME, (uint8_t)n};
    write(fd, header, sizeof(header));
    write(fd, raw, n);
}
