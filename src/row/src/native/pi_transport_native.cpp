#include "pi_transport_native.h"
#include "row_broker_msg.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>

static constexpr const char *SOCK_PATH = "/tmp/df2-row-bus.sock";

PiTransportNative::PiTransportNative(uint8_t row_addr) : row_addr(row_addr) {}

void PiTransportNative::init() {
    fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); exit(1); }

    struct sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCK_PATH, sizeof(addr.sun_path) - 1);

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("connect");
        exit(1);
    }

    uint8_t msg[2] = {(uint8_t)RowBrokerMsg::IDENTIFY, row_addr};
    write(fd, msg, sizeof(msg));
}

// Reads one byte from the socket (non-blocking). Drives the broker-protocol
// state machine; feeds frame bytes to parser. Returns true exactly once per
// complete Row Bus frame received, filling *out.
bool PiTransportNative::poll(RowBusFrameParser &parser, RowBusFrame *out) {
    uint8_t byte;
    if (recv(fd, &byte, 1, MSG_DONTWAIT) != 1) return false;

    switch (rx_state) {
    case RxState::TYPE:
        switch (static_cast<RowBrokerMsg>(byte)) {
        case RowBrokerMsg::FRAME:
            rx_state = RxState::FRAME_LEN_H;
            break;
        default:
            break;
        }
        return false;

    case RxState::FRAME_LEN_H:
        rx_frame_len = (uint16_t)(byte << 8);
        rx_state     = RxState::FRAME_LEN_L;
        return false;

    case RxState::FRAME_LEN_L:
        rx_frame_len |= byte;
        rx_frame_remaining = rx_frame_len;
        rx_state = (rx_frame_len == 0) ? RxState::TYPE : RxState::FRAME_DATA;
        return false;

    case RxState::FRAME_DATA: {
        bool complete = parser.feed(byte, out);
        if (--rx_frame_remaining == 0) rx_state = RxState::TYPE;
        return complete;
    }
    }
    return false;
}

void PiTransportNative::send(const RowBusFrame &frame) {
    uint8_t raw[ROWBUS_MAX_FRAME];
    int n = row_bus_frame_encode(frame, raw, sizeof(raw));
    if (n < 0) return;

    uint8_t header[3] = {(uint8_t)RowBrokerMsg::FRAME, (uint8_t)(n >> 8), (uint8_t)(n & 0xFF)};
    write(fd, header, sizeof(header));
    write(fd, raw, n);
}
