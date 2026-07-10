#include "tile_transport_native.h"
#include "broker_msg.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>

// Row controller's fixed identity on the broker: addr=0x00, slot=0xFF (the
// 0xFF wraps to slot 0 in the broker's SENSE routing - see broker_msg.h).
static constexpr uint8_t ROW_CONTROLLER_ADDR = 0x00;
static constexpr uint8_t ROW_CONTROLLER_SLOT = 0xFF;

TileTransportNative::TileTransportNative(int r) : row(r) {}

void TileTransportNative::init() {
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

    uint8_t msg[3] = {(uint8_t)BrokerMsg::IDENTIFY, ROW_CONTROLLER_ADDR, ROW_CONTROLLER_SLOT};
    write(fd, msg, sizeof(msg));
}

// Reads one byte from the socket (non-blocking). Drives the broker-protocol
// state machine; feeds frame bytes to parser. Returns true exactly once per
// complete tile-bus frame received, filling *out.
bool TileTransportNative::poll(FrameParser &parser, Frame *out) {
    uint8_t byte;
    if (recv(fd, &byte, 1, MSG_DONTWAIT) != 1) return false;

    switch (rx_state) {
    case RxState::TYPE:
        switch (static_cast<BrokerMsg>(byte)) {
        case BrokerMsg::FRAME:
            rx_state = RxState::FRAME_LEN;
            break;
        case BrokerMsg::SENSE_IS_ASSERTED:
        case BrokerMsg::SENSE_DEASSERTED:
            // The row controller has no incoming SENSE line to update (it's
            // the head of the chain) - the broker shouldn't send these to
            // slot 0xFF anyway, but ignore them defensively if it ever did.
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

void TileTransportNative::send(const Frame &frame) {
    uint8_t raw[MAX_FRAME_SIZE];
    int n = frame_encode(frame, raw, sizeof(raw));
    if (n < 0) return;

    uint8_t header[2] = {(uint8_t)BrokerMsg::FRAME, (uint8_t)n};
    write(fd, header, sizeof(header));
    write(fd, raw, n);
}
