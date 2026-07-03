// tile-bus-broker: models the shared RS-485 multidrop bus as a Unix socket.
//
// Up to 9 clients connect (8 tiles + 1 row controller). Every raw tile-bus
// frame received from any client is forwarded to all OTHER clients, mirroring
// RS-485 broadcast. SENSE sideband messages are routed between adjacent slots:
// an assertion from slot N wakes slot N+1 (uint8_t wrap, so the row controller
// at slot 0xFF drives tile slot 0).
//
// Usage: tile-bus-broker <row>
//   Listens on /tmp/df2-tile-bus-<row>.sock

#include <array>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/select.h>

#include "broker_msg.h"
#include "protocol.h"

static constexpr int MAX_CLIENTS = 9;

struct Client {
    int     fd           = -1;
    uint8_t slot         = 0xFF;
    uint8_t tile_addr    = 0x00;
    bool    identified   = false;

    enum class State : uint8_t { TYPE, FRAME_LEN, FRAME_DATA, IDENT_ADDR, IDENT_SLOT };
    State   state        = State::TYPE;
    uint8_t frame_buf[MAX_FRAME_SIZE]{};
    uint8_t frame_len    = 0;
    uint8_t frame_idx    = 0;
    uint8_t ident_addr   = 0;

    bool active() const { return fd >= 0; }
    void reset() { *this = Client{}; }
};

static std::array<Client, MAX_CLIENTS> clients;

static void send_to(int fd, const uint8_t *data, int len) {
    while (len > 0) {
        ssize_t n = write(fd, data, len);
        if (n <= 0) return;
        data += n;
        len  -= n;
    }
}

static void broadcast_frame(int sender_idx) {
    Client &s = clients[sender_idx];
    uint8_t header[2] = {(uint8_t)BrokerMsg::FRAME, s.frame_len};
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (i == sender_idx || !clients[i].active()) continue;
        send_to(clients[i].fd, header, 2);
        send_to(clients[i].fd, s.frame_buf, s.frame_len);
    }
}

static void route_sense(int sender_idx, BrokerMsg msg) {
    uint8_t next_slot = (uint8_t)(clients[sender_idx].slot + 1); // wraps: 0xFF → 0x00
    uint8_t byte = (uint8_t)msg;
    for (auto &c : clients) {
        if (c.active() && c.identified && c.slot == next_slot) {
            send_to(c.fd, &byte, 1);
            return;
        }
    }
}

// Returns false if the client has disconnected.
static bool process_byte(int idx, uint8_t byte) {
    Client &c = clients[idx];
    switch (c.state) {
    case Client::State::TYPE:
        switch (static_cast<BrokerMsg>(byte)) {
        case BrokerMsg::FRAME:
            c.state = Client::State::FRAME_LEN;
            c.frame_idx = 0;
            break;
        case BrokerMsg::IDENTIFY:
            c.state = Client::State::IDENT_ADDR;
            break;
        case BrokerMsg::SENSE_ASSERT:
            if (c.identified) route_sense(idx, BrokerMsg::SENSE_IS_ASSERTED);
            break;
        case BrokerMsg::SENSE_RELEASE:
            if (c.identified) route_sense(idx, BrokerMsg::SENSE_DEASSERTED);
            break;
        default:
            break;
        }
        break;

    case Client::State::FRAME_LEN:
        c.frame_len = byte;
        c.frame_idx = 0;
        c.state     = (byte == 0) ? Client::State::TYPE : Client::State::FRAME_DATA;
        break;

    case Client::State::FRAME_DATA:
        if (c.frame_idx < MAX_FRAME_SIZE) c.frame_buf[c.frame_idx] = byte;
        if (++c.frame_idx >= c.frame_len) {
            broadcast_frame(idx);
            c.state = Client::State::TYPE;
        }
        break;

    case Client::State::IDENT_ADDR:
        c.ident_addr = byte;
        c.state      = Client::State::IDENT_SLOT;
        break;

    case Client::State::IDENT_SLOT:
        c.slot       = byte;
        c.tile_addr  = c.ident_addr;
        c.identified = true;
        c.state      = Client::State::TYPE;
        printf("[broker] client fd=%d  addr=0x%02X  slot=%u\n", c.fd, c.tile_addr, c.slot);
        fflush(stdout);
        break;
    }
    return true;
}

// Drain all available bytes from a client. Returns false if disconnected.
static bool drain_client(int idx) {
    uint8_t byte;
    while (true) {
        ssize_t n = recv(clients[idx].fd, &byte, 1, MSG_DONTWAIT);
        if (n == 1) {
            process_byte(idx, byte);
        } else if (n == 0) {
            return false; // peer closed
        } else {
            if (errno == EAGAIN || errno == EWOULDBLOCK) return true;
            return false; // error
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <row>\n", argv[0]);
        return 1;
    }
    int row = atoi(argv[1]);

    char path[64];
    snprintf(path, sizeof(path), "/tmp/df2-tile-bus-%d.sock", row);
    unlink(path); // clear stale socket from a previous run

    int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return 1; }

    struct sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }
    if (listen(server_fd, MAX_CLIENTS) < 0) { perror("listen"); return 1; }

    printf("[broker] row=%d  listening on %s\n", row, path);
    fflush(stdout);

    while (true) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(server_fd, &rfds);
        int max_fd = server_fd;

        for (auto &c : clients) {
            if (c.active()) {
                FD_SET(c.fd, &rfds);
                max_fd = std::max(max_fd, c.fd);
            }
        }

        if (select(max_fd + 1, &rfds, nullptr, nullptr, nullptr) < 0) {
            if (errno == EINTR) continue;
            perror("select");
            return 1;
        }

        // Accept new connections
        if (FD_ISSET(server_fd, &rfds)) {
            int new_fd = accept(server_fd, nullptr, nullptr);
            if (new_fd >= 0) {
                bool added = false;
                for (auto &c : clients) {
                    if (!c.active()) { c.fd = new_fd; added = true; break; }
                }
                if (!added) {
                    fprintf(stderr, "[broker] max clients reached, rejecting fd=%d\n", new_fd);
                    close(new_fd);
                }
            }
        }

        // Service connected clients
        for (int i = 0; i < MAX_CLIENTS; i++) {
            Client &c = clients[i];
            if (!c.active() || !FD_ISSET(c.fd, &rfds)) continue;
            if (!drain_client(i)) {
                printf("[broker] client fd=%d slot=%u disconnected\n", c.fd, c.slot);
                fflush(stdout);
                close(c.fd);
                c.reset();
            }
        }
    }
}
