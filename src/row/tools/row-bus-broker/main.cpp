// row-bus-broker: models the shared RS-485 multidrop Row Bus as a Unix
// socket.
//
// Up to 9 clients connect (8 row controllers + 1 mock Pi). Every Row Bus
// frame received from any client is forwarded to all OTHER clients,
// mirroring RS-485 broadcast. Each client filters by ADDR itself, exactly
// as real hardware would - the broker does no address-based routing.
//
// Row Bus has no SENSE sideband (row addresses are static, no auto-mapping),
// so unlike tile_bus_protocol's tile-bus-broker this is plain frame relay
// only.
//
// Usage: row-bus-broker
//   Listens on /tmp/df2-row-bus.sock

#include <array>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <csignal>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/select.h>

#include "row_broker_msg.h"
#include "row_bus_protocol.h"

static constexpr int MAX_CLIENTS = 9;
static constexpr const char *SOCK_PATH = "/tmp/df2-row-bus.sock";

// SIGINT/SIGTERM (e.g. the test harness terminating this process) should
// still clean up the socket file rather than leaving a stale one behind.
static void handle_signal(int) {
    unlink(SOCK_PATH);
    _exit(0);
}

struct Client {
    int     fd         = -1;
    uint8_t row_addr   = 0xFF;
    bool    identified = false;

    enum class State : uint8_t { TYPE, FRAME_LEN_H, FRAME_LEN_L, FRAME_DATA, IDENT_ADDR };
    State    state     = State::TYPE;
    uint8_t  frame_buf[ROWBUS_MAX_FRAME]{};
    uint16_t frame_len = 0;
    uint16_t frame_idx = 0;

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
    uint8_t header[3] = {
        (uint8_t)RowBrokerMsg::FRAME,
        (uint8_t)(s.frame_len >> 8),
        (uint8_t)(s.frame_len & 0xFF),
    };
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (i == sender_idx || !clients[i].active()) continue;
        send_to(clients[i].fd, header, sizeof(header));
        send_to(clients[i].fd, s.frame_buf, s.frame_len);
    }
}

// Returns false if the client has disconnected.
static bool process_byte(int idx, uint8_t byte) {
    Client &c = clients[idx];
    switch (c.state) {
    case Client::State::TYPE:
        switch (static_cast<RowBrokerMsg>(byte)) {
        case RowBrokerMsg::FRAME:
            c.state     = Client::State::FRAME_LEN_H;
            c.frame_idx = 0;
            break;
        case RowBrokerMsg::IDENTIFY:
            c.state = Client::State::IDENT_ADDR;
            break;
        default:
            break;
        }
        break;

    case Client::State::FRAME_LEN_H:
        c.frame_len = (uint16_t)(byte << 8);
        c.state     = Client::State::FRAME_LEN_L;
        break;

    case Client::State::FRAME_LEN_L:
        c.frame_len |= byte;
        c.frame_idx  = 0;
        c.state      = (c.frame_len == 0) ? Client::State::TYPE : Client::State::FRAME_DATA;
        break;

    case Client::State::FRAME_DATA:
        if (c.frame_idx < ROWBUS_MAX_FRAME) c.frame_buf[c.frame_idx] = byte;
        if (++c.frame_idx >= c.frame_len) {
            broadcast_frame(idx);
            c.state = Client::State::TYPE;
        }
        break;

    case Client::State::IDENT_ADDR:
        c.row_addr   = byte;
        c.identified = true;
        c.state      = Client::State::TYPE;
        printf("[broker] client fd=%d  row_addr=0x%02X\n", c.fd, c.row_addr);
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

int main() {
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    unlink(SOCK_PATH); // clear stale socket from a previous run

    int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return 1; }

    struct sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCK_PATH, sizeof(addr.sun_path) - 1);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }
    if (listen(server_fd, MAX_CLIENTS) < 0) { perror("listen"); return 1; }

    printf("[broker] listening on %s\n", SOCK_PATH);
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
                printf("[broker] client fd=%d row_addr=0x%02X disconnected\n", c.fd, c.row_addr);
                fflush(stdout);
                close(c.fd);
                c.reset();
            }
        }
    }
}
