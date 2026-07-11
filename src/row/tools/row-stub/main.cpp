// row-stub: a minimal Row Bus client used only to test row-bus-broker +
// PiTransportNative's multidrop relay/filtering. It answers STATUS with a
// STATUS_RESP carrying its own row address in the payload (state=idle,
// discovered_count=0, all slots NOT_DISCOVERED) so a test driver can tell
// which of several connected stubs replied - it does not run real
// RowCommandHandler dispatch (that's already covered by #30's own tests).
//
// Usage: row-stub <row-addr>
//   row-addr  hex or decimal row address (0x00-0x07)

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include "pi_transport_native.h"
#include "row_bus_protocol.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <row-addr>\n", argv[0]);
        return 1;
    }
    uint8_t row_addr = (uint8_t)strtoul(argv[1], nullptr, 0);

    PiTransportNative transport(row_addr);
    transport.init();

    printf("[row-stub 0x%02X] ready\n", row_addr);
    fflush(stdout);

    RowBusFrameParser parser;
    RowBusFrame        in;
    while (true) {
        while (transport.poll(parser, &in)) {
            if (in.addr != row_addr && in.addr != ROWBUS_ADDR_BROADCAST) continue;
            if ((RowBusCmd)in.cmd != RowBusCmd::STATUS) continue;

            RowBusFrame resp{};
            resp.addr = row_addr;
            resp.cmd  = (uint8_t)RowBusCmd::STATUS_RESP;
            resp.len  = 10;
            resp.payload[0] = 0x00; // state: idle
            resp.payload[1] = 0x00; // discovered_count: 0
            for (int slot = 0; slot < 8; slot++) resp.payload[2 + slot] = 0x00;
            transport.send(resp);
        }
        usleep(100); // 100 us yield between poll bursts
    }
}
