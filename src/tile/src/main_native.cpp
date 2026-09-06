#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <ctime>
#include "transport.h"
#include "led_driver.h"
#include "sense.h"
#include "command_handler.h"
#include "pattern.h"
#include "protocol.h"
#include "native/transport_native.h"
#include "native/led_driver_native.h"
#include "native/sense_native.h"

// Monotonic milliseconds, matching Arduino millis() semantics for PatternEngine.
static uint32_t now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)(ts.tv_sec * 1000ULL + ts.tv_nsec / 1000000ULL);
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <tile-addr> <slot> <row>\n"
                        "  tile-addr  hex or decimal tile address (e.g. 0x0A or 10)\n"
                        "  slot       chain position 0-7\n"
                        "  row        bus row number (matches broker)\n", argv[0]);
        return 1;
    }

    // Given on the command line rather than left ADDR_UNASSIGNED as the real
    // firmware does: TransportNative takes the address at construction to
    // register with the broker, so the socket identity is fixed before
    // SET_ADDRESS could ever arrive. A SET_ADDRESS handled here still updates
    // my_addr and so changes which frames this process answers, while its
    // broker routing key does not follow - so socket-broker tests should not
    // be used to exercise address assignment. The row's own unit tests cover
    // that path (test_sense_mapper).
    uint8_t my_addr = (uint8_t)strtoul(argv[1], nullptr, 0);
    uint8_t slot    = (uint8_t)strtoul(argv[2], nullptr, 0);
    int     row     = (int)strtol(argv[3], nullptr, 10);

    SenseNative     sense;
    TransportNative transport(my_addr, slot, row);
    transport.set_sense(sense);

    LedDriverNative led_driver(my_addr);
    PixelBuffer     pixel_buf{};
    FrameParser     parser;
    PatternEngine   pattern;

    transport.init();
    sense.set_fd(transport.get_fd());
    led_driver.init();

    printf("[tile 0x%02X] slot=%u row=%d ready\n", my_addr, slot, row);

    Frame f;
    while (true) {
        while (transport.poll(parser, &f)) {
            if (f.addr != my_addr && f.addr != ADDR_BROADCAST) continue;

            const Frame *resp = handle_command(f, pixel_buf, sense, my_addr, &pattern);
            if (resp) transport.send(*resp);
        }

        const uint32_t now = now_ms();

        if (pixel_buf.latch_pending) {
            pattern.on_latch(pixel_buf, now);
            led_driver.push(pixel_buf);
            pixel_buf.latch_pending = false;
        } else if (pattern.poll(pixel_buf, now)) {
            led_driver.push(pixel_buf);
        }

        usleep(100); // 100 µs yield between poll bursts
    }
}
