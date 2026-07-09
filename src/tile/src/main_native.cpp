#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include "transport.h"
#include "led_driver.h"
#include "sense.h"
#include "command_handler.h"
#include "protocol.h"
#include "native/transport_native.h"
#include "native/led_driver_native.h"
#include "native/sense_native.h"

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <tile-addr> <slot> <row>\n"
                        "  tile-addr  hex or decimal tile address (e.g. 0x0A or 10)\n"
                        "  slot       chain position 0-7\n"
                        "  row        bus row number (matches broker)\n", argv[0]);
        return 1;
    }

    uint8_t my_addr = (uint8_t)strtoul(argv[1], nullptr, 0);
    uint8_t slot    = (uint8_t)strtoul(argv[2], nullptr, 0);
    int     row     = (int)strtol(argv[3], nullptr, 10);

    SenseNative     sense;
    TransportNative transport(my_addr, slot, row);
    transport.set_sense(sense);

    LedDriverNative led_driver(my_addr);
    PixelBuffer     pixel_buf{};
    FrameParser     parser;

    transport.init();
    sense.set_fd(transport.get_fd());
    led_driver.init();

    printf("[tile 0x%02X] slot=%u row=%d ready\n", my_addr, slot, row);

    Frame f;
    while (true) {
        while (transport.poll(parser, &f)) {
            if (f.addr != my_addr && f.addr != ADDR_BROADCAST) continue;

            const Frame *resp = handle_command(f, pixel_buf, sense, my_addr);
            if (resp) transport.send(*resp);
        }

        if (pixel_buf.latch_pending) {
            led_driver.push(pixel_buf);
            pixel_buf.latch_pending = false;
        }

        usleep(100); // 100 µs yield between poll bursts
    }
}
