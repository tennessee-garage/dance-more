#include <Arduino.h>
#include "rp2350/pins.h"
#include "row_address_generated.h"
#include "rp2350/pi_transport_rp2350.h"
#include "rp2350/tile_transport_rp2350.h"
#include "rp2350/row_sense_rp2350.h"
#include "rp2350/power_monitor_rp2350.h"
#include "rp2350/row_bus_frame_queue.h"
#include "sense_mapper.h"
#include "row_command_handler.h"

// Core 0 - Row Bus ingest: owns PiTransportRP2350, validates and address-
// filters incoming frames, forwards responses back out over Row Bus.
// Core 1 - Tile Bus egress + dispatch: owns everything Tile-Bus-facing and
// SenseMapper, consumes frames handed off from core 0 via RowCommandHandler.
//
// Global objects are constructed (by normal C++ static init) before either
// core's setup()/setup1() runs - see framework-arduinopico's main(), which
// launches core 1 only after core 0's static init and USB/serial bring-up -
// so both cores always see fully-constructed queues/objects, no lazy-init
// ordering to worry about.

static PiTransportRP2350   pi_transport;
static TileTransportRP2350 tile_transport;
static RowSenseRP2350      row_sense;
static PowerMonitorRP2350  power_monitor;
static TileMap             tile_map;
static SenseMapper         sense_mapper(tile_transport, row_sense, tile_map);
static RowCommandHandler   row_cmd_handler(tile_transport, sense_mapper, power_monitor, MY_ROW_ADDR);

// core 0 -> core 1: validated (CRC-good, addressed to us or broadcast) frames
static RowBusFrameQueue ingest_queue;
// core 1 -> core 0: RowCommandHandler responses to admin commands, bound for the Pi
static RowBusFrameQueue response_queue;

// ---- Core 0 : Row Bus ingest ----

void setup() {
    pi_transport.init();
}

void loop() {
    static RowBusFrameParser parser;
    RowBusFrame frame;

    while (pi_transport.poll(parser, &frame)) {
        if (frame.addr != MY_ROW_ADDR && frame.addr != ROWBUS_ADDR_BROADCAST) continue;
        ingest_queue.try_push(frame); // dropped if core 1 has fallen behind
    }

    RowBusFrame response;
    while (response_queue.try_pop(&response)) {
        pi_transport.send(response);
    }
}

// ---- Core 1 : Tile Bus egress + dispatch ----

void setup1() {
    tile_transport.init();
    row_sense.init();
    power_monitor.init();
    sense_mapper.start();
}

void loop1() {
    sense_mapper.poll(millis());

    RowBusFrame frame;
    while (ingest_queue.try_pop(&frame)) {
        const RowBusFrame *response = row_cmd_handler.handle(frame);
        if (response) response_queue.try_push(*response);
    }
}
