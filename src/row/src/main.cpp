#include <Arduino.h>
#include "rp2350/pins.h"
#include "row_address_generated.h"
#include "rp2350/pi_transport_rp2350.h"
#include "rp2350/tile_transport_rp2350.h"
#include "rp2350/row_sense_rp2350.h"
#include "rp2350/power_monitor_rp2350.h"
#include "rp2350/row_bus_frame_queue.h"
#include "rp2350/status_led_rp2350.h"
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
static StatusLedRP2350     status_led;
static RowSenseRP2350      row_sense;
static PowerMonitorRP2350  power_monitor;
static TileMap             tile_map;
static SenseMapper         sense_mapper(tile_transport, row_sense, tile_map);
static RowCommandHandler   row_cmd_handler(tile_transport, sense_mapper, power_monitor, MY_ROW_ADDR);

// core 0 -> core 1: validated (CRC-good, addressed to us or broadcast) frames
static RowBusFrameQueue ingest_queue;
// core 1 -> core 0: RowCommandHandler responses to admin commands, bound for the Pi
static RowBusFrameQueue response_queue;

// ---- Status LEDs ----
// DATA (core 0): pulses for DATA_LED_PULSE_MS whenever a Row Bus frame
// addressed to this row is accepted, so bus traffic is visible at a glance.
// READY (core 1): solid once SENSE discovery has finished, slow-blinking
// while it runs, fast-blinking on discovery error. Blinking in the
// non-solid states doubles as a core-1-is-alive heartbeat.
//
// Each LED is driven by exactly one core, so there's no shared state to
// guard - and both are plain digitalWrite()s to distinct pins. Writes are
// gated on a change so these hot loops aren't hammering the GPIOs.

static constexpr uint32_t DATA_LED_PULSE_MS = 20;

static void drive_data_led(uint32_t now_ms, uint32_t until_ms) {
    static bool last = false;
    // Signed difference so the comparison survives a millis() rollover.
    bool want = (int32_t)(until_ms - now_ms) > 0;
    if (want != last) {
        status_led.set_data(want);
        last = want;
    }
}

static void drive_ready_led(uint32_t now_ms, SenseMapState state) {
    static bool last = false;
    bool want;
    switch (state) {
    case SenseMapState::DONE:  want = true;                    break;
    case SenseMapState::ERROR: want = (now_ms / 125) & 1;      break;
    default:                   want = (now_ms / 500) & 1;      break;
    }
    if (want != last) {
        status_led.set_ready(want);
        last = want;
    }
}

// ---- ROW_DEBUG bring-up instrumentation (env:seeed_xiao_rp2350_debug) ----
// Kept out of the normal build: this repurposes the Tile Bus UART as a
// plain debug console, which is only safe while no tiles are attached.
//
// Emitted on Serial1 (PIN_ROW_TX, GPIO0) as ASCII at the Tile Bus's
// 1 Mbps. Tile Bus XDIR is asserted around the write, exactly as
// TileTransportRP2350::send() does, so the text reaches the RS-485 pair at
// J6 and can be probed there rather than on the MCU's fine-pitch pad.
// Decode the non-inverted line of the pair as plain async serial; the
// other line carries the complement and will show as framing errors.
//
// DATA LED is also retargeted to raw byte arrival rather than accepted
// frames, so it lights even when bytes are arriving but failing CRC.
#ifdef ROW_DEBUG
#include <hardware/clocks.h>
#include <hardware/uart.h>

#ifndef ROW_BUS_BAUD
#define ROW_BUS_BAUD 3125000UL
#endif

static volatile uint32_t dbg_accepted = 0;  // frames that passed the address filter
static volatile uint32_t dbg_dropped  = 0;  // accepted but ingest_queue was full
static volatile uint32_t dbg_responses = 0; // responses handed back to core 0

static void debug_report(uint32_t now_ms, SenseMapState state) {
    static uint32_t next_ms = 0;
    if ((int32_t)(now_ms - next_ms) < 0) return;
    next_ms = now_ms + 1000;

    // Key up the Tile Bus transceiver for the duration of the write, then
    // release it - safe only because no tiles share this bus yet.
    digitalWrite(PIN_ROW_XDIR, HIGH);

    // Actual achieved Row Bus baud, straight from UART1's divisor registers
    // rather than the rate we asked for - the two differ silently when the
    // requested rate isn't reachable from clk_peri, which is exactly the
    // failure this instrumentation exists to catch.
    uint32_t periclk = clock_get_hz(clk_peri);
    uint32_t ibrd    = uart_get_hw(uart1)->ibrd;
    uint32_t fbrd    = uart_get_hw(uart1)->fbrd;
    uint32_t actual_baud = (64u * ibrd + fbrd)
                         ? (uint32_t)(((uint64_t)periclk * 4u) / (64u * ibrd + fbrd))
                         : 0;

    Serial1.printf("[row 0x%02X] clk_peri=%lu baud_set=%lu baud_actual=%lu\r\n",
                   (unsigned)MY_ROW_ADDR, (unsigned long)periclk,
                   (unsigned long)ROW_BUS_BAUD, (unsigned long)actual_baud);

    Serial1.printf("[row 0x%02X] up=%lus rx_bytes=%lu rx_frames=%lu accepted=%lu "
                   "dropped=%lu responses=%lu tx_frames=%lu sense=%u\r\n",
                   (unsigned)MY_ROW_ADDR, (unsigned long)(now_ms / 1000),
                   (unsigned long)pi_transport.rx_bytes,
                   (unsigned long)pi_transport.rx_frames,
                   (unsigned long)dbg_accepted,
                   (unsigned long)dbg_dropped,
                   (unsigned long)dbg_responses,
                   (unsigned long)pi_transport.tx_frames,
                   (unsigned)state);

    Serial1.flush();  // wait for the last stop bit before releasing the line
    digitalWrite(PIN_ROW_XDIR, LOW);
}
#endif

// ---- Core 0 : Row Bus ingest ----

void setup() {
    status_led.init();
    pi_transport.init();
}

void loop() {
    static RowBusFrameParser parser;
    static uint32_t data_led_until_ms = 0;
    RowBusFrame frame;

    uint32_t now = millis();

    while (pi_transport.poll(parser, &frame)) {
        if (frame.addr != MY_ROW_ADDR && frame.addr != ROWBUS_ADDR_BROADCAST) continue;
        bool queued = ingest_queue.try_push(frame); // dropped if core 1 has fallen behind
        data_led_until_ms = now + DATA_LED_PULSE_MS;
#ifdef ROW_DEBUG
        dbg_accepted++;
        if (!queued) dbg_dropped++;
#else
        (void)queued;
#endif
    }

#ifdef ROW_DEBUG
    // Raw-byte activity, not accepted frames: lights even if every frame is
    // failing CRC, which is the distinction we're chasing during bring-up.
    static uint32_t last_rx_bytes = 0;
    if (pi_transport.rx_bytes != last_rx_bytes) {
        last_rx_bytes = pi_transport.rx_bytes;
        data_led_until_ms = now + DATA_LED_PULSE_MS;
    }
#endif

    drive_data_led(now, data_led_until_ms);

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
    drive_ready_led(millis(), sense_mapper.state());

    RowBusFrame frame;
    while (ingest_queue.try_pop(&frame)) {
        const RowBusFrame *response = row_cmd_handler.handle(frame);
        if (response) {
            response_queue.try_push(*response);
#ifdef ROW_DEBUG
            dbg_responses++;
#endif
        }
    }

#ifdef ROW_DEBUG
    debug_report(millis(), sense_mapper.state());
#endif

    // Advances any in-flight SEND_DATA forwarding by one tile slot. Runs
    // after draining the queue above so that a LATCH already queued behind
    // a SEND_DATA in the same batch is dispatched (and, if forwarding isn't
    // done yet, deferred - see #46) before any slots advance this iteration.
    row_cmd_handler.poll(millis());
}
