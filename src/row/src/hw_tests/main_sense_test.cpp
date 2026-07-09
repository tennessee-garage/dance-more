#include <Arduino.h>
#include "protocol.h"
#include "rp2350/tile_transport_rp2350.h"
#include "rp2350/row_sense_rp2350.h"

// Standalone SENSE-chain test — assumes exactly one tile wired directly to
// the row controller (slot 0). Exercises the same DETECT_SENSE /
// ACTIVATE_SENSE / CLEAR_SENSE sequence SenseMapper (lib/row_core) runs for
// real auto-mapping, one tile at a time:
//   1. Row controller asserts its own SENSE, broadcasts DETECT_SENSE, waits
//      for the tile's DETECT_RESP.
//   2. Row controller unicasts ACTIVATE_SENSE to the discovered tile, waits
//      for its ACK.
//   3. Row controller broadcasts CLEAR_SENSE (cleanup).
//
// At startup, and after every run, choose [r]un (immediate, auto) or [s]tep
// (pauses for [c]ontinue after steps 1 and 2, so the physical SENSE line can
// be probed with a meter/scope between them).

static TileTransportRP2350 transport;
static RowSenseRP2350      sense;
static FrameParser         parser;

static constexpr uint32_t RESPONSE_TIMEOUT_MS = 200; // generous for manual/bench use

static void send_broadcast(Cmd cmd) {
    Frame f{};
    f.addr = ADDR_BROADCAST;
    f.cmd  = (uint8_t)cmd;
    f.len  = 0;
    transport.send(f);
}

static void send_unicast(uint8_t addr, Cmd cmd) {
    Frame f{};
    f.addr = addr;
    f.cmd  = (uint8_t)cmd;
    f.len  = 0;
    transport.send(f);
}

// Waits up to RESPONSE_TIMEOUT_MS for a frame whose cmd == expected_cmd.
static bool wait_for(uint8_t expected_cmd, Frame *out) {
    uint32_t start = millis();
    while (millis() - start < RESPONSE_TIMEOUT_MS) {
        if (transport.poll(parser, out) && out->cmd == expected_cmd) return true;
    }
    return false;
}

static void wait_for_continue() {
    Serial.println("  Press [c] to continue...");
    while (true) {
        while (!Serial.available()) { /* wait */ }
        char c = (char)Serial.read();
        if (c == 'c' || c == 'C') return;
    }
}

static void run_sense_test(bool step) {
    parser.reset();
    bool ok = true;
    Frame f{};

    Serial.println("[1/3] Asserting row controller's own SENSE line...");
    sense.assert_out();
    Serial.println("      Broadcasting DETECT_SENSE, waiting for DETECT_RESP...");
    send_broadcast(Cmd::DETECT_SENSE);
    if (wait_for((uint8_t)Cmd::DETECT_RESP, &f)) {
        Serial.print("      OK: tile responded, addr=0x");
        Serial.println(f.addr, HEX);
    } else {
        Serial.println("      FAIL: no DETECT_RESP received");
        ok = false;
    }

    if (step) wait_for_continue();

    uint8_t tile_addr = f.addr;
    if (ok) {
        Serial.print("[2/3] Asking tile 0x");
        Serial.print(tile_addr, HEX);
        Serial.println(" to raise its own SENSE (ACTIVATE_SENSE)...");
        send_unicast(tile_addr, Cmd::ACTIVATE_SENSE);

        static constexpr uint8_t ACTIVATE_ACK = (uint8_t)Cmd::ACK | (uint8_t)Cmd::ACTIVATE_SENSE;
        if (wait_for(ACTIVATE_ACK, &f)) {
            Serial.print("      OK: ACK received, status=0x");
            Serial.println(f.payload[0], HEX);
        } else {
            Serial.println("      FAIL: no ACK received");
            ok = false;
        }
    } else {
        Serial.println("[2/3] Skipped (no tile discovered in step 1)");
    }

    if (step) wait_for_continue();

    Serial.println("[3/3] Broadcasting CLEAR_SENSE (cleanup)...");
    sense.release_out();
    send_broadcast(Cmd::CLEAR_SENSE);

    Serial.println(ok ? "RESULT: PASS" : "RESULT: FAIL");
    Serial.println();
}

static char prompt_menu() {
    Serial.println();
    Serial.println("Press [r] to run automatically, [s] to step through interactively:");
    while (true) {
        while (!Serial.available()) { /* wait */ }
        char c = (char)Serial.read();
        if (c == 'r' || c == 'R') return 'r';
        if (c == 's' || c == 'S') return 's';
    }
}

void setup() {
    Serial.begin(115200);
    transport.init();
    sense.init();
    Serial.println("SENSE chain test - assumes exactly one tile wired at slot 0");
}

void loop() {
    char choice = prompt_menu();
    run_sense_test(choice == 's');
}
