#include <Arduino.h>
#include "protocol.h"
#include "rp2350/tile_transport_rp2350.h"
#include "rp2350/row_sense_rp2350.h"
#include "sense_mapper.h"

// Row-wide bring-up test: runs SENSE auto-mapping to discover as many tiles
// as are actually wired (0-8, whatever's really there), then cycles display
// commands to every discovered tile forever: SET_COLOR, then SET_LEDS, each
// followed by the mandatory LATCH. SET_PATTERN is out of scope (no pattern
// library defined yet).

static TileTransportRP2350 transport;
static RowSenseRP2350      sense;
static TileMap             tile_map;
static SenseMapper         mapper(transport, sense, tile_map);

static constexpr uint32_t DISCOVERY_GIVEUP_MS = 5000; // safety net; the state
    // machine itself converges in well under this via its own retry budget.

static void send_unicast(uint8_t addr, Cmd cmd, const uint8_t *payload, uint8_t len) {
    Frame f{};
    f.addr = addr;
    f.cmd  = (uint8_t)cmd;
    f.len  = len;
    for (uint8_t i = 0; i < len; i++) f.payload[i] = payload[i];
    transport.send(f);
}

static void broadcast_latch() {
    Frame f{};
    f.addr = ADDR_BROADCAST;
    f.cmd  = (uint8_t)Cmd::LATCH;
    f.len  = 0;
    transport.send(f);
}

static void run_discovery() {
    Serial.println("Running SENSE auto-mapping...");
    mapper.start();
    uint32_t start = millis();
    while (mapper.state() == SenseMapState::DISCOVERING &&
           millis() - start < DISCOVERY_GIVEUP_MS) {
        mapper.poll(millis());
    }

    const TileMap &result = mapper.result();
    if (mapper.state() == SenseMapState::ERROR) {
        Serial.println("SENSE mapping FAILED (a discovered tile did not ACK ACTIVATE_SENSE)");
    }

    Serial.print("Discovered ");
    Serial.print(result.discovered_count());
    Serial.println(" tile(s):");
    for (uint8_t slot = 0; slot < TileMap::NUM_SLOTS; slot++) {
        if (!result.is_discovered(slot)) continue;
        Serial.print("  slot ");
        Serial.print(slot);
        Serial.print(" -> addr 0x");
        Serial.print(result.address_for(slot), HEX);
        if (result.retry_count(slot) > 0) {
            Serial.print(" (");
            Serial.print(result.retry_count(slot));
            Serial.print(" retries)");
        }
        Serial.println();
    }
    Serial.println();
}

struct Color { uint8_t r, g, b; };

static constexpr Color TEST_COLORS[] = {
    {255, 0,   0},   // red
    {0,   255, 0},   // green
    {0,   0,   255}, // blue
    {255, 255, 255}, // white
};
static constexpr uint16_t COLOR_HOLD_MS = 1000;

static void run_set_color_phase() {
    const TileMap &result = mapper.result();
    Serial.println("[SET_COLOR] cycling test colors...");

    for (const Color &c : TEST_COLORS) {
        uint8_t payload[3] = {c.r, c.g, c.b};
        for (uint8_t slot = 0; slot < TileMap::NUM_SLOTS; slot++) {
            if (!result.is_discovered(slot)) continue;
            send_unicast(result.address_for(slot), Cmd::SET_COLOR, payload, sizeof(payload));
        }
        broadcast_latch();
        delay(COLOR_HOLD_MS);
    }
}

static constexpr uint16_t LEDS_HOLD_MS = 3000;
static constexpr uint8_t  NUM_LEDS     = 40;

static void run_set_leds_phase() {
    const TileMap &result = mapper.result();
    Serial.println("[SET_LEDS] displaying gradient...");

    // Linear gradient from red (LED 0) to blue (LED 39) - deterministic and
    // easy to verify by eye that every one of the 40 LEDs got its own
    // distinct value (i.e. the full 120-byte SET_LEDS payload arrived
    // intact, not truncated).
    uint8_t payload[NUM_LEDS * 3];
    for (uint8_t i = 0; i < NUM_LEDS; i++) {
        uint8_t r = (uint8_t)(255 - (255 * i) / (NUM_LEDS - 1));
        uint8_t b = (uint8_t)((255 * i) / (NUM_LEDS - 1));
        payload[i * 3 + 0] = r;
        payload[i * 3 + 1] = 0;
        payload[i * 3 + 2] = b;
    }

    for (uint8_t slot = 0; slot < TileMap::NUM_SLOTS; slot++) {
        if (!result.is_discovered(slot)) continue;
        send_unicast(result.address_for(slot), Cmd::SET_LEDS, payload, sizeof(payload));
    }
    broadcast_latch();
    delay(LEDS_HOLD_MS);
}

void setup() {
    Serial.begin(115200);
    // Native USB serial: begin() returns immediately, before the host has
    // actually opened the port. Anything printed before that happens is
    // silently dropped - block here so setup's report isn't lost.
    while (!Serial) delay(10);

    transport.init();
    sense.init();
    Serial.println("Row test - discovers all wired tiles, then cycles SET_COLOR/SET_LEDS");
    Serial.println();

    run_discovery();
}

void loop() {
    if (tile_map.discovered_count() == 0) {
        Serial.println("0 tiles discovered - nothing to display. Check wiring and reset.");
        delay(5000);
        return;
    }

    run_set_color_phase();
    run_set_leds_phase();
}
