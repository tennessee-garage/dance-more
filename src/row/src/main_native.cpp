#include <cstdio>
#include <cstdlib>
#include <chrono>
#include "native/tile_transport_native.h"
#include "native/row_sense_native.h"
#include "sense_mapper.h"

static uint32_t now_ms() {
    using namespace std::chrono;
    return (uint32_t)duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <row>\n"
                        "  row  Tile Bus broker row number (matches tile-bus-broker's <row> arg)\n",
                argv[0]);
        return 1;
    }
    int row = atoi(argv[1]);

    TileTransportNative transport(row);
    transport.init();
    RowSenseNative sense(transport.get_fd());

    TileMap     map;
    SenseMapper mapper(transport, sense, map);

    printf("[row controller] connected to broker for row %d, running SENSE discovery...\n", row);

    mapper.start();
    while (mapper.state() == SenseMapState::DISCOVERING) {
        mapper.poll(now_ms());
    }

    if (mapper.state() == SenseMapState::ERROR) {
        printf("[row controller] SENSE mapping FAILED\n");
        return 1;
    }

    printf("[row controller] discovered %u tile(s):\n", map.discovered_count());
    for (uint8_t slot = 0; slot < TileMap::NUM_SLOTS; slot++) {
        if (!map.is_discovered(slot)) continue;
        printf("  slot %u -> addr 0x%02X (retries=%u)\n",
               slot, map.address_for(slot), map.retry_count(slot));
    }

    return 0;
}
