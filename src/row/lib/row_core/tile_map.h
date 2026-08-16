#pragma once
#include <stdint.h>
#include "firmware_version.h" // src/common/tile_bus_protocol/, via lib_extra_dirs

// Wire-compatible with docs/row-bus-protocol.md's STATUS_RESP tile_status[] values.
enum class TileStatus : uint8_t {
    NOT_DISCOVERED = 0x00,
    OK             = 0x01,
    NON_RESPONSIVE = 0x02,
    TEST_FAILED    = 0x03,
};

struct TileSlot {
    bool       discovered  = false;
    uint8_t    address     = 0;
    uint8_t    retry_count = 0;
    TileStatus status      = TileStatus::NOT_DISCOVERED;
    // Filled by SenseMapper's post-discovery version sweep
    // (docs/row-bus-protocol.md's VERSION command); version_valid is false
    // until then, and stays false if the tile didn't answer.
    FirmwareVersion version       = {};
    bool            version_valid = false;
};

// Slot (0..7) -> discovered tile address/status/retry-count, one TileSlot per
// logical position. Array-of-structs so "everything about slot N" is a single
// index, not several parallel array lookups.
class TileMap {
public:
    static constexpr uint8_t NUM_SLOTS = 8;

    void reset() {
        for (uint8_t i = 0; i < NUM_SLOTS; i++) slots_[i] = TileSlot{};
    }

    void set_discovered(uint8_t slot, uint8_t address) {
        slots_[slot].discovered = true;
        slots_[slot].address    = address;
        slots_[slot].status     = TileStatus::OK;
    }

    void set_status(uint8_t slot, TileStatus status) {
        slots_[slot].status = status;
    }

    void increment_retry(uint8_t slot) {
        slots_[slot].retry_count++;
    }

    void set_version(uint8_t slot, const FirmwareVersion &version) {
        slots_[slot].version       = version;
        slots_[slot].version_valid = true;
    }

    bool is_discovered(uint8_t slot) const { return slots_[slot].discovered; }
    uint8_t address_for(uint8_t slot) const { return slots_[slot].address; }
    uint8_t retry_count(uint8_t slot) const { return slots_[slot].retry_count; }
    TileStatus status_for(uint8_t slot) const { return slots_[slot].status; }
    bool has_version(uint8_t slot) const { return slots_[slot].version_valid; }
    const FirmwareVersion &version_for(uint8_t slot) const { return slots_[slot].version; }

    uint8_t discovered_count() const {
        uint8_t count = 0;
        for (uint8_t i = 0; i < NUM_SLOTS; i++)
            if (slots_[i].discovered) count++;
        return count;
    }

private:
    TileSlot slots_[NUM_SLOTS] = {};
};
