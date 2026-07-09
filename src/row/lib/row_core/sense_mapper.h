#pragma once
#include <stdint.h>
#include "protocol.h"   // src/common/tile_bus_protocol/, via lib_extra_dirs
#include "transport.h"  // src/common/tile_bus_protocol/, via lib_extra_dirs
#include "../../include/row_sense_control.h"
#include "tile_map.h"

enum class SenseMapState : uint8_t { IDLE, DISCOVERING, DONE, ERROR };

// Drives the Tile Bus SENSE auto-mapping sequence (docs/communication.md,
// docs/tile-bus-protocol.md §9) to discover which physical tile address sits
// at each of the 8 logical slots, populating a caller-owned TileMap.
class SenseMapper {
public:
    SenseMapper(ITransport &transport, IRowSenseControl &sense, TileMap &map);

    void start();                 // begin/restart discovery — calls map_.reset()
    void poll(uint32_t now_ms);   // drive the state machine; call every loop iteration
    SenseMapState state() const { return state_; }
    const TileMap &result() const { return map_; }

private:
    enum class Step : uint8_t { START, WAIT_DETECT_RESP, WAIT_ACTIVATE_ACK };

    // 3 total attempts per docs/tile-bus-protocol.md §7 (1 initial + 2 retries).
    static constexpr uint8_t  MAX_RETRIES = 2;
    // 5ms placeholder per docs/tile-bus-protocol.md §7's "Response timeout
    // value: 5ms is a placeholder" note.
    static constexpr uint32_t TIMEOUT_MS  = 5;

    void send_detect_sense();
    void send_activate_sense(uint8_t addr);
    void send_clear_sense(uint8_t addr);
    void broadcast_clear_sense();
    void advance_to_next_slot(uint32_t now_ms);
    void finish_discovery();
    void fail_discovery();

    ITransport       &transport_;
    IRowSenseControl &sense_;
    TileMap          &map_;
    FrameParser      parser_;
    SenseMapState    state_         = SenseMapState::IDLE;
    Step             step_          = Step::START;
    uint8_t          current_slot_  = 0;
    uint32_t         request_sent_ms_ = 0;
};
