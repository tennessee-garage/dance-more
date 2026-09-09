#pragma once
#include "row_bus_protocol.h"
#include "protocol.h"   // src/common/tile_bus_protocol/, via lib_extra_dirs
#include "transport.h"  // src/common/tile_bus_protocol/, via lib_extra_dirs
#include "sense_mapper.h"
#include "../../include/power_monitor.h"

// One entry in the error log ring buffer (docs/row-bus-protocol.md §5.1's
// ERROR_LOG_RESP payload format). For LATCH_OVERRUN entries the fields are
// repurposed per that doc: slot = number of tile slots forwarded when LATCH
// arrived, tile_bus_cmd = the tile command that was in flight at the time.
struct ErrorLogEntry {
    uint8_t  slot;
    uint8_t  tile_bus_cmd;
    uint8_t  error_type;
    uint16_t timestamp_s;
};

// error_type values (docs/row-bus-protocol.md §5.1). LATCH_OVERRUN and
// ROW_BUS_RX_OVERFLOW have producers; the others describe future error
// sources (SenseMapper retry exhaustion, CRC failures, sense collisions)
// with nothing wired up to log them yet.
static constexpr uint8_t ERROR_TYPE_LATCH_OVERRUN = 0x04;
// Row Bus receive overrun: the Pi-facing UART dropped bytes, so whatever
// frame was in flight died on CRC. Unlike the types above this describes a
// fault on the *upstream* link, so slot and tile_bus_cmd carry no meaning
// and are logged as 0.
static constexpr uint8_t ERROR_TYPE_ROW_BUS_RX_OVERFLOW = 0x05;
// Diagnostic, not a fault: one entry per boot carrying the RP2350's sticky
// reset-cause bits, so a restart says *why* it happened rather than just
// that it did. slot/tile_bus_cmd are the high/low bytes of
// POWMAN_CHIP_RESET >> 16, which is where every HAD_* cause bit lives
// (POR, BOR, RUN_LOW, the four watchdog flavours, GLITCH_DETECT, ...).
//
// Added to explain a restart ~3.6 s into every power-up that neither bus
// showed the cause of, and it earned its keep immediately: POR with no
// watchdog, BOR or glitch bit set is what pointed at the supply rather than
// the firmware, and the row was in fact browning out. A plain
// watchdog_caused_reboot() bool could not have told those apart.
static constexpr uint8_t ERROR_TYPE_ROW_BOOT = 0x06;
// 0x07 is retired. It used to flag "discovery assigned a tile to slot 1 or
// higher", which was a usable proxy for the phantom-8 cascade only while the
// bench had a single tile. Since addresses are assigned by position
// (SET_ADDRESS), slot 1 holding 0x02 is simply correct, so it fired on every
// sweep of a healthy two-tile row - twice a second through the boot retry
// window, filling the ring in 16 seconds and evicting ROW_BOOT, the one entry
// worth keeping. A diagnostic that destroys the evidence it was added to
// preserve is worse than none. Deliberately not reused: an older row's 0x07
// entries decode as unknown on a newer host rather than being mislabelled as
// a fault that now means something else.
// One entry each time a discovery sweep begins, slot = a wrapping counter.
// The error log does not survive a chip reset (it is a plain global, zeroed
// by static init), so these say whether a restart was a reset at all: sweeps
// logged either side of a gap mean the chip kept running and only the sweep
// restarted, while a log that begins again after the gap means it rebooted.
// Reading the counter alongside a bus capture is also what pins the row's
// millis() epoch to a point on the wire.
static constexpr uint8_t ERROR_TYPE_SENSE_START = 0x08;
// A tile that discovery mapped, and that then would not answer VERSION.
// slot = the slot, tile_bus_cmd = the address it was assigned.
//
// This is what 0x07 was reaching for. Because SET_ADDRESS makes a tile
// abandon its previous address, a row that mistakenly walks one tile into
// several slots leaves every earlier slot pointing at an address nothing
// answers to - so the phantom cascade shows up here as "every slot but the
// last went silent", a signature no correct row can produce. It also catches
// the plainer cases the row was previously blind to: a tile that took an
// address and then died, or one unplugged mid-run. The sweep already
// detected all of this and simply moved on without saying so.
static constexpr uint8_t ERROR_TYPE_TILE_NO_VERSION = 0x09;

// Dispatches Row Bus commands arriving from the Raspberry Pi
// (docs/row-bus-protocol.md §5). in.addr must already be filtered by the
// caller (== my_row_addr or broadcast).
class RowCommandHandler {
public:
    RowCommandHandler(ITransport &tile_transport, SenseMapper &sense,
                       IPowerMonitor &power, uint8_t my_row_addr);

    // Returns a pointer to a statically-allocated response frame, or nullptr
    // if no response is needed (SEND_DATA / LATCH / BLACKOUT).
    const RowBusFrame *handle(const RowBusFrame &in);

    // Advances any in-flight SEND_DATA forwarding by one tile slot. Call
    // every loop iteration (mirrors SenseMapper::poll()'s convention) -
    // this is what makes a LATCH arriving mid-forward observable/deferrable
    // rather than everything completing atomically within handle().
    void poll(uint32_t now_ms);

    // Records a Row Bus receive overrun in the error log. Detected on core 0
    // (which owns the Pi-facing UART) but logged here, because core 1 owns
    // the error log - main.cpp carries the flag across.
    void log_row_bus_overflow(uint32_t now_ms);

    // Diagnostics, logged by main.cpp: one per boot, one per discovery
    // sweep, and one per discovered tile that then would not answer VERSION.
    // All go in the error log because it is the only channel that already
    // reaches the Pi with a timestamp attached.
    void log_boot(uint32_t chip_reset_reason, uint32_t now_ms);
    void log_tile_no_version(uint8_t slot, uint8_t addr, uint32_t now_ms);
    void log_sense_start(uint32_t now_ms);

private:
    // The boot entry is held outside the ring, so a busy row cannot evict the
    // one entry that says why it restarted - which is exactly what happened
    // when two per-sweep diagnostics filled 32 slots in 16 seconds. The ring
    // is one shorter to keep the response at the documented 161-byte
    // maximum: 1 + 32 x 5, boot included.
    static constexpr uint8_t ERROR_LOG_CAPACITY = 31;

    void handle_test();
    void handle_status();
    void handle_power();
    void handle_re_discover();
    void handle_error_log();
    void handle_version();
    void handle_send_data(const RowBusFrame &in);
    void handle_latch();
    void handle_blackout();

    void send_tile_frame(uint8_t addr, Cmd cmd, const uint8_t *payload, uint8_t len);
    void broadcast_tile_latch();

    // Per-tile SEND_DATA entry parsing, shared by handle_send_data's
    // forwarding loop and handle_latch's overrun peek. Returns false if
    // offset is out of range or the tile_cmd is unrecognized.
    bool parse_entry(uint16_t offset, uint8_t *tile_cmd_out, uint8_t *data_len_out) const;
    void advance_forwarding();  // forwards one slot; called from poll()
    void finish_forwarding();   // fires a deferred LATCH + logs the overrun, if any

    void log_error(uint8_t slot, uint8_t tile_bus_cmd, uint8_t error_type, uint32_t now_ms);

    ITransport    &tile_transport_;
    SenseMapper   &sense_;
    IPowerMonitor &power_;
    uint8_t        my_row_addr_;
    RowBusFrame    response_ = {};

    // SEND_DATA forwarding state (#46).
    bool        forwarding_       = false;
    RowBusFrame forwarding_frame_ = {};
    uint16_t    forward_offset_   = 0;
    uint8_t     forward_slot_     = 0;
    bool        latch_deferred_   = false;
    uint8_t     overrun_slot_     = 0;
    uint8_t     overrun_tile_cmd_ = 0;
    uint32_t    last_now_ms_      = 0;

    // Error log ring buffer (#46). Oldest entries are overwritten when full;
    // reading (ERROR_LOG) does not clear it, per docs/row-bus-protocol.md §5.1.
    ErrorLogEntry error_log_[ERROR_LOG_CAPACITY]{};
    uint8_t       error_log_count_ = 0; // valid entries, caps at ERROR_LOG_CAPACITY
    uint8_t       error_log_next_  = 0; // next write index, wraps
    ErrorLogEntry boot_entry_      = {}; // outside the ring; see ERROR_LOG_CAPACITY
    bool          has_boot_entry_  = false;
};
