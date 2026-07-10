#pragma once
#include "row_bus_protocol.h"
#include "protocol.h"   // src/common/tile_bus_protocol/, via lib_extra_dirs
#include "transport.h"  // src/common/tile_bus_protocol/, via lib_extra_dirs
#include "sense_mapper.h"
#include "../../include/power_monitor.h"

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

private:
    void handle_test();
    void handle_status();
    void handle_power();
    void handle_re_discover();
    void handle_error_log();
    void handle_send_data(const RowBusFrame &in);
    void handle_latch();
    void handle_blackout();

    void send_tile_frame(uint8_t addr, Cmd cmd, const uint8_t *payload, uint8_t len);
    void broadcast_tile_latch();

    ITransport    &tile_transport_;
    SenseMapper   &sense_;
    IPowerMonitor &power_;
    uint8_t        my_row_addr_;
    RowBusFrame    response_ = {};
};
