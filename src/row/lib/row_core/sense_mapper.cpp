#include "sense_mapper.h"
#include "firmware_version.h" // src/common/tile_bus_protocol/, via lib_extra_dirs

SenseMapper::SenseMapper(ITransport &transport, IRowSenseControl &sense, TileMap &map)
    : transport_(transport), sense_(sense), map_(map) {}

void SenseMapper::send_detect_sense() {
    Frame f{};
    f.addr = ADDR_BROADCAST;
    f.cmd  = (uint8_t)Cmd::DETECT_SENSE;
    f.len  = 0;
    transport_.send(f);
}

void SenseMapper::send_set_address(uint8_t addr) {
    // Broadcast on purpose: the tile being addressed may not have an address
    // yet, so there is nothing to unicast to. The SENSE walk is what makes
    // this unambiguous - exactly one tile has its incoming line asserted, and
    // only that tile acts on this.
    Frame f{};
    f.addr       = ADDR_BROADCAST;
    f.cmd        = (uint8_t)Cmd::SET_ADDRESS;
    f.len        = 1;
    f.payload[0] = addr;
    transport_.send(f);
}

void SenseMapper::send_activate_sense(uint8_t addr) {
    Frame f{};
    f.addr = addr;
    f.cmd  = (uint8_t)Cmd::ACTIVATE_SENSE;
    f.len  = 0;
    transport_.send(f);
}

void SenseMapper::send_clear_sense(uint8_t addr) {
    Frame f{};
    f.addr = addr;
    f.cmd  = (uint8_t)Cmd::CLEAR_SENSE;
    f.len  = 0;
    transport_.send(f);
}

void SenseMapper::broadcast_clear_sense() {
    Frame f{};
    f.addr = ADDR_BROADCAST;
    f.cmd  = (uint8_t)Cmd::CLEAR_SENSE;
    f.len  = 0;
    transport_.send(f);
}

void SenseMapper::send_version_query(uint8_t addr) {
    Frame f{};
    f.addr = addr;
    f.cmd  = (uint8_t)Cmd::VERSION;
    f.len  = 0;
    transport_.send(f);
}

void SenseMapper::start() {
    sweep_started_pending_ = true;
    map_.reset();
    current_slot_ = 0;
    state_ = SenseMapState::DISCOVERING;
    step_  = Step::START;
}

void SenseMapper::finish_discovery(uint32_t now_ms) {
    // The last discovered tile's own outgoing SENSE is still asserted
    // (activated to reach the now-missing next slot) - release everything.
    broadcast_clear_sense();

    // Sweep VERSION across every discovered slot before declaring DONE, so
    // the cache RowCommandHandler serves (docs/row-bus-protocol.md's
    // VERSION) is populated by the time the Pi can see state == running.
    version_slot_ = 0;
    advance_version_query(now_ms);
}

void SenseMapper::advance_version_query(uint32_t now_ms) {
    while (version_slot_ < TileMap::NUM_SLOTS && !map_.is_discovered(version_slot_))
        version_slot_++;

    if (version_slot_ >= TileMap::NUM_SLOTS) {
        state_ = SenseMapState::DONE;
        return;
    }

    version_retry_count_ = 0;
    send_version_query(map_.address_for(version_slot_));
    request_sent_ms_ = now_ms;
    step_ = Step::WAIT_VERSION_RESP;
}

bool SenseMapper::take_sweep_started() {
    if (!sweep_started_pending_) return false;
    sweep_started_pending_ = false;
    return true;
}

bool SenseMapper::take_silent_tile(uint8_t *slot_out, uint8_t *addr_out) {
    if (!silent_tile_pending_) return false;
    silent_tile_pending_ = false;
    *slot_out = silent_tile_slot_;
    *addr_out = silent_tile_addr_;
    return true;
}

void SenseMapper::fail_discovery() {
    broadcast_clear_sense();
    state_ = SenseMapState::ERROR;
}

void SenseMapper::advance_to_next_slot(uint32_t now_ms) {
    // Release whatever is currently driving this slot's SENSE-in so it
    // stops answering DETECT_SENSE once we move on to the next slot.
    if (current_slot_ == 0) {
        sense_.release_out();
    } else {
        send_clear_sense(map_.address_for(current_slot_ - 1));
    }

    current_slot_++;
    if (current_slot_ >= TileMap::NUM_SLOTS) {
        finish_discovery(now_ms);
        return;
    }

    step_ = Step::SETTLE;
    request_sent_ms_ = now_ms;
}

void SenseMapper::poll(uint32_t now_ms) {
    if (state_ != SenseMapState::DISCOVERING) return;

    switch (step_) {

    case Step::START:
        sense_.assert_out();
        step_ = Step::SETTLE;
        request_sent_ms_ = now_ms;
        break;

    case Step::SETTLE:
        if (now_ms - request_sent_ms_ >= SETTLE_MS) {
            send_detect_sense();
            request_sent_ms_ = now_ms;
            step_ = Step::WAIT_DETECT_RESP;
        }
        break;

    case Step::WAIT_DETECT_RESP: {
        Frame f;
        if (transport_.poll(parser_, &f)) {
            if (f.cmd == (uint8_t)Cmd::DETECT_RESP) {
                // Deliberately ignore the address this frame carries. It
                // reports whatever the tile currently holds, which for a
                // freshly booted tile is ADDR_UNASSIGNED and for a tile that
                // survived a row reset is a stale assignment. DETECT_RESP
                // means "a tile is here"; what it is called is decided next.
                send_set_address(address_for_slot(current_slot_));
                request_sent_ms_ = now_ms;
                step_ = Step::WAIT_SET_ADDRESS_ACK;
            }
            // else: unrelated frame, ignore and keep waiting.
        } else if (now_ms - request_sent_ms_ >= TIMEOUT_MS) {
            map_.increment_retry(current_slot_);
            if (map_.retry_count(current_slot_) > MAX_RETRIES) {
                // No tile answered after the full retry budget - this is the
                // expected end-of-chain signal, not a failure.
                finish_discovery(now_ms);
            } else {
                send_detect_sense();
                request_sent_ms_ = now_ms;
            }
        }
        break;
    }

    case Step::WAIT_SET_ADDRESS_ACK: {
        Frame f;
        const uint8_t assigned = address_for_slot(current_slot_);
        static constexpr uint8_t SET_ADDRESS_ACK =
            (uint8_t)Cmd::ACK | (uint8_t)Cmd::SET_ADDRESS;
        if (transport_.poll(parser_, &f)) {
            // The tile answers from its new address, so matching on it is
            // what confirms the assignment landed rather than merely that
            // something replied.
            if (f.cmd == SET_ADDRESS_ACK && f.addr == assigned) {
                map_.set_discovered(current_slot_, assigned);
                send_activate_sense(assigned);
                request_sent_ms_ = now_ms;
                step_ = Step::WAIT_ACTIVATE_ACK;
            }
            // else: unrelated frame, ignore and keep waiting.
        } else if (now_ms - request_sent_ms_ >= TIMEOUT_MS) {
            map_.increment_retry(current_slot_);
            if (map_.retry_count(current_slot_) > MAX_RETRIES) {
                // A tile answered DETECT_SENSE and then would not take an
                // address, so it is present but unusable - a real fault, not
                // the end of the chain.
                fail_discovery();
            } else {
                send_set_address(assigned);
                request_sent_ms_ = now_ms;
            }
        }
        break;
    }

    case Step::WAIT_ACTIVATE_ACK: {
        Frame f;
        static constexpr uint8_t ACTIVATE_ACK = (uint8_t)Cmd::ACK | (uint8_t)Cmd::ACTIVATE_SENSE;
        if (transport_.poll(parser_, &f)) {
            if (f.cmd == ACTIVATE_ACK && f.addr == map_.address_for(current_slot_)) {
                advance_to_next_slot(now_ms);
            }
            // else: unrelated frame, ignore and keep waiting.
        } else if (now_ms - request_sent_ms_ >= TIMEOUT_MS) {
            map_.increment_retry(current_slot_);
            if (map_.retry_count(current_slot_) > MAX_RETRIES) {
                // A tile we already discovered failed to ACK a direct
                // command - that's a real fault, not end-of-chain.
                fail_discovery();
            } else {
                send_activate_sense(map_.address_for(current_slot_));
                request_sent_ms_ = now_ms;
            }
        }
        break;
    }

    case Step::WAIT_VERSION_RESP: {
        Frame f;
        if (transport_.poll(parser_, &f)) {
            if (f.cmd == (uint8_t)Cmd::VERSION_RESP && f.addr == map_.address_for(version_slot_)) {
                FirmwareVersion v{};
                if (f.len == FW_VERSION_WIRE_SIZE && fw_version_decode(f.payload, &v))
                    map_.set_version(version_slot_, v);
                // else: malformed response - leave this slot's cache entry
                // invalid, same as a tile that never answered.
                version_slot_++;
                advance_version_query(now_ms);
            }
            // else: unrelated frame, ignore and keep waiting.
        } else if (now_ms - request_sent_ms_ >= TIMEOUT_MS) {
            version_retry_count_++;
            if (version_retry_count_ > MAX_RETRIES) {
                // Discovered but not answering VERSION - leave its cache
                // entry invalid and move on. Not a discovery fault: the
                // tile passed SENSE mapping, so its TileStatus stays OK.
                //
                // Reported all the same. A tile that took an address and
                // then went quiet is exactly the condition nothing else
                // here surfaces, and it is also how a mis-walked row shows
                // itself: SET_ADDRESS makes a tile drop its old address, so
                // one tile walked into several slots leaves every slot but
                // the last answering to nothing.
                silent_tile_pending_ = true;
                silent_tile_slot_    = version_slot_;
                silent_tile_addr_    = map_.address_for(version_slot_);
                version_slot_++;
                advance_version_query(now_ms);
            } else {
                send_version_query(map_.address_for(version_slot_));
                request_sent_ms_ = now_ms;
            }
        }
        break;
    }
    }
}
