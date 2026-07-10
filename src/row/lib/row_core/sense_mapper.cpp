#include "sense_mapper.h"

SenseMapper::SenseMapper(ITransport &transport, IRowSenseControl &sense, TileMap &map)
    : transport_(transport), sense_(sense), map_(map) {}

void SenseMapper::send_detect_sense() {
    Frame f{};
    f.addr = ADDR_BROADCAST;
    f.cmd  = (uint8_t)Cmd::DETECT_SENSE;
    f.len  = 0;
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

void SenseMapper::start() {
    map_.reset();
    current_slot_ = 0;
    state_ = SenseMapState::DISCOVERING;
    step_  = Step::START;
}

void SenseMapper::finish_discovery() {
    // The last discovered tile's own outgoing SENSE is still asserted
    // (activated to reach the now-missing next slot) - release everything.
    broadcast_clear_sense();
    state_ = SenseMapState::DONE;
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
        finish_discovery();
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
                map_.set_discovered(current_slot_, f.addr);
                send_activate_sense(f.addr);
                request_sent_ms_ = now_ms;
                step_ = Step::WAIT_ACTIVATE_ACK;
            }
            // else: unrelated frame, ignore and keep waiting.
        } else if (now_ms - request_sent_ms_ >= TIMEOUT_MS) {
            map_.increment_retry(current_slot_);
            if (map_.retry_count(current_slot_) > MAX_RETRIES) {
                // No tile answered after the full retry budget - this is the
                // expected end-of-chain signal, not a failure.
                finish_discovery();
            } else {
                send_detect_sense();
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
    }
}
