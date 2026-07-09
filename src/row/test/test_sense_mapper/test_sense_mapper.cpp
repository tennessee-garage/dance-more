#include <unity.h>
#include "sense_mapper.h"

void setUp() {}
void tearDown() {}

// ---------------------------------------------------------------------------
// Fakes
// ---------------------------------------------------------------------------

// Models a chain of `tile_count` tiles addressed 1..tile_count, one slot at a
// time becoming "active" (visible to DETECT_SENSE) as each prior tile is
// successfully ACTIVATE_SENSE'd, mirroring the real SENSE hardware chain
// closely enough to drive SenseMapper's state machine and timing.
class FakeChainTransport : public ITransport {
public:
    explicit FakeChainTransport(uint8_t tile_count) : tile_count_(tile_count) {}

    void init() override {}

    void send(const Frame &frame) override {
        Cmd cmd = (Cmd)frame.cmd;

        if (cmd == Cmd::DETECT_SENSE) {
            if (active_slot_ >= tile_count_) {
                pending_ = false; // no tile at this slot - real hardware silence
                return;
            }
            if (fail_detect_slot_ == active_slot_ && !detect_failed_once_) {
                detect_failed_once_ = true;
                pending_ = false; // simulate one dropped response
                return;
            }
            pending_frame_ = Frame{};
            pending_frame_.addr = tile_address(active_slot_);
            pending_frame_.cmd  = (uint8_t)Cmd::DETECT_RESP;
            pending_frame_.len  = 0;
            pending_ = true;

        } else if (cmd == Cmd::ACTIVATE_SENSE) {
            if (never_ack_activate_ || frame.addr != tile_address(active_slot_)) {
                pending_ = false;
                return;
            }
            pending_frame_ = Frame{};
            pending_frame_.addr       = frame.addr;
            pending_frame_.cmd        = (uint8_t)Cmd::ACK | (uint8_t)Cmd::ACTIVATE_SENSE;
            pending_frame_.len        = 1;
            pending_frame_.payload[0] = 0x00;
            pending_ = true;
            active_slot_++; // this tile now asserts its own SENSE-out

        } else {
            pending_ = false; // CLEAR_SENSE etc: no response expected
        }
    }

    bool poll(FrameParser &, Frame *out) override {
        if (!pending_) return false;
        *out = pending_frame_;
        pending_ = false;
        return true;
    }

    void fail_first_detect_for_slot(uint8_t slot) { fail_detect_slot_ = slot; }
    void never_ack_activate() { never_ack_activate_ = true; }

private:
    static uint8_t tile_address(uint8_t slot) { return (uint8_t)(slot + 1); }

    uint8_t tile_count_;
    uint8_t active_slot_ = 0;
    bool    pending_ = false;
    Frame   pending_frame_{};

    int     fail_detect_slot_ = -1;
    bool    detect_failed_once_ = false;
    bool    never_ack_activate_ = false;
};

class FakeRowSense : public IRowSenseControl {
public:
    void assert_out() override { asserted = true; assert_count++; }
    void release_out() override { asserted = false; release_count++; }
    bool asserted = false;
    int  assert_count = 0;
    int  release_count = 0;
};

// Drives poll() with an advancing clock until DONE/ERROR or a safety cap.
static void run_to_completion(SenseMapper &mapper, uint32_t &now) {
    for (int i = 0; i < 2000 && mapper.state() == SenseMapState::DISCOVERING; i++) {
        mapper.poll(now);
        now += 1;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void test_full_discovery_8_tiles() {
    FakeChainTransport transport(8);
    FakeRowSense sense;
    TileMap map;
    SenseMapper mapper(transport, sense, map);

    mapper.start();
    uint32_t now = 0;
    run_to_completion(mapper, now);

    TEST_ASSERT_EQUAL(SenseMapState::DONE, mapper.state());
    TEST_ASSERT_EQUAL(8, map.discovered_count());
    for (uint8_t i = 0; i < 8; i++) {
        TEST_ASSERT_TRUE(map.is_discovered(i));
        TEST_ASSERT_EQUAL_HEX8(i + 1, map.address_for(i));
    }
    TEST_ASSERT_TRUE(sense.assert_count >= 1);
    TEST_ASSERT_TRUE(sense.release_count >= 1);
}

void test_short_chain_ends_via_timeout_not_error() {
    FakeChainTransport transport(3);
    FakeRowSense sense;
    TileMap map;
    SenseMapper mapper(transport, sense, map);

    mapper.start();
    uint32_t now = 0;
    run_to_completion(mapper, now);

    TEST_ASSERT_EQUAL(SenseMapState::DONE, mapper.state());
    TEST_ASSERT_EQUAL(3, map.discovered_count());
    TEST_ASSERT_EQUAL_HEX8(1, map.address_for(0));
    TEST_ASSERT_EQUAL_HEX8(2, map.address_for(1));
    TEST_ASSERT_EQUAL_HEX8(3, map.address_for(2));
}

void test_activate_sense_never_acked_reaches_error() {
    FakeChainTransport transport(8);
    transport.never_ack_activate();
    FakeRowSense sense;
    TileMap map;
    SenseMapper mapper(transport, sense, map);

    mapper.start();
    uint32_t now = 0;
    run_to_completion(mapper, now);

    TEST_ASSERT_EQUAL(SenseMapState::ERROR, mapper.state());
}

void test_retry_is_tracked_per_slot() {
    FakeChainTransport transport(8);
    transport.fail_first_detect_for_slot(3);
    FakeRowSense sense;
    TileMap map;
    SenseMapper mapper(transport, sense, map);

    mapper.start();
    uint32_t now = 0;
    run_to_completion(mapper, now);

    TEST_ASSERT_EQUAL(SenseMapState::DONE, mapper.state());
    TEST_ASSERT_EQUAL(8, map.discovered_count());
    TEST_ASSERT_EQUAL(1, map.retry_count(3));
    for (uint8_t i = 0; i < 8; i++) {
        if (i == 3) continue;
        TEST_ASSERT_EQUAL(0, map.retry_count(i));
    }
}

int main(int, char **) {
    UNITY_BEGIN();

    RUN_TEST(test_full_discovery_8_tiles);
    RUN_TEST(test_short_chain_ends_via_timeout_not_error);
    RUN_TEST(test_activate_sense_never_acked_reaches_error);
    RUN_TEST(test_retry_is_tracked_per_slot);

    return UNITY_END();
}
