#include <unity.h>
#include <string.h>
#include "pattern.h"

static PatternEngine  pat;
static PixelBuffer    buf;

// SET_PATTERN payload: id + speed + depth + spread + seed.
static void arm(PatternEngine &p, uint8_t id, uint8_t speed, uint8_t depth,
                uint8_t spread, uint8_t seed) {
    const uint8_t payload[5] = {id, speed, depth, spread, seed};
    TEST_ASSERT_TRUE(p.arm(payload, sizeof(payload)));
}

static void fill(PixelBuffer &b, uint8_t r, uint8_t g, uint8_t bl) {
    for (uint8_t i = 0; i < PixelBuffer::NUM_LEDS; i++) b.leds[i] = {r, g, bl};
}

void setUp() {
    pat = PatternEngine{};
    buf = PixelBuffer{};
}
void tearDown() {}

// ---------------------------------------------------------------------------
// arm() validation
// ---------------------------------------------------------------------------

void test_arm_rejects_short_payload() {
    const uint8_t payload[4] = {PatternEngine::SHIMMER, 60, 128, 255};
    TEST_ASSERT_FALSE(pat.arm(payload, sizeof(payload)));
    TEST_ASSERT_FALSE(pat.armed());
}

void test_arm_rejects_reserved_bits() {
    // Bits 7:5 must be zero (docs/tile-bus-protocol.md §5.2).
    const uint8_t payload[5] = {0x20 | PatternEngine::SHIMMER, 60, 128, 255, 0};
    TEST_ASSERT_FALSE(pat.arm(payload, sizeof(payload)));
    TEST_ASSERT_FALSE(pat.armed());
}

void test_arm_rejects_unimplemented_id() {
    const uint8_t payload[5] = {7, 60, 128, 255, 0};   // reserved in the library
    TEST_ASSERT_FALSE(pat.arm(payload, sizeof(payload)));
    TEST_ASSERT_FALSE(pat.armed());
}

void test_rejected_arm_leaves_running_pattern_alone() {
    fill(buf, 200, 100, 50);
    arm(pat, PatternEngine::SHIMMER, 60, 255, 255, 1);
    pat.on_latch(buf, 0);
    TEST_ASSERT_TRUE(pat.active());

    const uint8_t bad[5] = {7, 0, 0, 0, 0};
    TEST_ASSERT_FALSE(pat.arm(bad, sizeof(bad)));
    TEST_ASSERT_TRUE(pat.active());
}

// ---------------------------------------------------------------------------
// Arming is staged; LATCH starts it
// ---------------------------------------------------------------------------

void test_arm_does_not_start_until_latch() {
    fill(buf, 255, 255, 255);
    arm(pat, PatternEngine::SHIMMER, 60, 255, 255, 1);
    TEST_ASSERT_TRUE(pat.armed());
    TEST_ASSERT_FALSE(pat.active());

    // Nothing renders before the LATCH, however long we wait.
    TEST_ASSERT_FALSE(pat.poll(buf, 10000));

    pat.on_latch(buf, 0);
    TEST_ASSERT_FALSE(pat.armed());
    TEST_ASSERT_TRUE(pat.active());
}

void test_latch_with_nothing_armed_does_not_restart() {
    fill(buf, 255, 0, 0);
    arm(pat, PatternEngine::SHIMMER, 60, 255, 255, 1);
    pat.on_latch(buf, 0);

    // The host's ordinary per-frame LATCH must not re-snapshot the base from
    // an already-modulated buffer, or the pattern would ratchet to black.
    TEST_ASSERT_TRUE(pat.poll(buf, PatternEngine::FRAME_MS));
    PixelBuffer after_render = buf;
    pat.on_latch(buf, PatternEngine::FRAME_MS);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(&after_render.leds[0].r, &buf.leds[0].r,
                                  PixelBuffer::NUM_LEDS * 3);
}

// ---------------------------------------------------------------------------
// OFF / SOLID - static, painted on the LATCH push
// ---------------------------------------------------------------------------

void test_off_blacks_the_buffer_and_stops() {
    fill(buf, 255, 255, 255);
    arm(pat, PatternEngine::OFF, 0, 0, 0, 0);
    pat.on_latch(buf, 0);

    TEST_ASSERT_FALSE(pat.active());
    for (uint8_t i = 0; i < PixelBuffer::NUM_LEDS; i++) {
        TEST_ASSERT_EQUAL_UINT8(0, buf.leds[i].r);
        TEST_ASSERT_EQUAL_UINT8(0, buf.leds[i].g);
        TEST_ASSERT_EQUAL_UINT8(0, buf.leds[i].b);
    }
    TEST_ASSERT_FALSE(pat.poll(buf, 10000));
}

void test_solid_fills_from_params_and_stops() {
    arm(pat, PatternEngine::SOLID, 0x11, 0x22, 0x33, 0);
    pat.on_latch(buf, 0);

    TEST_ASSERT_FALSE(pat.active());
    for (uint8_t i = 0; i < PixelBuffer::NUM_LEDS; i++) {
        TEST_ASSERT_EQUAL_UINT8(0x11, buf.leds[i].r);
        TEST_ASSERT_EQUAL_UINT8(0x22, buf.leds[i].g);
        TEST_ASSERT_EQUAL_UINT8(0x33, buf.leds[i].b);
    }
}

// ---------------------------------------------------------------------------
// SHIMMER rendering
// ---------------------------------------------------------------------------

void test_poll_respects_frame_interval() {
    fill(buf, 255, 255, 255);
    arm(pat, PatternEngine::SHIMMER, 60, 255, 255, 1);
    pat.on_latch(buf, 1000);

    TEST_ASSERT_FALSE(pat.poll(buf, 1000));
    TEST_ASSERT_FALSE(pat.poll(buf, 1000 + PatternEngine::FRAME_MS - 1));
    TEST_ASSERT_TRUE(pat.poll(buf, 1000 + PatternEngine::FRAME_MS));
}

void test_depth_zero_leaves_base_untouched() {
    fill(buf, 200, 100, 50);
    arm(pat, PatternEngine::SHIMMER, 60, 0, 255, 1);
    pat.on_latch(buf, 0);

    // 255/256 of the base: the multiply-and-shift, not the modulation.
    uint32_t t = 0;
    for (int frame = 0; frame < 200; frame++) {
        t += PatternEngine::FRAME_MS;
        TEST_ASSERT_TRUE(pat.poll(buf, t));
        for (uint8_t i = 0; i < PixelBuffer::NUM_LEDS; i++)
            TEST_ASSERT_EQUAL_UINT8(199, buf.leds[i].r);
    }
}

void test_pattern_only_ever_darkens_the_base() {
    fill(buf, 200, 100, 50);
    arm(pat, PatternEngine::SHIMMER, 60, 255, 255, 42);
    pat.on_latch(buf, 0);

    uint32_t t = 0;
    for (int frame = 0; frame < 300; frame++) {
        t += PatternEngine::FRAME_MS;
        pat.poll(buf, t);
        for (uint8_t i = 0; i < PixelBuffer::NUM_LEDS; i++) {
            TEST_ASSERT_LESS_OR_EQUAL_UINT8(200, buf.leds[i].r);
            TEST_ASSERT_LESS_OR_EQUAL_UINT8(100, buf.leds[i].g);
            TEST_ASSERT_LESS_OR_EQUAL_UINT8(50,  buf.leds[i].b);
        }
    }
}

void test_full_depth_reaches_near_black_and_returns_to_base() {
    fill(buf, 255, 255, 255);
    arm(pat, PatternEngine::SHIMMER, 60, 255, 0, 0);  // spread 0 = whole tile in unison
    pat.on_latch(buf, 0);

    uint8_t lo = 255, hi = 0;
    uint32_t t = 0;
    for (int frame = 0; frame < 200; frame++) {       // ~1 cycle at 60 cpm
        t += PatternEngine::FRAME_MS;
        pat.poll(buf, t);
        if (buf.leds[0].r < lo) lo = buf.leds[0].r;
        if (buf.leds[0].r > hi) hi = buf.leds[0].r;
    }
    TEST_ASSERT_LESS_THAN_UINT8(4, lo);
    TEST_ASSERT_GREATER_THAN_UINT8(250, hi);
}

void test_spread_zero_moves_all_leds_together() {
    fill(buf, 255, 255, 255);
    arm(pat, PatternEngine::SHIMMER, 60, 255, 0, 7);
    pat.on_latch(buf, 0);
    TEST_ASSERT_TRUE(pat.poll(buf, PatternEngine::FRAME_MS));

    for (uint8_t i = 1; i < PixelBuffer::NUM_LEDS; i++)
        TEST_ASSERT_EQUAL_UINT8(buf.leds[0].r, buf.leds[i].r);
}

void test_spread_max_decorrelates_leds() {
    fill(buf, 255, 255, 255);
    arm(pat, PatternEngine::SHIMMER, 60, 255, 255, 7);
    pat.on_latch(buf, 0);
    TEST_ASSERT_TRUE(pat.poll(buf, PatternEngine::FRAME_MS));

    bool differs = false;
    for (uint8_t i = 1; i < PixelBuffer::NUM_LEDS; i++)
        if (buf.leds[i].r != buf.leds[0].r) { differs = true; break; }
    TEST_ASSERT_TRUE(differs);
}

void test_speed_zero_freezes_the_pattern() {
    fill(buf, 255, 255, 255);
    arm(pat, PatternEngine::SHIMMER, 0, 255, 255, 3);
    pat.on_latch(buf, 0);

    TEST_ASSERT_TRUE(pat.poll(buf, PatternEngine::FRAME_MS));
    PixelBuffer first = buf;
    TEST_ASSERT_TRUE(pat.poll(buf, PatternEngine::FRAME_MS * 2));
    TEST_ASSERT_EQUAL_UINT8_ARRAY(&first.leds[0].r, &buf.leds[0].r,
                                  PixelBuffer::NUM_LEDS * 3);
}

// ---------------------------------------------------------------------------
// Seed reproducibility - the lever for correlating or de-correlating tiles
// ---------------------------------------------------------------------------

static void render_one(PatternEngine &p, PixelBuffer &b, uint8_t seed) {
    fill(b, 255, 255, 255);
    arm(p, PatternEngine::SHIMMER, 60, 255, 255, seed);
    p.on_latch(b, 0);
    p.poll(b, PatternEngine::FRAME_MS);
}

void test_same_seed_gives_identical_output() {
    PatternEngine a, c;
    PixelBuffer ba{}, bc{};
    render_one(a, ba, 99);
    render_one(c, bc, 99);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(&ba.leds[0].r, &bc.leds[0].r,
                                  PixelBuffer::NUM_LEDS * 3);
}

void test_different_seed_gives_different_output() {
    PatternEngine a, c;
    PixelBuffer ba{}, bc{};
    render_one(a, ba, 1);
    render_one(c, bc, 2);
    TEST_ASSERT_FALSE(memcmp(&ba.leds[0].r, &bc.leds[0].r,
                             PixelBuffer::NUM_LEDS * 3) == 0);
}

// ---------------------------------------------------------------------------
// cancel()
// ---------------------------------------------------------------------------

void test_cancel_stops_a_running_pattern() {
    fill(buf, 255, 255, 255);
    arm(pat, PatternEngine::SHIMMER, 60, 255, 255, 1);
    pat.on_latch(buf, 0);
    TEST_ASSERT_TRUE(pat.active());

    pat.cancel();
    TEST_ASSERT_FALSE(pat.active());
    TEST_ASSERT_FALSE(pat.poll(buf, 10000));
}

void test_cancel_discards_an_armed_pattern() {
    arm(pat, PatternEngine::SHIMMER, 60, 255, 255, 1);
    pat.cancel();
    TEST_ASSERT_FALSE(pat.armed());

    pat.on_latch(buf, 0);
    TEST_ASSERT_FALSE(pat.active());
}

int main(int, char **) {
    UNITY_BEGIN();

    RUN_TEST(test_arm_rejects_short_payload);
    RUN_TEST(test_arm_rejects_reserved_bits);
    RUN_TEST(test_arm_rejects_unimplemented_id);
    RUN_TEST(test_rejected_arm_leaves_running_pattern_alone);

    RUN_TEST(test_arm_does_not_start_until_latch);
    RUN_TEST(test_latch_with_nothing_armed_does_not_restart);

    RUN_TEST(test_off_blacks_the_buffer_and_stops);
    RUN_TEST(test_solid_fills_from_params_and_stops);

    RUN_TEST(test_poll_respects_frame_interval);
    RUN_TEST(test_depth_zero_leaves_base_untouched);
    RUN_TEST(test_pattern_only_ever_darkens_the_base);
    RUN_TEST(test_full_depth_reaches_near_black_and_returns_to_base);
    RUN_TEST(test_spread_zero_moves_all_leds_together);
    RUN_TEST(test_spread_max_decorrelates_leds);
    RUN_TEST(test_speed_zero_freezes_the_pattern);

    RUN_TEST(test_same_seed_gives_identical_output);
    RUN_TEST(test_different_seed_gives_different_output);

    RUN_TEST(test_cancel_stops_a_running_pattern);
    RUN_TEST(test_cancel_discards_an_armed_pattern);

    return UNITY_END();
}
