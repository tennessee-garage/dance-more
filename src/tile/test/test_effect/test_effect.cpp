#include <unity.h>
#include <string.h>
#include "effect.h"

static EffectEngine fx;
static PixelBuffer  buf;

static constexpr uint8_t N = PixelBuffer::NUM_LEDS;

static bool stage(EffectEngine &e, uint8_t id, uint8_t p0 = 0, uint8_t p1 = 0,
                  uint8_t p2 = 0, uint8_t p3 = 0) {
    const uint8_t payload[5] = {id, p0, p1, p2, p3};
    return e.stage(payload, sizeof(payload));
}

static void fill(PixelBuffer &b, uint8_t r, uint8_t g, uint8_t bl) {
    for (uint8_t i = 0; i < N; i++) b.leds[i] = {r, g, bl};
}

static const Pixel &out(uint8_t i) { return fx.output().leds[i]; }

static void assert_out_equals_buffer() {
    TEST_ASSERT_EQUAL_UINT8_ARRAY(&buf.leds[0].r, &fx.output().leds[0].r, N * 3);
}

static void assert_pixel(uint8_t r, uint8_t g, uint8_t b, const Pixel &p) {
    TEST_ASSERT_EQUAL_UINT8(r, p.r);
    TEST_ASSERT_EQUAL_UINT8(g, p.g);
    TEST_ASSERT_EQUAL_UINT8(b, p.b);
}

void setUp() {
    fx  = EffectEngine{};
    buf = PixelBuffer{};
}
void tearDown() {}

// ---------------------------------------------------------------------------
// stage() validation
// ---------------------------------------------------------------------------

void test_stage_rejects_short_payload() {
    const uint8_t payload[4] = {EffectEngine::SHIMMER, 60, 128, 255};
    TEST_ASSERT_FALSE(fx.stage(payload, sizeof(payload)));
    TEST_ASSERT_FALSE(fx.staged());
}

void test_stage_rejects_reserved_bits() {
    // Bits 7:5 must be zero (docs/tile-bus-protocol.md §5.2).
    TEST_ASSERT_FALSE(stage(fx, 0x20 | EffectEngine::SHIMMER, 60, 128, 255, 0));
    TEST_ASSERT_FALSE(fx.staged());
}

void test_stage_rejects_unassigned_ids() {
    // 1 (was SOLID), 2 (was BREATHE), 5 (SPARKLE, undefined), 8-31 reserved.
    const uint8_t ids[] = {1, 2, 5, 8, 31};
    for (uint8_t id : ids) TEST_ASSERT_FALSE(stage(fx, id));
    TEST_ASSERT_FALSE(fx.staged());
}

void test_stage_accepts_every_implemented_id() {
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::NONE, 9, 9, 9, 9));  // params ignored
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::SHIMMER, 255, 255, 255, 255));
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::CHASE, 255, 255, 255, 1));
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::HUE_SPLIT, 255, N - 1));
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::FADE, 255));
}

void test_stage_rejects_chase_spacing_out_of_range() {
    TEST_ASSERT_FALSE(stage(fx, EffectEngine::CHASE, 0, 255, 0, 0));
    TEST_ASSERT_FALSE(stage(fx, EffectEngine::CHASE, 0, 255, 0, N));
    TEST_ASSERT_FALSE(stage(fx, EffectEngine::CHASE, 0, 255, 0, 255));
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::CHASE, 0, 255, 0, N - 1));
}

void test_stage_rejects_nonzero_reserved_params() {
    TEST_ASSERT_FALSE(stage(fx, EffectEngine::FADE, 230, 1, 0, 0));
    TEST_ASSERT_FALSE(stage(fx, EffectEngine::FADE, 230, 0, 0, 1));
    TEST_ASSERT_FALSE(stage(fx, EffectEngine::HUE_SPLIT, 10, 0, 1, 0));
    TEST_ASSERT_FALSE(stage(fx, EffectEngine::HUE_SPLIT, 10, N, 0, 0));
}

void test_rejected_stage_leaves_running_and_staged_effects_alone() {
    fill(buf, 200, 100, 50);
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::SHIMMER, 60, 255, 255, 1));
    fx.on_latch(buf, 0);
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::FADE, 230));

    TEST_ASSERT_FALSE(stage(fx, 5));
    TEST_ASSERT_EQUAL_UINT8(EffectEngine::SHIMMER, fx.id());
    TEST_ASSERT_TRUE(fx.staged());
    fx.on_latch(buf, 0);
    TEST_ASSERT_EQUAL_UINT8(EffectEngine::FADE, fx.id());
}

// ---------------------------------------------------------------------------
// The register: staged by SET_EFFECT, committed by LATCH, persistent
// ---------------------------------------------------------------------------

void test_no_effect_passes_the_buffer_straight_through() {
    for (uint8_t i = 0; i < N; i++) buf.leds[i] = {i, (uint8_t)(i * 2), (uint8_t)(255 - i)};
    fx.on_latch(buf, 0);
    assert_out_equals_buffer();
}

void test_stage_does_not_take_effect_until_latch() {
    fill(buf, 255, 255, 255);
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::SHIMMER, 60, 255, 255, 1));
    TEST_ASSERT_TRUE(fx.staged());
    TEST_ASSERT_EQUAL_UINT8(EffectEngine::NONE, fx.id());
    TEST_ASSERT_FALSE(fx.poll(buf, 10000));   // nothing runs before LATCH

    fx.on_latch(buf, 0);
    TEST_ASSERT_FALSE(fx.staged());
    TEST_ASSERT_EQUAL_UINT8(EffectEngine::SHIMMER, fx.id());
}

void test_effect_persists_across_latches_without_resend() {
    fill(buf, 255, 0, 0);
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::HUE_SPLIT, 20, 0));
    fx.on_latch(buf, 0);
    for (int n = 0; n < 10; n++) fx.on_latch(buf, (uint32_t)n * 33);
    TEST_ASSERT_EQUAL_UINT8(EffectEngine::HUE_SPLIT, fx.id());
    TEST_ASSERT_FALSE(memcmp(&buf.leds[0], &out(0), sizeof(Pixel)) == 0);
}

void test_none_clears_the_effect_without_blanking() {
    fill(buf, 255, 0, 0);
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::HUE_SPLIT, 40, 0));
    fx.on_latch(buf, 0);

    TEST_ASSERT_TRUE(stage(fx, EffectEngine::NONE));
    fx.on_latch(buf, 33);
    TEST_ASSERT_EQUAL_UINT8(EffectEngine::NONE, fx.id());
    assert_out_equals_buffer();                // buffer as-is, not black
    TEST_ASSERT_FALSE(fx.poll(buf, 10000));
}

void test_effect_never_writes_the_buffer() {
    fill(buf, 200, 100, 50);
    PixelBuffer before = buf;
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::SHIMMER, 200, 255, 255, 3));
    fx.on_latch(buf, 0);
    for (uint32_t t = 0; t < 1000; t += EffectEngine::FRAME_MS) fx.poll(buf, t);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(&before.leds[0].r, &buf.leds[0].r, N * 3);
}

void test_new_buffer_under_running_effect_is_used_without_rearm() {
    // The old pattern model snapshotted a base image at arm time; the effect
    // model reads the live buffer, so new content shows through at once.
    fill(buf, 255, 255, 255);
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::SHIMMER, 60, 0, 0, 0));   // depth 0
    fx.on_latch(buf, 0);
    fill(buf, 10, 20, 30);
    fx.on_latch(buf, 33);
    assert_pixel(10, 20, 30, out(0));
}

void test_latch_without_stage_does_not_restart_the_clock() {
    fill(buf, 255, 255, 255);
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::SHIMMER, 60, 255, 0, 0));
    fx.on_latch(buf, 0);
    uint32_t t = 0;
    for (int f = 0; f < 10; f++) fx.poll(buf, t += EffectEngine::FRAME_MS);
    const Pixel mid = out(0);

    // An ordinary host LATCH re-renders at the current phase: same output.
    fx.on_latch(buf, t);
    TEST_ASSERT_EQUAL_MEMORY(&mid, &out(0), sizeof(Pixel));
}

void test_restaging_the_same_effect_restarts_its_clock() {
    fill(buf, 255, 255, 255);
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::SHIMMER, 60, 255, 0, 0));
    fx.on_latch(buf, 0);
    const Pixel start = out(0);
    uint32_t t = 0;
    for (int f = 0; f < 10; f++) fx.poll(buf, t += EffectEngine::FRAME_MS);

    TEST_ASSERT_TRUE(stage(fx, EffectEngine::SHIMMER, 60, 255, 0, 0));
    fx.on_latch(buf, t);
    TEST_ASSERT_EQUAL_MEMORY(&start, &out(0), sizeof(Pixel));
}

void test_reset_clears_register_and_anything_staged() {
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::CHASE, 0, 255, 0, 3));
    fx.on_latch(buf, 0);
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::FADE, 200));

    fx.reset();
    TEST_ASSERT_EQUAL_UINT8(EffectEngine::NONE, fx.id());
    TEST_ASSERT_FALSE(fx.staged());
    TEST_ASSERT_FALSE(fx.poll(buf, 100000));

    fill(buf, 1, 2, 3);
    fx.on_latch(buf, 0);
    TEST_ASSERT_EQUAL_UINT8(EffectEngine::NONE, fx.id());
    assert_out_equals_buffer();
}

// The row controller's BLACKOUT sweep: SET_EFFECT(0), SET_COLOR(0,0,0), LATCH.
void test_blackout_sequence_is_black_even_from_fade_and_chase() {
    const uint8_t ids[] = {EffectEngine::FADE, EffectEngine::CHASE};
    for (uint8_t id : ids) {
        setUp();
        fill(buf, 255, 255, 255);
        TEST_ASSERT_TRUE(id == EffectEngine::FADE ? stage(fx, id, 255)
                                                   : stage(fx, id, 0, 255, 0, 1));
        fx.on_latch(buf, 0);

        TEST_ASSERT_TRUE(stage(fx, EffectEngine::NONE));
        fill(buf, 0, 0, 0);
        fx.on_latch(buf, 20);
        for (uint8_t i = 0; i < N; i++) assert_pixel(0, 0, 0, out(i));
        TEST_ASSERT_FALSE(fx.poll(buf, 100000));
    }
}

// ---------------------------------------------------------------------------
// SHIMMER
// ---------------------------------------------------------------------------

void test_shimmer_poll_respects_frame_interval() {
    fill(buf, 255, 255, 255);
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::SHIMMER, 60, 255, 255, 1));
    fx.on_latch(buf, 1000);

    TEST_ASSERT_FALSE(fx.poll(buf, 1000));
    TEST_ASSERT_FALSE(fx.poll(buf, 1000 + EffectEngine::FRAME_MS - 1));
    TEST_ASSERT_TRUE(fx.poll(buf, 1000 + EffectEngine::FRAME_MS));
}

void test_shimmer_depth_zero_leaves_buffer_untouched() {
    fill(buf, 200, 100, 50);
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::SHIMMER, 60, 0, 255, 1));
    fx.on_latch(buf, 0);
    uint32_t t = 0;
    for (int f = 0; f < 200; f++) {
        TEST_ASSERT_TRUE(fx.poll(buf, t += EffectEngine::FRAME_MS));
        assert_out_equals_buffer();
    }
}

void test_shimmer_only_ever_darkens() {
    fill(buf, 200, 100, 50);
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::SHIMMER, 60, 255, 255, 42));
    fx.on_latch(buf, 0);
    uint32_t t = 0;
    for (int f = 0; f < 300; f++) {
        fx.poll(buf, t += EffectEngine::FRAME_MS);
        for (uint8_t i = 0; i < N; i++) {
            TEST_ASSERT_LESS_OR_EQUAL_UINT8(200, out(i).r);
            TEST_ASSERT_LESS_OR_EQUAL_UINT8(100, out(i).g);
            TEST_ASSERT_LESS_OR_EQUAL_UINT8(50,  out(i).b);
        }
    }
}

void test_shimmer_full_depth_reaches_near_black_and_back() {
    fill(buf, 255, 255, 255);
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::SHIMMER, 60, 255, 0, 0));
    fx.on_latch(buf, 0);
    uint8_t lo = 255, hi = 0;
    uint32_t t = 0;
    for (int f = 0; f < 200; f++) {           // ~1 cycle at 60 cpm
        fx.poll(buf, t += EffectEngine::FRAME_MS);
        if (out(0).r < lo) lo = out(0).r;
        if (out(0).r > hi) hi = out(0).r;
    }
    TEST_ASSERT_LESS_THAN_UINT8(4, lo);
    TEST_ASSERT_GREATER_THAN_UINT8(250, hi);
}

void test_shimmer_spread_zero_moves_all_leds_together() {
    fill(buf, 255, 255, 255);
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::SHIMMER, 60, 255, 0, 7));
    fx.on_latch(buf, 0);
    TEST_ASSERT_TRUE(fx.poll(buf, EffectEngine::FRAME_MS));
    for (uint8_t i = 1; i < N; i++) TEST_ASSERT_EQUAL_UINT8(out(0).r, out(i).r);
}

void test_shimmer_spread_max_decorrelates_leds() {
    fill(buf, 255, 255, 255);
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::SHIMMER, 60, 255, 255, 7));
    fx.on_latch(buf, 0);
    TEST_ASSERT_TRUE(fx.poll(buf, EffectEngine::FRAME_MS));
    bool differs = false;
    for (uint8_t i = 1; i < N; i++)
        if (out(i).r != out(0).r) { differs = true; break; }
    TEST_ASSERT_TRUE(differs);
}

void test_shimmer_speed_zero_is_a_still_frame() {
    // Rendered on LATCH only - nothing to push on the frame clock.
    fill(buf, 255, 255, 255);
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::SHIMMER, 0, 255, 255, 3));
    fx.on_latch(buf, 0);
    TEST_ASSERT_FALSE(fx.poll(buf, EffectEngine::FRAME_MS * 10));
}

static PixelBuffer shimmer_frame(uint8_t seed) {
    EffectEngine e;
    PixelBuffer b{};
    fill(b, 255, 255, 255);
    TEST_ASSERT_TRUE(stage(e, EffectEngine::SHIMMER, 60, 255, 255, seed));
    e.on_latch(b, 0);
    e.poll(b, EffectEngine::FRAME_MS);
    return e.output();
}

void test_shimmer_same_seed_gives_identical_output() {
    PixelBuffer a = shimmer_frame(99), c = shimmer_frame(99);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(&a.leds[0].r, &c.leds[0].r, N * 3);
}

void test_shimmer_different_seed_gives_different_output() {
    PixelBuffer a = shimmer_frame(1), c = shimmer_frame(2);
    TEST_ASSERT_FALSE(memcmp(&a.leds[0].r, &c.leds[0].r, N * 3) == 0);
}

// ---------------------------------------------------------------------------
// CHASE
// ---------------------------------------------------------------------------

static bool is_chase_lit(uint8_t i, uint8_t step, uint8_t spacing) {
    // docs/tile-effects.md: lit when i ≡ step (mod spacing + 1).
    const int period = spacing + 1;
    return (((int)i - (int)step) % period + period) % period == 0;
}

static void assert_chase_frame(uint8_t step, uint8_t spacing, Pixel lit) {
    for (uint8_t i = 0; i < N; i++) {
        const Pixel expect = is_chase_lit(i, step, spacing) ? lit : buf.leds[i];
        TEST_ASSERT_EQUAL_MEMORY(&expect, &out(i), sizeof(Pixel));
    }
}

void test_chase_positions_match_the_doc_formula() {
    const uint8_t spacings[] = {1, 2, 6, 14, 29, 58, 59};
    for (uint8_t spacing : spacings) {
        setUp();
        for (uint8_t i = 0; i < N; i++) buf.leds[i] = {0, 0, (uint8_t)(i + 1)};
        TEST_ASSERT_TRUE(stage(fx, EffectEngine::CHASE, 0, 255, 0, spacing));
        fx.on_latch(buf, 0);
        assert_chase_frame(0, spacing, {255, 0, 0});

        uint32_t t = 0;
        for (uint16_t step = 1; step <= N + 3; step++) {
            TEST_ASSERT_TRUE(fx.poll(buf, t += EffectEngine::CHASE_BASE_MS));
            assert_chase_frame((uint8_t)(step % N), spacing, {255, 0, 0});
        }
    }
}

void test_chase_spacing_one_lights_every_other_led() {
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::CHASE, 0, 255, 0, 1));
    fx.on_latch(buf, 0);
    uint8_t lit = 0;
    for (uint8_t i = 0; i < N; i++) if (out(i).r) lit++;
    TEST_ASSERT_EQUAL_UINT8(N / 2, lit);
}

void test_chase_step_period_from_speed() {
    // speed 3 -> 40 + 30 = 70 ms per step.
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::CHASE, 0, 255, 3, 59));
    fx.on_latch(buf, 1000);
    TEST_ASSERT_FALSE(fx.poll(buf, 1069));
    TEST_ASSERT_TRUE(fx.poll(buf, 1070));
    TEST_ASSERT_FALSE(fx.poll(buf, 1139));
    TEST_ASSERT_TRUE(fx.poll(buf, 1140));
}

void test_chase_timing_accumulates_rather_than_drifting() {
    // Polled late every time, the steps still average the nominal period.
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::CHASE, 0, 255, 1, 59));  // 50 ms
    fx.on_latch(buf, 0);
    uint16_t steps = 0;
    for (uint32_t t = 0; t <= 5000; t += 20)
        if (fx.poll(buf, t)) steps++;
    TEST_ASSERT_EQUAL_UINT16(100, steps);
}

void test_chase_hue_and_brightness() {
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::CHASE, 85, 255, 0, 59));
    fx.on_latch(buf, 0);
    assert_pixel(0, 255, 0, out(0));

    setUp();
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::CHASE, 170, 127, 0, 59));
    fx.on_latch(buf, 0);
    assert_pixel(0, 0, 127, out(0));

    // Brightness 0: a dark gap travelling through the buffer.
    setUp();
    fill(buf, 50, 50, 50);
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::CHASE, 0, 0, 0, 59));
    fx.on_latch(buf, 0);
    assert_pixel(0, 0, 0, out(0));
    assert_pixel(50, 50, 50, out(1));
}

// ---------------------------------------------------------------------------
// HUE_SPLIT
// ---------------------------------------------------------------------------

void test_hue_split_shift_zero_is_identity() {
    for (uint8_t i = 0; i < N; i++) buf.leds[i] = {(uint8_t)(i * 4), 17, (uint8_t)(255 - i)};
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::HUE_SPLIT, 0, 0));
    fx.on_latch(buf, 0);
    assert_out_equals_buffer();
}

void test_hue_split_is_idempotent_and_stateless() {
    for (uint8_t i = 0; i < N; i++) buf.leds[i] = {(uint8_t)(i * 4), 90, (uint8_t)(255 - i)};
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::HUE_SPLIT, 25, 7));
    fx.on_latch(buf, 0);
    PixelBuffer first = fx.output();
    for (int n = 1; n < 5; n++) fx.on_latch(buf, (uint32_t)n * 33);
    TEST_ASSERT_EQUAL_UINT8_ARRAY(&first.leds[0].r, &fx.output().leds[0].r, N * 3);
    TEST_ASSERT_FALSE(fx.poll(buf, 100000));   // not time-varying
}

void test_hue_split_shifts_halves_in_opposite_directions() {
    fill(buf, 255, 0, 0);
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::HUE_SPLIT, 20, 0));
    fx.on_latch(buf, 0);
    // + half (LEDs 0..29): red toward green. - half: red toward blue.
    for (uint8_t i = 0; i < N; i++) {
        if (i < N / 2) {
            TEST_ASSERT_GREATER_THAN_UINT8(0, out(i).g);
            TEST_ASSERT_EQUAL_UINT8(0, out(i).b);
        } else {
            TEST_ASSERT_GREATER_THAN_UINT8(0, out(i).b);
            TEST_ASSERT_EQUAL_UINT8(0, out(i).g);
        }
    }
}

void test_hue_split_offset_moves_the_split() {
    fill(buf, 255, 0, 0);
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::HUE_SPLIT, 20, 50));
    fx.on_latch(buf, 0);
    // + half is LEDs 50..59 and 0..19, wrapping.
    for (uint8_t i = 0; i < N; i++) {
        const bool plus = i >= 50 || i < 20;
        TEST_ASSERT_EQUAL(plus, out(i).g > 0);
        TEST_ASSERT_EQUAL(!plus, out(i).b > 0);
    }
}

void test_hue_split_leaves_greys_and_black_untouched() {
    for (uint8_t i = 0; i < N; i++) buf.leds[i] = {(uint8_t)(i * 4), (uint8_t)(i * 4), (uint8_t)(i * 4)};
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::HUE_SPLIT, 200, 13));
    fx.on_latch(buf, 0);
    assert_out_equals_buffer();
}

void test_hue_split_preserves_brightness() {
    for (uint8_t i = 0; i < N; i++) buf.leds[i] = {(uint8_t)(i * 4), 60, (uint8_t)(255 - i * 3)};
    const uint8_t shifts[] = {1, 20, 84, 85, 86, 128, 200, 255};
    for (uint8_t s : shifts) {
        TEST_ASSERT_TRUE(stage(fx, EffectEngine::HUE_SPLIT, s, 0));
        fx.on_latch(buf, 0);
        for (uint8_t i = 0; i < N; i++) {
            const int in  = buf.leds[i].r + buf.leds[i].g + buf.leds[i].b;
            const int got = out(i).r + out(i).g + out(i).b;
            TEST_ASSERT_INT_WITHIN(3, in, got);   // rounding, one per channel
        }
    }
}

void test_hue_split_a_third_turn_rotates_channels() {
    buf.leds[0] = {200, 100, 50};                // + half
    buf.leds[N - 1] = {200, 100, 50};            // - half
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::HUE_SPLIT, 85, 0));
    fx.on_latch(buf, 0);
    assert_pixel(50, 200, 100, out(0));          // r->g, g->b, b->r
}

// ---------------------------------------------------------------------------
// FADE
// ---------------------------------------------------------------------------

void test_fade_released_pixel_decays_monotonically_to_exact_black() {
    const uint8_t ks[] = {1, 128, 230, 252, 255};
    for (uint8_t k : ks) {
        setUp();
        fill(buf, 255, 255, 255);
        TEST_ASSERT_TRUE(stage(fx, EffectEngine::FADE, k));
        fx.on_latch(buf, 0);

        buf.leds[5] = {0, 0, 0};                 // host releases LED 5
        fx.on_latch(buf, 0);
        uint8_t prev = out(5).r;
        TEST_ASSERT_EQUAL_UINT8(255, prev);      // not decayed by the LATCH itself

        uint32_t t = 0;
        uint16_t frames = 0;
        while (out(5).r != 0) {
            TEST_ASSERT_TRUE(fx.poll(buf, t += EffectEngine::FRAME_MS));
            TEST_ASSERT_LESS_THAN_UINT8(prev, out(5).r);
            prev = out(5).r;
            TEST_ASSERT_LESS_THAN_UINT16(300, ++frames);
        }
        assert_pixel(0, 0, 0, out(5));
        // Done: the frame clock stops rather than pushing a black tile forever.
        TEST_ASSERT_FALSE(fx.poll(buf, t += EffectEngine::FRAME_MS));
    }
}

void test_fade_tail_length_matches_the_doc_table() {
    // docs/tile-effects.md: decay 230 is black after 36 frames (720 ms).
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::FADE, 230));
    fill(buf, 255, 255, 255);
    fx.on_latch(buf, 0);
    fill(buf, 0, 0, 0);
    fx.on_latch(buf, 0);
    uint32_t t = 0;
    uint8_t frames = 0;
    while (fx.poll(buf, t += EffectEngine::FRAME_MS) && out(0).r) frames++;
    TEST_ASSERT_EQUAL_UINT8(36, frames + 1);
}

void test_fade_never_dims_a_driven_pixel() {
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::FADE, 100));
    fill(buf, 255, 255, 255);
    fx.on_latch(buf, 0);
    buf.leds[0] = {0, 0, 0};                     // keep something decaying
    fx.on_latch(buf, 0);

    uint32_t t = 0;
    for (int f = 0; f < 50; f++) {
        fx.poll(buf, t += EffectEngine::FRAME_MS);
        for (uint8_t i = 1; i < N; i++) assert_pixel(255, 255, 255, out(i));
    }
}

void test_fade_tail_completes_without_further_latch() {
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::FADE, 200));
    fill(buf, 255, 0, 0);
    fx.on_latch(buf, 0);
    fill(buf, 0, 0, 0);
    fx.on_latch(buf, 0);
    for (uint32_t t = 0; t < 2000; t += EffectEngine::FRAME_MS) fx.poll(buf, t);
    for (uint8_t i = 0; i < N; i++) assert_pixel(0, 0, 0, out(i));
}

void test_fade_relit_pixel_jumps_to_new_value() {
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::FADE, 230));
    fill(buf, 255, 255, 255);
    fx.on_latch(buf, 0);
    fill(buf, 0, 0, 0);
    fx.on_latch(buf, 0);
    for (uint32_t t = 20; t <= 100; t += 20) fx.poll(buf, t);

    fill(buf, 10, 200, 30);
    fx.on_latch(buf, 120);
    assert_out_equals_buffer();
}

void test_fade_decay_zero_is_indistinguishable_from_none() {
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::FADE, 0));
    fill(buf, 255, 255, 255);
    fx.on_latch(buf, 0);
    for (uint8_t i = 0; i < N; i += 2) buf.leds[i] = {0, 0, 0};
    fx.on_latch(buf, 33);
    assert_out_equals_buffer();                  // released pixels off at once
    TEST_ASSERT_FALSE(fx.poll(buf, 100000));
}

void test_fade_starts_from_what_was_showing() {
    // Arming FADE over a lit tile and releasing everything in the same LATCH
    // starts the tail from the old output - no black flash.
    fill(buf, 255, 255, 255);
    fx.on_latch(buf, 0);
    TEST_ASSERT_TRUE(stage(fx, EffectEngine::FADE, 230));
    fill(buf, 0, 0, 0);
    fx.on_latch(buf, 33);
    assert_pixel(255, 255, 255, out(0));
    TEST_ASSERT_TRUE(fx.poll(buf, 33 + EffectEngine::FRAME_MS));
    TEST_ASSERT_LESS_THAN_UINT8(255, out(0).r);
}

// ---------------------------------------------------------------------------

int main() {
    UNITY_BEGIN();
    RUN_TEST(test_stage_rejects_short_payload);
    RUN_TEST(test_stage_rejects_reserved_bits);
    RUN_TEST(test_stage_rejects_unassigned_ids);
    RUN_TEST(test_stage_accepts_every_implemented_id);
    RUN_TEST(test_stage_rejects_chase_spacing_out_of_range);
    RUN_TEST(test_stage_rejects_nonzero_reserved_params);
    RUN_TEST(test_rejected_stage_leaves_running_and_staged_effects_alone);

    RUN_TEST(test_no_effect_passes_the_buffer_straight_through);
    RUN_TEST(test_stage_does_not_take_effect_until_latch);
    RUN_TEST(test_effect_persists_across_latches_without_resend);
    RUN_TEST(test_none_clears_the_effect_without_blanking);
    RUN_TEST(test_effect_never_writes_the_buffer);
    RUN_TEST(test_new_buffer_under_running_effect_is_used_without_rearm);
    RUN_TEST(test_latch_without_stage_does_not_restart_the_clock);
    RUN_TEST(test_restaging_the_same_effect_restarts_its_clock);
    RUN_TEST(test_reset_clears_register_and_anything_staged);
    RUN_TEST(test_blackout_sequence_is_black_even_from_fade_and_chase);

    RUN_TEST(test_shimmer_poll_respects_frame_interval);
    RUN_TEST(test_shimmer_depth_zero_leaves_buffer_untouched);
    RUN_TEST(test_shimmer_only_ever_darkens);
    RUN_TEST(test_shimmer_full_depth_reaches_near_black_and_back);
    RUN_TEST(test_shimmer_spread_zero_moves_all_leds_together);
    RUN_TEST(test_shimmer_spread_max_decorrelates_leds);
    RUN_TEST(test_shimmer_speed_zero_is_a_still_frame);
    RUN_TEST(test_shimmer_same_seed_gives_identical_output);
    RUN_TEST(test_shimmer_different_seed_gives_different_output);

    RUN_TEST(test_chase_positions_match_the_doc_formula);
    RUN_TEST(test_chase_spacing_one_lights_every_other_led);
    RUN_TEST(test_chase_step_period_from_speed);
    RUN_TEST(test_chase_timing_accumulates_rather_than_drifting);
    RUN_TEST(test_chase_hue_and_brightness);

    RUN_TEST(test_hue_split_shift_zero_is_identity);
    RUN_TEST(test_hue_split_is_idempotent_and_stateless);
    RUN_TEST(test_hue_split_shifts_halves_in_opposite_directions);
    RUN_TEST(test_hue_split_offset_moves_the_split);
    RUN_TEST(test_hue_split_leaves_greys_and_black_untouched);
    RUN_TEST(test_hue_split_preserves_brightness);
    RUN_TEST(test_hue_split_a_third_turn_rotates_channels);

    RUN_TEST(test_fade_released_pixel_decays_monotonically_to_exact_black);
    RUN_TEST(test_fade_tail_length_matches_the_doc_table);
    RUN_TEST(test_fade_never_dims_a_driven_pixel);
    RUN_TEST(test_fade_tail_completes_without_further_latch);
    RUN_TEST(test_fade_relit_pixel_jumps_to_new_value);
    RUN_TEST(test_fade_decay_zero_is_indistinguishable_from_none);
    RUN_TEST(test_fade_starts_from_what_was_showing);
    return UNITY_END();
}
