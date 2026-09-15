import numpy as np
import pytest

from df2_pi.encode import DECODE_LUT, GAMMA, from_linear, to_linear

ALL_BYTES = np.arange(256, dtype=np.uint8)


def test_decode_lut_spans_zero_to_one_and_is_strictly_increasing():
    assert DECODE_LUT.shape == (256,)
    assert DECODE_LUT.dtype == np.float32
    assert DECODE_LUT[0] == 0.0
    assert DECODE_LUT[255] == 1.0
    assert (np.diff(DECODE_LUT) > 0).all()


def test_decode_is_a_power_law_not_srgb():
    # 128 is not half as bright as 255 - that is the whole reason this
    # module exists. Check the exact exponent rather than "it curves".
    assert DECODE_LUT[128] == pytest.approx((128 / 255) ** GAMMA, rel=1e-6)
    assert DECODE_LUT[128] < 0.25


def test_from_linear_round_trips_every_byte_value():
    # A LUT construction invariant the frame ops depend on: a uniform tile
    # must reduce to its own colour, and blend(a, b, 0) must be exactly a.
    assert (from_linear(to_linear(ALL_BYTES)) == ALL_BYTES).all()


def test_from_linear_saturates_rather_than_wrapping():
    out = from_linear(np.array([-0.5, 0.0, 1.0, 1.5, 100.0]))
    assert out.dtype == np.uint8
    assert out.tolist() == [0, 0, 255, 255, 255]


def test_half_light_encodes_to_186():
    # The number the frame tests lean on: the linear midpoint between
    # black and full white, re-encoded.
    assert from_linear(np.array(0.5)) == 186


def test_to_linear_preserves_shape_and_rejects_non_uint8():
    data = np.zeros((4, 5, 3), dtype=np.uint8)
    assert to_linear(data).shape == (4, 5, 3)
    with pytest.raises(TypeError):
        to_linear(data.astype(np.float32))


# ---- FrameEncoder -------------------------------------------------------------------

from df2_pi.effects import FADE, Effect  # noqa: E402
from df2_pi.encode import EncodeStats, FrameEncoder, parse_send_data  # noqa: E402
from df2_pi.geometry import FloorGeometry  # noqa: E402
from df2_pi.pixels import PixelFrame, TileFrame, default_geometry  # noqa: E402
from df2_pi.protocol.constants import MAX_PAYLOAD, TILE_ENTRY_SIZE, TileCmd  # noqa: E402

SET_COLOR, SET_EFFECT, SET_LEDS = TileCmd.SET_COLOR, TileCmd.SET_EFFECT, TileCmd.SET_LEDS


@pytest.fixture
def linear() -> FrameEncoder:
    """gamma=1 at full brightness: wire bytes == frame bytes."""
    return FrameEncoder(gamma=1.0)


def decode(payloads: list[bytes], geo=None) -> PixelFrame:
    """Decoder stub: rebuild the LED colours a row controller would forward."""
    geo = geo or default_geometry()
    frame = PixelFrame.black(geo)
    for row, payload in enumerate(payloads):
        for slot, (cmd, data) in enumerate(parse_send_data(payload)):
            tile = row * geo.tile_cols + slot
            if cmd is SET_COLOR:
                frame.tile(tile)[:] = tuple(data)
            elif cmd is SET_LEDS:
                frame.tile(tile)[:] = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
    return frame


def test_byte_exact_payloads_for_a_known_frame(linear):
    t = TileFrame.black()
    t[0, 0] = (255, 128, 1)
    t[0, 7] = (9, 8, 7)
    payloads = linear.encode(t)
    assert len(payloads) == 8
    expected_row0 = (
        bytes((SET_COLOR, 255, 128, 1)) + bytes((SET_COLOR, 0, 0, 0)) * 6 + bytes((SET_COLOR, 9, 8, 7))
    )
    assert payloads[0] == expected_row0
    assert all(p == bytes((SET_COLOR, 0, 0, 0)) * 8 for p in payloads[1:])

    p = t.to_pixels()
    p.tile(9)[0] = (1, 2, 3)  # row 1, slot 1 becomes non-uniform
    payloads = linear.encode(p)
    assert payloads[0] == expected_row0
    leds = bytes([1, 2, 3]) + bytes(59 * 3)
    assert payloads[1] == bytes((SET_COLOR, 0, 0, 0)) + bytes((SET_LEDS,)) + leds + bytes((SET_COLOR, 0, 0, 0)) * 6


def test_uniform_tiles_go_as_set_color_and_others_as_set_leds(linear):
    p = PixelFrame.black()
    p.tile(3)[:] = (10, 20, 30)
    p.tile(4)[:] = (10, 20, 30)
    p.tile(4)[59] = (10, 20, 31)
    entries = parse_send_data(linear.encode(p)[0])
    assert entries[3] == (SET_COLOR, bytes((10, 20, 30)))
    assert entries[4][0] is SET_LEDS and len(entries[4][1]) == 180
    assert entries[4][1][-3:] == bytes((10, 20, 31))
    assert linear.stats.entries == {SET_COLOR: 63, SET_LEDS: 1}


def test_fully_addressed_row_is_exactly_max_payload(linear):
    p = PixelFrame.black()
    p.data[:, 0] = 1  # every tile non-uniform
    payloads = linear.encode(p)
    assert all(len(x) == MAX_PAYLOAD == 1448 for x in payloads)
    assert linear.stats.total_bytes == 8 * 1448


def test_every_payload_length_is_the_sum_of_its_entry_sizes(linear):
    rng = np.random.default_rng(63)
    p = PixelFrame(rng.integers(0, 256, (64, 60, 3), dtype=np.uint8))
    p.tile(5)[:] = 7
    p.tile(20)[:] = 0
    for payload in linear.encode(p, {5: Effect.NONE, 21: Effect(FADE)}):
        entries = parse_send_data(payload)
        assert len(entries) == 8
        assert len(payload) == sum(TILE_ENTRY_SIZE[cmd] for cmd, _ in entries)


def test_tile_frame_and_its_pixels_encode_to_the_same_leds():
    enc = FrameEncoder()  # default gamma and all
    t = TileFrame.black()
    t.data[:] = np.arange(192, dtype=np.uint8).reshape(8, 8, 3)
    assert decode(enc.encode(t)) == decode(enc.encode(t.to_pixels()))
    assert enc.stats.entries == {SET_COLOR: 64}  # to_pixels() is still uniform per tile


def test_round_trip_through_the_decoder_stub(linear):
    rng = np.random.default_rng(1)
    p = PixelFrame(rng.integers(0, 256, (64, 60, 3), dtype=np.uint8))
    assert decode(linear.encode(p)) == p
    enc = FrameEncoder(brightness=200, white_balance=(1.0, 0.8, 0.9))
    expected = PixelFrame(enc.lut[p.data, np.arange(3)])
    assert decode(enc.encode(p)) == expected


# ---- the LUT ---------------------------------------------------------------------


def test_lut_is_monotonic_per_channel_and_endpoints_behave():
    enc = FrameEncoder(brightness=180, white_balance=(1.0, 0.7, 0.9))
    assert enc.lut.shape == (256, 3) and enc.lut.dtype == np.uint8
    assert (np.diff(enc.lut.astype(int), axis=0) >= 0).all()
    assert (enc.lut[0] == 0).all()
    enc.brightness = 0
    assert not enc.lut.any()
    enc.brightness = 255
    enc.white_balance = (1.0, 1.0, 1.0)
    enc.gamma = 1.0
    assert (enc.lut == np.arange(256)[:, None]).all()  # identity


def test_default_gamma_decodes_to_linear_for_the_leds():
    enc = FrameEncoder()
    assert enc.lut[255].tolist() == [255, 255, 255]
    assert enc.lut[128].tolist() == [56, 56, 56]  # round(255 * (128/255) ** 2.2)
    expected = np.rint(255 * (np.arange(256) / 255) ** GAMMA).astype(np.uint8)
    assert (enc.lut == expected[:, None]).all()
    # ...which is exactly the frame's own decode curve, so what the LED
    # emits is what to_tiles() and blend() thought they were computing
    assert (enc.lut[:, 0] == np.rint(to_linear(ALL_BYTES) * 255)).all()


def test_white_balance_scales_channels_in_linear_light():
    enc = FrameEncoder(gamma=1.0, white_balance=(1.0, 0.5, 0.25))
    assert enc.lut[200].tolist() == [200, 100, 50]
    enc = FrameEncoder(gamma=1.0, white_balance=(1.0, 2.0, 1.0))
    assert enc.lut[200].tolist() == [200, 255, 200]  # saturates, never wraps


def test_low_brightness_costs_bit_depth():
    enc = FrameEncoder(brightness=64)
    assert enc.lut[255].tolist() == [64, 64, 64]
    assert len(set(enc.lut[:, 0].tolist())) <= 65


def test_settings_are_validated():
    enc = FrameEncoder()
    with pytest.raises(ValueError):
        enc.brightness = 256
    with pytest.raises(ValueError):
        enc.gamma = 0
    with pytest.raises(ValueError):
        enc.white_balance = (1.0, 1.0)
    with pytest.raises(ValueError):
        enc.white_balance = (1.0, -1.0, 1.0)
    with pytest.raises(ValueError):
        enc.uniform_tolerance = 300


def test_uniform_tolerance_collapses_near_uniform_tiles(linear):
    p = PixelFrame.black()
    p.tile(0)[:] = (100, 100, 100)
    p.tile(0)[:30] = (102, 100, 100)
    assert parse_send_data(linear.encode(p)[0])[0][0] is SET_LEDS
    linear.uniform_tolerance = 2
    cmd, data = parse_send_data(linear.encode(p)[0])[0]
    assert cmd is SET_COLOR and data == bytes((101, 100, 100))  # the mean


# ---- effects --------------------------------------------------------------------


def test_an_effect_write_replaces_the_pixel_entry_for_one_frame(linear):
    p = PixelFrame.black()
    p.data[:, 0] = 5  # every tile would be SET_LEDS
    effect = Effect(FADE, (230, 0, 0, 0))
    entries = parse_send_data(linear.encode(p, {2: effect})[0])
    assert entries[2] == (SET_EFFECT, bytes((FADE, 230, 0, 0, 0)))
    assert len(bytes((SET_EFFECT,)) + bytes(effect)) == 6
    assert all(cmd is SET_LEDS for i, (cmd, _) in enumerate(entries) if i != 2)
    # the next frame, with no writes, the tile's pixel entry resumes
    entries = parse_send_data(linear.encode(p)[0])
    assert entries[2][0] is SET_LEDS
    assert linear.stats.entries == {SET_LEDS: 64}


def test_several_effects_in_one_row_land_in_the_same_send_data(linear):
    p = PixelFrame.black()
    payloads = linear.encode(p, {8: Effect(1), 9: Effect(2), 15: Effect.NONE})
    entries = parse_send_data(payloads[1])
    assert [cmd for cmd, _ in entries] == [SET_EFFECT, SET_EFFECT] + [SET_COLOR] * 5 + [SET_EFFECT]
    assert entries[7][1] == b"\x00" * 5  # Effect.NONE is an id-0 entry
    assert all(all(cmd is SET_COLOR for cmd, _ in parse_send_data(x)) for x in payloads[2:])


def test_one_effect_and_seven_set_leds_is_1273_bytes(linear):
    p = PixelFrame.black()
    p.data[:, 0] = 5
    payloads = linear.encode(p, {0: Effect(FADE)})
    assert len(payloads[0]) == 6 + 7 * 181 == 1273
    assert len(payloads[1]) == 1448


def test_bad_effect_writes_are_rejected(linear):
    p = PixelFrame.black()
    with pytest.raises(ValueError, match="effect id"):
        Effect(32)  # rejected before it can ever reach the encoder
    with pytest.raises(ValueError, match="tile must be"):
        linear.encode(p, {64: Effect.NONE})
    with pytest.raises(TypeError):
        linear.encode(p, {0: (1, 0, 0, 0, 0)})


# ---- also here --------------------------------------------------------------------


def test_blackout_payload(linear):
    assert linear.blackout_payload() == bytes((SET_COLOR, 0, 0, 0)) * 8
    black = PixelFrame.black()
    assert linear.encode(black) == [linear.blackout_payload()] * 8


def test_stats_and_wire_time_estimates(linear):
    t = TileFrame.black()
    linear.encode(t)
    assert isinstance(linear.stats, EncodeStats)
    assert linear.stats.row_bytes == (32,) * 8
    assert linear.stats.total_bytes == 256
    assert linear.stats.wire_seconds() == pytest.approx(4 * (8 + 32) * 10 / 3_125_000)  # ~0.5 ms
    p = PixelFrame.black()
    p.data[:, 0] = 1
    linear.encode(p)
    assert linear.stats.wire_seconds() == pytest.approx(4 * (8 + 1448) * 10 / 3_125_000)  # ~18.6 ms
    assert linear.stats.wire_seconds(chains=1) == pytest.approx(8 * (8 + 1448) * 10 / 3_125_000)


def test_geometry_mismatch_is_rejected(linear):
    small = FloorGeometry(tile_grid=(2, 2), leds_per_side=3)
    with pytest.raises(ValueError, match="geometry"):
        linear.encode(PixelFrame.black(small))
    enc = FrameEncoder(small, gamma=1.0)
    payloads = enc.encode(TileFrame.black(small))
    assert len(payloads) == 2 and all(len(x) == 8 for x in payloads)


def test_parse_send_data_rejects_truncated_entries():
    with pytest.raises(ValueError, match="truncated"):
        parse_send_data(bytes((SET_LEDS, 1, 2, 3)))
    assert parse_send_data(b"") == []
