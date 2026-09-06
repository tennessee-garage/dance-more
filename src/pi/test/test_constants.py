"""The wire format and the floor model both carry the LED count. Keep them honest.

protocol/constants.py deliberately does not import geometry.py - the wire
format shouldn't depend on the floor model - so nothing but this file stops
the two drifting apart.
"""

from df2_pi.geometry import FloorGeometry
from df2_pi.protocol import constants as c


def test_wire_constants_match_the_floor_geometry():
    geo = FloorGeometry.default()
    assert c.LEDS_PER_SIDE == geo.leds_per_side
    assert c.LEDS_PER_TILE == geo.leds_per_tile


def test_send_data_sizes_derive_from_the_led_count():
    assert c.SET_LEDS_PAYLOAD == c.LEDS_PER_TILE * c.BYTES_PER_LED
    assert c.TILE_ENTRY_SIZE[c.TileCmd.SET_LEDS] == 1 + c.SET_LEDS_PAYLOAD
    # The worst-case frame: every slot sending explicit pixels.
    assert c.MAX_PAYLOAD == c.ROW_SLOTS * c.TILE_ENTRY_SIZE[c.TileCmd.SET_LEDS]
    assert c.MAX_FRAME_SIZE == c.MAX_PAYLOAD + c.FRAME_OVERHEAD


def test_set_leds_payload_still_fits_tile_buss_one_byte_len():
    # Tile Bus's LEN is 1 byte and its whole frame is length-checked with
    # uint8_t arithmetic in src/common/tile_bus_protocol/protocol.h, which
    # static_asserts the same bound firmware-side. At 3 bytes/LED this caps a
    # side at 20; past that needs a 2-byte LEN or fewer bytes per LED.
    assert c.SET_LEDS_PAYLOAD + 7 <= 255


def test_the_build_spec_numbers():
    # Mirrors test_geometry.py's spec test - changing the floor's LED count
    # should have to be a deliberate edit in both places.
    assert c.LEDS_PER_SIDE == 15
    assert c.LEDS_PER_TILE == 60
    assert c.SET_LEDS_PAYLOAD == 180
    assert c.MAX_PAYLOAD == 1448
    assert c.MAX_FRAME_SIZE == 1456
