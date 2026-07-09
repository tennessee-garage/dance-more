#include <unity.h>
#include "tile_map.h"

void setUp() {}
void tearDown() {}

void test_reset_clears_all_slots() {
    TileMap map;
    map.reset();
    for (uint8_t i = 0; i < TileMap::NUM_SLOTS; i++)
        TEST_ASSERT_FALSE(map.is_discovered(i));
    TEST_ASSERT_EQUAL(0, map.discovered_count());
}

void test_set_discovered_populates_slot() {
    TileMap map;
    map.reset();
    map.set_discovered(3, 0x07);

    TEST_ASSERT_TRUE(map.is_discovered(3));
    TEST_ASSERT_EQUAL_HEX8(0x07, map.address_for(3));
    TEST_ASSERT_EQUAL(TileStatus::OK, map.status_for(3));
    TEST_ASSERT_EQUAL(1, map.discovered_count());
}

void test_retry_count_is_per_slot() {
    TileMap map;
    map.reset();
    map.increment_retry(3);
    map.increment_retry(3);
    map.increment_retry(3);

    TEST_ASSERT_EQUAL(3, map.retry_count(3));
    TEST_ASSERT_EQUAL(0, map.retry_count(4)); // untouched, confirms per-slot isolation
}

void test_set_status_does_not_affect_address_or_retry_count() {
    TileMap map;
    map.reset();
    map.set_discovered(3, 0x07);
    map.increment_retry(3);

    map.set_status(3, TileStatus::NON_RESPONSIVE);

    TEST_ASSERT_EQUAL(TileStatus::NON_RESPONSIVE, map.status_for(3));
    TEST_ASSERT_EQUAL_HEX8(0x07, map.address_for(3));
    TEST_ASSERT_EQUAL(1, map.retry_count(3));
}

void test_discovered_count_counts_only_discovered_slots() {
    TileMap map;
    map.reset();
    map.set_discovered(0, 0x01);
    map.set_discovered(1, 0x02);
    map.set_discovered(5, 0x06);

    TEST_ASSERT_EQUAL(3, map.discovered_count());
}

int main(int, char **) {
    UNITY_BEGIN();

    RUN_TEST(test_reset_clears_all_slots);
    RUN_TEST(test_set_discovered_populates_slot);
    RUN_TEST(test_retry_count_is_per_slot);
    RUN_TEST(test_set_status_does_not_affect_address_or_retry_count);
    RUN_TEST(test_discovered_count_counts_only_discovered_slots);

    return UNITY_END();
}
