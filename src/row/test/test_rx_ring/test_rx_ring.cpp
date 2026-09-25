#include <unity.h>
#include "rx_ring.h"

void setUp() {}
void tearDown() {}

static uint8_t buf[16];

void test_empty_ring_has_nothing_available() {
    RxRing<16> r(buf);
    const uint8_t *p;
    TEST_ASSERT_EQUAL(0, r.available(0));
    TEST_ASSERT_EQUAL(0, r.contiguous(0, &p));
}

void test_contiguous_run_without_wrap() {
    RxRing<16> r(buf);
    const uint8_t *p;
    TEST_ASSERT_EQUAL(5, r.available(5));
    TEST_ASSERT_EQUAL(5, r.contiguous(5, &p));
    TEST_ASSERT_EQUAL_PTR(buf, p);
    r.consume(5);
    TEST_ASSERT_EQUAL(0, r.available(5));
}

void test_wrapped_region_takes_two_runs() {
    RxRing<16> r(buf);
    const uint8_t *p;
    r.consume(12);                    // reader at 12
    // Writer has written 12..15 and wrapped to 3: 7 bytes unread.
    TEST_ASSERT_EQUAL(7, r.available(3));
    TEST_ASSERT_EQUAL(4, r.contiguous(3, &p));
    TEST_ASSERT_EQUAL_PTR(buf + 12, p);
    r.consume(4);
    TEST_ASSERT_EQUAL(0, r.read_index());
    TEST_ASSERT_EQUAL(3, r.contiguous(3, &p));
    TEST_ASSERT_EQUAL_PTR(buf, p);
    r.consume(3);
    TEST_ASSERT_EQUAL(0, r.available(3));
}

void test_write_index_is_masked() {
    // The DMA write pointer is an address; callers pass it relative to the
    // buffer, and anything above SIZE must alias back into range.
    RxRing<16> r(buf);
    TEST_ASSERT_EQUAL(5, r.available(16 + 5));
}

void test_skip_to_discards_everything_unread() {
    RxRing<16> r(buf);
    r.consume(2);
    r.skip_to(9);
    TEST_ASSERT_EQUAL(9, r.read_index());
    TEST_ASSERT_EQUAL(0, r.available(9));
}

int main(int, char **) {
    UNITY_BEGIN();
    RUN_TEST(test_empty_ring_has_nothing_available);
    RUN_TEST(test_contiguous_run_without_wrap);
    RUN_TEST(test_wrapped_region_takes_two_runs);
    RUN_TEST(test_write_index_is_masked);
    RUN_TEST(test_skip_to_discards_everything_unread);
    return UNITY_END();
}
