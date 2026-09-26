#pragma once
#include <stddef.h>
#include <stdint.h>

// Read side of a byte ring that something else - on the RP2350, a DMA
// channel draining the Row Bus UART - writes into. The writer's position is
// passed in on every call rather than stored, because it is a hardware
// register that moves on its own.
//
// SIZE must be a power of two: the DMA ring wraps its write address on a
// power-of-two boundary, and indices here wrap with the same mask.
//
// There is no way to tell from the indices alone that the writer has lapped
// the reader - a full lap looks exactly like an empty ring. Keeping the
// reader close enough behind that a lap cannot happen is the caller's job;
// see PiTransportRP2350::poll().
template <size_t SIZE>
class RxRing {
    static_assert(SIZE >= 2 && (SIZE & (SIZE - 1)) == 0, "RxRing size must be a power of two");

public:
    static constexpr size_t MASK = SIZE - 1;

    explicit RxRing(const uint8_t *buf) : buf_(buf) {}

    // Bytes written but not yet consumed.
    size_t available(size_t write_idx) const { return (write_idx - read_) & MASK; }

    // The unread bytes that sit contiguously from the read position: up to
    // the writer, or up to the end of the buffer if the unread region wraps.
    // A wrapped region therefore takes two calls.
    size_t contiguous(size_t write_idx, const uint8_t **data) const {
        size_t n    = available(write_idx);
        size_t tail = SIZE - read_;
        *data = buf_ + read_;
        return n < tail ? n : tail;
    }

    void consume(size_t n) { read_ = (read_ + n) & MASK; }

    // Discard everything unread.
    void skip_to(size_t write_idx) { read_ = write_idx & MASK; }

    size_t read_index() const { return read_; }

private:
    const uint8_t *buf_;
    size_t         read_ = 0;
};
