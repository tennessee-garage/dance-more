#pragma once
#include <pico/mutex.h>
#include "row_bus_protocol.h"

// Fixed-capacity, mutex-guarded ring buffer for handing RowBusFrame values
// between the RP2350's two cores (#45). One instance per direction - this
// is not a general-purpose multi-producer queue, just a small SPSC handoff.
//
// A pico_util queue_t (pico/util/queue.h) would do this job too, but it
// calloc()s its backing storage at init time; this project keeps everything
// statically sized (no dynamic allocation anywhere, see TileMap etc.), so
// this uses a fixed member array instead.
class RowBusFrameQueue {
public:
    static constexpr uint8_t CAPACITY = 4;

    RowBusFrameQueue() { mutex_init(&mutex_); }

    // Returns false (frame dropped) if the queue is full.
    bool try_push(const RowBusFrame &frame) {
        mutex_enter_blocking(&mutex_);
        bool ok = count_ < CAPACITY;
        if (ok) {
            slots_[head_] = frame;
            head_ = (uint8_t)((head_ + 1) % CAPACITY);
            count_++;
        }
        mutex_exit(&mutex_);
        return ok;
    }

    // Returns false if the queue is empty.
    bool try_pop(RowBusFrame *out) {
        mutex_enter_blocking(&mutex_);
        bool ok = count_ > 0;
        if (ok) {
            *out = slots_[tail_];
            tail_ = (uint8_t)((tail_ + 1) % CAPACITY);
            count_--;
        }
        mutex_exit(&mutex_);
        return ok;
    }

private:
    mutex_t     mutex_;
    RowBusFrame slots_[CAPACITY]{};
    uint8_t     head_  = 0;
    uint8_t     tail_  = 0;
    uint8_t     count_ = 0;
};
