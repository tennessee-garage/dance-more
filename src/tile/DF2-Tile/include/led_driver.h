#pragma once
#include "pixel_buffer.h"

class ILedDriver {
public:
    virtual ~ILedDriver() = default;
    virtual void init() = 0;
    virtual void push(const PixelBuffer &buf) = 0;
};
