#pragma once

class IStatusLed {
public:
    virtual void init() = 0;
    virtual void set_ready(bool on) = 0;  // true = lit
    virtual void set_data(bool on) = 0;   // true = lit
};
