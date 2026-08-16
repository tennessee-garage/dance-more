#include <Arduino.h>
#include <Wire.h>
#include "power_monitor_rp2350.h"

// INA226-Q1 power monitor (docs/datasheets/ina226-q1.pdf). This board was
// originally laid out for the INA219/INA220 family, and the driver was
// written against that family's register format; the part itself was later
// swapped to the INA226 (INA219/220 went EOL on Mouser) without the driver
// being updated. Same package pinout and I2C address strapping, but the
// register contents differ throughout - CONFIG's bit fields are laid out
// differently, the Bus Voltage Register is a full 16-bit field rather than
// 13 bits packed into bits 15:3, and the calibration constant is 8x smaller.
//
// The mismatch was silent rather than a bus error, because addressing and
// register offsets (0x00 CONFIG .. 0x05 CALIBRATION) happen to be identical
// between the two families - only the bit-level meaning inside each register
// differs. It surfaced on the bench as bus voltage reading a suspiciously
// clean 0.4x of the true rail (measured against three known-good supply
// settings: 12V/11V/9V all landed within 20 mV of exactly 40%) - consistent
// with computing (raw >> 3) * 4mV against a register that actually holds the
// value unshifted at 1.25 mV/bit: (raw >> 3) * 4 ~= raw * 0.5, against the
// true raw * 1.25, and 0.5 / 1.25 = 0.4 exactly.
//
// A0 and A1 both tied to VS -> I2C address 0x45 (Table 6-2) - this part of
// the old comment was correct, the address table is identical between the
// two families.
static constexpr uint8_t INA226_ADDR = 0x45;

static constexpr uint8_t REG_CONFIG      = 0x00;
static constexpr uint8_t REG_SHUNT_V     = 0x01; // unused - shunt voltage not read directly
static constexpr uint8_t REG_BUS_V       = 0x02;
static constexpr uint8_t REG_POWER       = 0x03;
static constexpr uint8_t REG_CURRENT     = 0x04;
static constexpr uint8_t REG_CALIBRATION = 0x05;

// Power-on-reset value (Table 7-2), written back explicitly rather than
// relied upon: AVG=000 (1 sample), VBUSCT=100 (1.1 ms bus conversion),
// VSHCT=010 (332 us shunt conversion), MODE=111 (shunt + bus, continuous).
// Same operating point the INA219/220 driver intended by using its own
// POR default - continuous conversion of both channels, no averaging.
static constexpr uint16_t CONFIG_VALUE = 0x4127;

// 5 mOhm shunt on the row's 12V rail (docs/power.md); row current peaks
// ~4A at full white. Calibrate for 5A of headroom rather than that
// measured max, per the issue's guidance - unchanged from the original
// INA219/220 sizing, only the formula that turns it into a register value
// differs for this chip.
static constexpr float R_SHUNT_OHMS           = 0.005f;
static constexpr float MAX_EXPECTED_CURRENT_A = 5.0f;
// Current_LSB = Max_Expected_Current / 2^15 (Equation 2), rounded up to a
// round number - 152.6 uA/bit minimum, 200 uA/bit chosen for a clean
// mA-per-count relationship, giving 6.55A of headroom above the 5A target.
static constexpr float CURRENT_LSB_MA = 0.2f;
// Power_LSB is fixed at 25x Current_LSB for the INA226 (the INA219/220
// datasheet's driver code this replaced used a 20x fixed ratio - a
// different constant on a different part, not a typo carried over).
static constexpr float POWER_LSB_MW = 25.0f * CURRENT_LSB_MA;
// CAL = 0.00512 / (Current_LSB[A] * R_shunt) - Equation 1. The 0.00512
// constant is INA226-specific; the INA219/220 equivalent is 0.04096, exactly
// 8x larger, which is what silently inflated every current/power reading by
// 8x before this fix (the calibration register has no cross-checking; the
// silicon just uses whatever value is written).
static constexpr uint16_t CALIBRATION_VALUE =
    (uint16_t)(0.00512f / ((CURRENT_LSB_MA / 1000.0f) * R_SHUNT_OHMS));

static void write_reg(uint8_t reg, uint16_t value) {
    Wire.beginTransmission(INA226_ADDR);
    Wire.write(reg);
    Wire.write((uint8_t)(value >> 8));
    Wire.write((uint8_t)(value & 0xFF));
    Wire.endTransmission();
}

static uint16_t read_reg(uint8_t reg) {
    Wire.beginTransmission(INA226_ADDR);
    Wire.write(reg);
    Wire.endTransmission(false);  // repeated start, keep the bus held
    Wire.requestFrom((uint8_t)INA226_ADDR, (uint8_t)2);
    uint16_t hi = Wire.read();
    uint16_t lo = Wire.read();
    return (uint16_t)((hi << 8) | lo);
}

void PowerMonitorRP2350::init() {
    Wire.begin();
    write_reg(REG_CONFIG, CONFIG_VALUE);
    write_reg(REG_CALIBRATION, CALIBRATION_VALUE);
}

PowerReading PowerMonitorRP2350::read() {
    // The calibration register is known to reset to 0 under electrical
    // noise on this device family, so it is rewritten before every
    // current/power read rather than trusting it survived since init().
    write_reg(REG_CALIBRATION, CALIBRATION_VALUE);

    uint16_t bus_raw     = read_reg(REG_BUS_V);
    int16_t  current_raw = (int16_t)read_reg(REG_CURRENT);
    uint16_t power_raw   = read_reg(REG_POWER);

    PowerReading r;
    // Bus Voltage Register (Table 7-8): D15 is always 0 (bus voltage can
    // only be positive); D14:D0 is the value directly, LSB = 1.25 mV. No
    // shift - unlike the INA219/220, where the 13-bit field sits in bits
    // 15:3 above 3 status bits. (raw * 5) / 4 == raw * 1.25 while staying
    // in integer arithmetic.
    r.voltage_mV = (uint16_t)(((uint32_t)bus_raw * 5) / 4);
    r.current_mA = (uint16_t)((current_raw < 0 ? 0 : current_raw) * CURRENT_LSB_MA);
    r.power_mW   = (uint16_t)(power_raw * POWER_LSB_MW);
    return r;
}
