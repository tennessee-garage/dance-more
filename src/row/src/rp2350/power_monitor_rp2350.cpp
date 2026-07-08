#include <Arduino.h>
#include <Wire.h>
#include "power_monitor_rp2350.h"

// INA220BIDGSR power monitor. Register map, calibration formula, and bus
// voltage decoding are identical to the INA219's (TI datasheet §8.5,
// §8.6.3.2). Verified against Adafruit_INA219's reference implementation.
//
// A0 and A1 both tied to Vs -> I2C address 0x45 (datasheet Table 2).
static constexpr uint8_t INA220_ADDR = 0x45;

static constexpr uint8_t REG_CONFIG      = 0x00;
static constexpr uint8_t REG_BUS_V       = 0x02;
static constexpr uint8_t REG_POWER       = 0x03;
static constexpr uint8_t REG_CURRENT     = 0x04;
static constexpr uint8_t REG_CALIBRATION = 0x05;

// BRNG=32V, PGA=/8 (+-320mV shunt range), BADC/SADC=12-bit, continuous
// shunt+bus conversion. This is the chip's power-on-reset default
// (0x399F) and comfortably covers our ~25 mV max shunt swing (see below).
static constexpr uint16_t CONFIG_VALUE = 0x399F;

// 5 mOhm shunt on the row's 12V rail (docs/power.md); row current peaks
// ~4A at full white. Calibrate for 5A of headroom rather than that
// measured max, per the issue's guidance.
static constexpr float R_SHUNT_OHMS           = 0.005f;
static constexpr float MAX_EXPECTED_CURRENT_A = 5.0f;
// Current_LSB = Max_Expected_Current / 2^15, rounded up to a round number.
static constexpr float CURRENT_LSB_MA = 0.2f;
static constexpr float POWER_LSB_MW   = 20.0f * CURRENT_LSB_MA;
// Calibration = trunc(0.04096 / (Current_LSB[A] * R_SHUNT)).
static constexpr uint16_t CALIBRATION_VALUE =
    (uint16_t)(0.04096f / ((CURRENT_LSB_MA / 1000.0f) * R_SHUNT_OHMS));

static void write_reg(uint8_t reg, uint16_t value) {
    Wire.beginTransmission(INA220_ADDR);
    Wire.write(reg);
    Wire.write((uint8_t)(value >> 8));
    Wire.write((uint8_t)(value & 0xFF));
    Wire.endTransmission();
}

static uint16_t read_reg(uint8_t reg) {
    Wire.beginTransmission(INA220_ADDR);
    Wire.write(reg);
    Wire.endTransmission(false);  // repeated start, keep the bus held
    Wire.requestFrom((uint8_t)INA220_ADDR, (uint8_t)2);
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
    // noise on INA219-family parts; rewrite it before every current/power
    // read rather than trusting it survived since init().
    write_reg(REG_CALIBRATION, CALIBRATION_VALUE);

    uint16_t bus_raw     = read_reg(REG_BUS_V);
    int16_t  current_raw = (int16_t)read_reg(REG_CURRENT);
    uint16_t power_raw   = read_reg(REG_POWER);

    PowerReading r;
    r.voltage_mV = (uint16_t)((bus_raw >> 3) * 4);
    r.current_mA = (uint16_t)((current_raw < 0 ? 0 : current_raw) * CURRENT_LSB_MA);
    r.power_mW   = (uint16_t)(power_raw * POWER_LSB_MW);
    return r;
}
