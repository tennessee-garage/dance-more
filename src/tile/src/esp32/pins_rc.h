#pragma once
// Pin assignments for the XIAO ESP32-C3 row-controller tester.
// Verify all pins against your physical wiring before use.

// RS-485 UART (Serial1)
#define PIN_RC_TX   21    // GPIO21 — to transceiver DI
#define PIN_RC_RX   20    // GPIO20 — from transceiver RO

// RS-485 direction control
// HIGH = TX (driver enabled), LOW = RX (default)
#define PIN_RC_DE   5     // GPIO5  — to transceiver DE/RE
