#pragma once
// Pin assignments for the Seeed Xiao RP2350 row controller board.
// These are tentative values from the issue spec — verify against
// pcb/row-controller/ KiCad schematic before flashing hardware.

// Status LEDs — active-low (pin sinks current to light the LED)
#define PIN_LED_READY  D0
#define PIN_LED_DATA   D1

// RS-485 direction control (THVD1420DR DE/RE)
// LOW = RX (default), HIGH = TX
#define PIN_PI_XDIR    D2   // Row Bus (Pi-facing) transceiver
#define PIN_ROW_XDIR   D3   // Tile Bus (row-facing) transceiver

// I2C — INA220BIDGSR power monitor (default Wire/I2C0 pins on this board)
#define PIN_SDA        D4
#define PIN_SCL        D5

// Tile Bus UART (row controller is bus master)
#define PIN_ROW_TX     D6
#define PIN_ROW_RX     D7

// Tile Bus SENSE line (row controller runs SENSE auto-mapping)
#define PIN_ROW_SENSE  D8

// Row Bus UART (upstream to Raspberry Pi)
#define PIN_PI_RX      D9
#define PIN_PI_TX      D10
