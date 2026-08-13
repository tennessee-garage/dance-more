#pragma once
// Pin assignments for the Seeed Xiao RP2350 row controller board.
// Verified against pcb/row-controller/row-controller.kicad_sch (U1's net
// connections) — not the issue spec, whose D3/D9/D10 assignments were
// rotated relative to the actual board.
//
// Dn labels are the Xiao's silkscreen pads; the GPIO numbers in the
// comments below come from framework-arduinopico's seeed_xiao_rp2350
// variant. They matter because RP2350 UART function is only available on
// specific GPIOs: UART0 TX {0,12,16,28} / RX {1,13,17,29}, UART1 TX
// {4,8,20,24} / RX {5,9,21,25}. setTX()/setRX() silently return false for
// anything else and begin() then falls back to the variant's default pins,
// so a wrong assignment here fails quietly rather than loudly.

// Status LEDs — active-low (pin sinks current to light the LED)
#define PIN_LED_READY  D0
#define PIN_LED_DATA   D1

// RS-485 direction control (THVD1420DR DE/RE)
// LOW = RX (default), HIGH = TX
#define PIN_PI_XDIR    D2   // GPIO28 — Row Bus (Pi-facing) transceiver
#define PIN_ROW_XDIR   D10  // GPIO3  — Tile Bus (row-facing) transceiver

// I2C — INA220BIDGSR power monitor (default Wire/I2C0 pins on this board)
#define PIN_SDA        D4
#define PIN_SCL        D5

// Tile Bus UART (row controller is bus master) — Serial1 / UART0 defaults
#define PIN_ROW_TX     D6   // GPIO0 — UART0 TX
#define PIN_ROW_RX     D7   // GPIO1 — UART0 RX

// Tile Bus SENSE line (row controller runs SENSE auto-mapping)
#define PIN_ROW_SENSE  D8

// Row Bus UART (upstream to Raspberry Pi) — Serial2 / UART1, remapped off
// that variant's defaults (GPIO20/21) onto the pins the PCB actually uses.
#define PIN_PI_RX      D3   // GPIO5 — UART1 RX
#define PIN_PI_TX      D9   // GPIO4 — UART1 TX
