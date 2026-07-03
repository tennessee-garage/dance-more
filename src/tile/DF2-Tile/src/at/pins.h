#pragma once
// Pin assignments for the ATtiny3224 tile board.
// These are tentative values from the issue spec — verify against
// pcb/tile/tile.kicad_sch and update before flashing hardware.

// RS-485 direction control (THVD1420DR XDIR/DE)
// HIGH = TX (driver enabled), LOW = RX (default)
#define PIN_DE          PIN_PB0

// SENSE chain GPIO
// SENSE_IN:  driven LOW by the upstream node to signal this tile
// SENSE_OUT: driven LOW by this tile to signal the downstream node;
//            released by returning the pin to high-impedance input
#define PIN_SENSE_IN    PIN_PA2
#define PIN_SENSE_OUT   PIN_PA3

// UART0 pin selection (ATtiny3224 datasheet Table 6-1, PORTMUX.USARTROUTEA).
// megaTinyCore maps USART0 pins via Serial.swap(n) before Serial.begin():
//   swap 0 (default): TX=PA1, RX=PA2
//   swap 1 (alt):     TX=PB2, RX=PB3
#define PIN_UART_TX   PIN_PB2
#define PIN_UART_RX   PIN_PB3
#define UART0_SWAP    1
