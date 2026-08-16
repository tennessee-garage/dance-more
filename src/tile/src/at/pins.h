#pragma once
// Pin assignments for the ATtiny3224 tile board.
// Verified against pcb/tile/tile.kicad_sch (U1's net connections) — the
// nets there are LED_Data, SENSE_IN, SENSE_OUT, XDIR, TxD and RxD.

// WS2815 LED data line (800 KHz, GRB)
#define PIN_LED_DATA  PIN_PA1

// RS-485 direction control (THVD1420DR XDIR/DE)
// HIGH = TX (driver enabled), LOW = RX (default)
#define PIN_DE        PIN_PB0

// SENSE chain GPIO
// SENSE_IN:  driven LOW by the upstream node to signal this tile
// SENSE_OUT: driven LOW by this tile to signal the downstream node;
//            released by returning the pin to high-impedance input
#define PIN_SENSE_IN  PIN_PA2
#define PIN_SENSE_OUT PIN_PA3

// UART0 pin selection (PORTMUX.USARTROUTEA). megaTinyCore maps USART0 pins
// via Serial.swap(n) before Serial.begin(); per the txy4 variant's
// pins_arduino.h (HWSERIAL0_MUX_DEFAULT = 0):
//   swap 0 (default): TX=PB2, RX=PB3   <- what the PCB wires to the transceiver
//   swap 1 (alt):     TX=PA1, RX=PA2
// The default mux is already the one we want, so TransportAT::init() calls
// Serial.begin() with no swap(). Don't "fix" that by adding swap(1): the
// alternate mux lands TX on PA1 (the WS2815 data line) and RX on PA2
// (SENSE_IN), which would corrupt the LEDs and mute the Tile Bus at once.
//
// These two are documentation of that wiring, not inputs to Serial.begin() —
// megaTinyCore selects pins by mux index, not by pin number.
#define PIN_UART_TX   PIN_PB2
#define PIN_UART_RX   PIN_PB3
