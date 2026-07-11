#include <Arduino.h>
#include "rp2350/pins.h"
#include "row_address_generated.h"

void setup() {
    Serial.begin(115200);
}

void loop() {
    Serial.print("row controller: not yet implemented (row address 0x");
    Serial.print(MY_ROW_ADDR, HEX);
    Serial.println(")");
    delay(1000);
}
