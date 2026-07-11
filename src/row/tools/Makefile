CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra \
            -I../include \
            -I../lib/row_core \
            -I../src/native

BIN := bin

.PHONY: all clean

all: $(BIN)/row-bus-broker $(BIN)/row-stub

$(BIN):
	mkdir -p $(BIN)

# The broker only needs row_bus_protocol.h for ROWBUS_MAX_FRAME; it doesn't
# parse frames itself so no .cpp lib files are required.
$(BIN)/row-bus-broker: row-bus-broker/main.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@

# row-stub links PiTransportNative + its row_bus_protocol.cpp dependency
# directly (not via PlatformIO) since it's a test-only tool, not shipped
# firmware.
$(BIN)/row-stub: row-stub/main.cpp ../src/native/pi_transport_native.cpp ../lib/row_core/row_bus_protocol.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -rf $(BIN)
