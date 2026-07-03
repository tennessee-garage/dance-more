CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra \
            -I../lib/df2_core \
            -I../include \
            -I../src \
            -I../src/native

BIN := bin

.PHONY: all clean

all: $(BIN)/tile-bus-broker

$(BIN):
	mkdir -p $(BIN)

# The broker only needs protocol.h for MAX_FRAME_SIZE; it doesn't parse frames
# itself so no .cpp lib files are required.
$(BIN)/tile-bus-broker: tile-bus-broker/main.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -rf $(BIN)
