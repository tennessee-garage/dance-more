#!/usr/bin/env bash
# Full row-controller programming cycle: assign/look up this board's Row Bus
# address from its RP2350 CHIPID, regenerate the address header, then build
# and flash. Requires the board in BOOTSEL mode.
#
# Usage: tools/flash_row.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$HERE")"

python3 "$HERE/assign_row_address.py"

cd "$PROJECT"
pio run -e seeed_xiao_rp2350 --target upload
