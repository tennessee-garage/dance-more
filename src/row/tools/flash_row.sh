#!/usr/bin/env bash
# Build and upload row controller firmware for a specific row address.
#
# Addressing is a build-time choice (row0..row7 in platformio.ini, see
# include/row_address.h) rather than something read off the board, so this
# is a thin wrapper: it just picks the matching PlatformIO environment.
# Uploading directly with the PlatformIO extension's own env picker (e.g.
# VS Code's "Big Buttons") works identically - this script exists for the
# CLI / scripted case.
#
# Usage: tools/flash_row.sh <row>   (row is 0-7)

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: tools/flash_row.sh <row>   (row is 0-7)" >&2
  exit 1
fi

ROW="$1"
if ! [[ "$ROW" =~ ^[0-7]$ ]]; then
  echo "row must be a single digit 0-7, got: $ROW" >&2
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$HERE")"

cd "$PROJECT"
pio run -e "row${ROW}" --target upload
