#!/usr/bin/env bash
#
# Watch src/pi locally and push changes to a Raspberry Pi via rsync.
#
# The local src/pi/ *contents* land directly in PI_DEST on the Pi (i.e.
# src/pi/pyproject.toml -> $PI_DEST/pyproject.toml), not nested under a
# "pi" subdirectory.
#
# Usage:
#   tools/sync-to-pi.sh              # one-shot sync, then watch and re-sync on change
#   tools/sync-to-pi.sh --once       # sync once and exit, no watching
#
# Override the target with env vars:
#   PI_HOST=garth@testing-pi PI_DEST=/home/garth/dance-floor tools/sync-to-pi.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # src/pi
PI_HOST="${PI_HOST:-garth@testing-pi}"
PI_DEST="${PI_DEST:-/home/garth/dance-floor}"

RSYNC_EXCLUDES=(
  --exclude=venv/
  --exclude=__pycache__/
  --exclude=.pytest_cache/
  --exclude=*.egg-info/
  --exclude=.git/
  --exclude=.DS_Store
  --exclude=build/
  --exclude=dist/
)

sync_once() {
  echo "[sync-to-pi] $(date '+%H:%M:%S') syncing ${HERE}/ -> ${PI_HOST}:${PI_DEST}/"
  if rsync -avz --delete "${RSYNC_EXCLUDES[@]}" "${HERE}/" "${PI_HOST}:${PI_DEST}/"; then
    echo "[sync-to-pi] done"
  else
    echo "[sync-to-pi] rsync failed (Pi unreachable?) - will retry on next change" >&2
  fi
}

sync_once

if [[ "${1:-}" == "--once" ]]; then
  exit 0
fi

if ! command -v fswatch >/dev/null 2>&1; then
  echo "fswatch not found - install with: brew install fswatch" >&2
  exit 1
fi

echo "[sync-to-pi] watching ${HERE} for changes (Ctrl-C to stop)"
fswatch -o -l 0.5 \
  --exclude='venv/' \
  --exclude='__pycache__' \
  --exclude='pytest_cache' \
  --exclude='egg-info' \
  --exclude='\.git/' \
  --exclude='\.DS_Store' \
  --exclude='build/' \
  --exclude='dist/' \
  "$HERE" |
while read -r _; do
  sync_once
done
