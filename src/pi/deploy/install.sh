#!/usr/bin/env bash
# Install the df2-pi systemd unit: copy it, enable it for boot, and create
# the data directory its DF2_DB points into. Safe to re-run after the unit
# changes. Does not start or restart the service - that interrupts the floor.
#
#   sudo deploy/install.sh              # on the Pi, from ~/dance-floor
#   deploy/install.sh --dry-run         # print what it would do; needs no root
#
# The user, package directory and database path are read from the unit, so
# the unit is the one place they are set.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT="$HERE/df2-pi.service"
TARGET=/etc/systemd/system/df2-pi.service

DRY_RUN=0
case "${1:-}" in
  --dry-run) DRY_RUN=1 ;;
  "") ;;
  *) echo "usage: $0 [--dry-run]" >&2; exit 2 ;;
esac

unit_value() { sed -n "s/^$1=//p" "$UNIT" | head -n 1; }
SERVICE_USER="$(unit_value User)"
PKG_DIR="$(unit_value WorkingDirectory)"
DB="$(sed -n 's/^Environment=DF2_DB=//p' "$UNIT")"
DATA_DIR="$(dirname "$DB")"
if [[ -z "$SERVICE_USER" || -z "$PKG_DIR" || -z "$DB" ]]; then
  echo "install: could not read User, WorkingDirectory and DF2_DB from $UNIT" >&2
  exit 1
fi

run() {
  echo "+ $*"
  if [[ $DRY_RUN -eq 0 ]]; then "$@"; fi
}

if [[ $DRY_RUN -eq 0 ]]; then
  if [[ $EUID -ne 0 ]]; then
    echo "install: run with sudo (or --dry-run to see the plan)" >&2
    exit 1
  fi
  id "$SERVICE_USER" >/dev/null
  # A unit that cannot import the web extra restarts every RestartSec forever.
  if ! "$PKG_DIR/venv/bin/python" -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "install: warning: $PKG_DIR/venv lacks the web extra;" \
      "run: cd $PKG_DIR && ./venv/bin/pip install -e '.[dev]'" >&2
  fi
fi

run install -m 644 "$UNIT" "$TARGET"
# As the service user, so every directory it creates is theirs.
run runuser -u "$SERVICE_USER" -- mkdir -p "$DATA_DIR"
run systemctl daemon-reload
run systemctl enable df2-pi.service

echo
if [[ $DRY_RUN -eq 1 ]]; then
  echo "Dry run: nothing was changed."
  exit 0
fi
echo "Installed. It starts on the next boot; to start (or pick up a changed unit) now:"
echo "  sudo systemctl restart df2-pi"
echo "  journalctl -u df2-pi -f"
