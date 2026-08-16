#!/usr/bin/env python3
"""Fail a PR that changes firmware without bumping its version constant.

See src/row/include/fw_version.h and src/tile/include/fw_version.h - each
holds a hand-bumped integer (ROW_FW_VERSION / TILE_FW_VERSION) that #55
requires be incremented in the PR that changes that side. A PR touching the
shared wire protocol (src/common/tile_bus_protocol/) must bump both, since
that code ships in both images.

"Non-doc" is approximated as "not a .md file" - a PR that only edits docs
under src/row or src/tile doesn't need a version bump.
"""

from __future__ import annotations

import re
import subprocess
import sys

ROW_FILE = "src/row/include/fw_version.h"
TILE_FILE = "src/tile/include/fw_version.h"


def changed_files(base_ref: str) -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def touches(files: list[str], prefix: str) -> bool:
    return any(f.startswith(prefix) and not f.endswith(".md") for f in files)


def read_constant(ref: str, path: str, name: str) -> int | None:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"], capture_output=True, text=True
    )
    if result.returncode != 0:
        return None  # file didn't exist at ref - treat as version 0
    match = re.search(rf"{name}\s*=\s*(\d+)", result.stdout)
    if not match:
        return None
    return int(match.group(1))


def check(base_ref: str, files: list[str], prefix: str, path: str, name: str) -> str | None:
    if not touches(files, prefix):
        return None
    old = read_constant(base_ref, path, name) or 0
    new = read_constant("HEAD", path, name)
    if new is None:
        return f"{path} is missing {name} (or it's unparseable) on this branch"
    if new <= old:
        return (
            f"{prefix} changed but {name} did not increase "
            f"({path}: {old} -> {new})"
        )
    return None


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_fw_version_bump.py <base_ref>", file=sys.stderr)
        return 2
    base_ref = sys.argv[1]

    files = changed_files(base_ref)
    errors = []

    common_touched = touches(files, "src/common/tile_bus_protocol/")
    for prefix, path, name in (
        ("src/row/", ROW_FILE, "ROW_FW_VERSION"),
        ("src/tile/", TILE_FILE, "TILE_FW_VERSION"),
    ):
        if common_touched:
            err = check(base_ref, files, "src/common/tile_bus_protocol/", path, name)
        else:
            err = check(base_ref, files, prefix, path, name)
        if err:
            errors.append(err)

    if errors:
        for err in errors:
            print(f"::error::{err}")
        print(
            "\nBump the affected version constant(s) - see "
            "src/row/include/fw_version.h and src/tile/include/fw_version.h.",
            file=sys.stderr,
        )
        return 1

    print("fw version bump check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
