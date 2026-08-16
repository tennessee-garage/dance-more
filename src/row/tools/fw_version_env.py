"""PlatformIO pre-script: injects FW_GIT_SHA / FW_DIRTY build flags.

Consumed by lib/row_core/fw_version_info.cpp (see include/fw_version.h for
ROW_FW_VERSION, the hand-bumped half of the identity this completes).

FW_GIT_SHA is HEAD's commit hash, truncated to a fixed 8 hex chars - not
`git rev-parse --short`, whose length varies with repo size and would
silently change what fits in the 32-bit wire field. FW_DIRTY is 1 if
src/row or src/common (the code that actually lands in this image) has
uncommitted changes; a dirty doc or an open editor elsewhere in the repo
does not count, so the flag stays meaningful.

Must not break a build outside a git checkout (e.g. a source tarball, or
`pio test -e native` on a CI runner without .git): falls back to
FW_GIT_SHA=0, FW_DIRTY=1 whenever git is unavailable or this isn't a repo.
"""

import subprocess

Import("env")  # noqa: F821 - injected by PlatformIO's SCons environment


def _git(args, cwd):
    try:
        out = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


project_dir = env.subst("$PROJECT_DIR")
repo_root = _git(["rev-parse", "--show-toplevel"], project_dir)
sha = _git(["rev-parse", "HEAD"], project_dir) if repo_root else None

if sha:
    git_sha = "0x" + sha[:8]
    dirty_status = _git(
        ["status", "--porcelain", "--", "src/row", "src/common"], repo_root
    )
    dirty = 1 if dirty_status else 0
else:
    git_sha = "0x0"
    dirty = 1

env.Append(CPPDEFINES=[("FW_GIT_SHA", git_sha), ("FW_DIRTY", dirty)])
