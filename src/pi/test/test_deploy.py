"""deploy/: the systemd unit and its installer (#96)."""

import logging
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from df2_pi import cli

DEPLOY = Path(__file__).resolve().parent.parent / "deploy"
UNIT = DEPLOY / "df2-pi.service"
INSTALL = DEPLOY / "install.sh"


def parse_unit(text: str) -> dict[str, list[tuple[str, str]]]:
    """A unit file as {section: [(key, value), ...]}. Not configparser:
    systemd repeats keys (Environment=) and each one counts."""
    sections: dict[str, list[tuple[str, str]]] = {}
    current = None
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line[0] in "#;":
            continue
        if line.startswith("[") and line.endswith("]"):
            current = sections.setdefault(line[1:-1], [])
            continue
        assert current is not None, f"line {number}: key outside a section"
        key, sep, value = line.partition("=")
        assert sep, f"line {number}: not key=value: {raw!r}"
        current.append((key.strip(), value.strip()))
    return sections


@pytest.fixture(scope="module")
def unit() -> dict[str, list[tuple[str, str]]]:
    return parse_unit(UNIT.read_text())


def one(section: list[tuple[str, str]], key: str) -> str:
    values = [v for k, v in section if k == key]
    assert len(values) == 1, f"{key}= appears {len(values)} times"
    return values[0]


def environment(section: list[tuple[str, str]]) -> dict[str, str]:
    return dict(v.split("=", 1) for k, v in section if k == "Environment")


def test_unit_has_its_sections_and_starts_on_boot(unit):
    assert set(unit) == {"Unit", "Service", "Install"}
    assert one(unit["Install"], "WantedBy") == "multi-user.target"
    # The floor must light with no network; waiting for one could hold it dark.
    assert "network-online.target" not in " ".join(v for _, v in unit["Unit"])


def test_unit_restarts_and_outwaits_the_shutdown_blackout(unit):
    from df2_pi.web.app import SHUTDOWN_TIMEOUT_S

    service = unit["Service"]
    assert one(service, "Restart") == "always"
    assert one(service, "KillSignal") == "SIGTERM"
    # systemd SIGKILLs at TimeoutStopSec: before the blackout lands, the floor
    # would freeze on its last frame.
    assert float(one(service, "TimeoutStopSec")) > SHUTDOWN_TIMEOUT_S


def test_unit_execstart_is_a_valid_serve_command(unit):
    service = unit["Service"]
    argv = shlex.split(one(service, "ExecStart"))
    assert argv[0] == one(service, "WorkingDirectory") + "/venv/bin/df2-pi"
    assert "$DF2_SERVE_ARGS" in argv
    env = environment(service)
    assert env["DF2_SERVE_ARGS"] == ""  # realtime is opt-in, from a drop-in
    assert env["DF2_DB"].startswith("/home/" + one(service, "User") + "/")
    args = cli.build_parser().parse_args([a for a in argv[1:] if not a.startswith("$")])
    assert args.command == "serve" and args.realtime is False
    # What the README's drop-in adds must parse too.
    assert cli.build_parser().parse_args([*argv[1:-1], "--realtime"]).realtime is True


@pytest.mark.skipif(shutil.which("systemd-analyze") is None, reason="systemd-analyze not available")
def test_unit_passes_systemd_analyze_verify(tmp_path):
    # Verified from a copy, so the check doesn't depend on ExecStart existing
    # here: only the unit's syntax and directives are being tested.
    text = UNIT.read_text()
    executable = shlex.split(one(parse_unit(text)["Service"], "ExecStart"))[0]
    text = text.replace(executable, "/bin/true")
    copy = tmp_path / "df2-pi.service"
    copy.write_text(text)
    result = subprocess.run(["systemd-analyze", "verify", str(copy)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "df2-pi.service" not in result.stderr, result.stderr


def run_install(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(INSTALL), *args], capture_output=True, text=True)


def test_install_dry_run_prints_the_plan_and_touches_nothing(unit):
    result = run_install("--dry-run")
    assert result.returncode == 0, result.stderr
    service = unit["Service"]
    data_dir = str(Path(environment(service)["DF2_DB"]).parent)
    commands = [line[2:] for line in result.stdout.splitlines() if line.startswith("+ ")]
    assert commands == [
        f"install -m 644 {UNIT} /etc/systemd/system/df2-pi.service",
        f"runuser -u {one(service, 'User')} -- mkdir -p {data_dir}",
        "systemctl daemon-reload",
        "systemctl enable df2-pi.service",
    ]
    # Every step is safe to repeat, and none of them starts or restarts the floor.
    assert not any("start" in c.split() or "restart" in c.split() for c in commands)


def test_install_is_idempotent():
    assert run_install("--dry-run").stdout == run_install("--dry-run").stdout


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs a non-root user")
def test_install_refuses_without_root():
    result = run_install()
    assert result.returncode == 1
    assert "sudo" in result.stderr


def test_install_rejects_unknown_arguments():
    assert run_install("--now").returncode == 2


@pytest.mark.parametrize(("level", "prefix"), [(logging.DEBUG, "<7>"), (logging.INFO, "<6>"), (logging.WARNING, "<4>"), (logging.ERROR, "<3>")])
def test_journal_formatter_prefixes_the_syslog_priority(level, prefix):
    record = logging.LogRecord("df2_pi.engine.runner", level, __file__, 1, "playing %s", ("chase",), None)
    line = cli._JournalFormatter("%(levelname)s %(name)s: %(message)s").format(record)
    assert line == f"{prefix}{logging.getLevelName(level)} df2_pi.engine.runner: playing chase"
