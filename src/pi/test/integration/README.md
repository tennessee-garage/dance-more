Hardware-in-the-loop test scripts: standalone Python scripts run manually
on a real Raspberry Pi with a Pi hat attached to a live Row Bus (and, through
it, at least one row controller and tile). Unlike `test/*.py`, these are not
collected by pytest and are not run in CI - they need physical hardware and
are meant to be invoked directly, e.g.:

```bash
python3 test/integration/<script>.py
```

This mirrors the `test/integration/` convention used by
[src/row](../../../row/test/integration) and
[src/tile](../../../tile/test/integration) for scripts that drive their
respective buses against real boards.
