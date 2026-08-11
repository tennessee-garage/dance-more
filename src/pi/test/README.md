Automated unit tests for `df2_pi`, run with pytest. These exercise pure
logic (CRC, frame encode/decode) and never touch real hardware.

```bash
pytest
```

For scripts that drive real hardware (a Pi connected to a row controller
and/or tile over an actual RS-485 link), see [integration/](integration/).
