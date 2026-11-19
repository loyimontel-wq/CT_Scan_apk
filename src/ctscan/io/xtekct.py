from pathlib import Path


def _convert_value(value: str) -> int | float | str:
    value = value.strip()
    if not value:
        return ""
    try:   
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_xtekct(path: str | Path) -> dict:

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError (f" Xtekct file not found:{path}")

    cfg = {}

    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line or line.startswith(("#",";","[")):
                continue
            if "=" not in line:
                continue

            key, value = line.split("=", 1)

            key = key.strip()
            if not key:
                raise ValueError(f"Empty configuration key at line {line_number}")
            if key in cfg:
                raise ValueError(f"Duplicate configuration key '{key}'" f"at line {line_number }")

            cfg[key] = _convert_value(value)

        if not cfg:
            raise ValueError(f"No configuration entries found in: {path}")

        return cfg