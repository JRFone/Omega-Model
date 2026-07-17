from __future__ import annotations

import json
from pathlib import Path

from omega_self_check import run_self_check


def main() -> int:
    result = run_self_check(full_tests=True)
    output = Path(__file__).resolve().parent / "reports" / "release_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["software_status"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
