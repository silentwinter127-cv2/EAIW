#!/usr/bin/env python3
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.engine import EAIWEngine


def main():
    raw = sys.stdin.read().strip()
    if not raw:
        sys.stderr.write("Empty JSON payload received on stdin\n")
        sys.exit(1)
    payload = json.loads(raw)
    engine = EAIWEngine(ROOT)
    result = engine.execute_request(payload)
    sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
