from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_policy_chunks() -> list[dict[str, Any]]:
    policy_path = Path(__file__).with_name("policies.json")
    return json.loads(policy_path.read_text(encoding="utf-8"))


POLICY_CHUNKS = _load_policy_chunks()
