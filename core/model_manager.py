"""
Model Registry & Subprocess Isolation Dispatcher.
"""
from pathlib import Path
from typing import Dict, Any, List
import json
import subprocess
import sys


class ModelManager:
    def __init__(self, registry_path: str | Path):
        self.registry_path = Path(registry_path)
        data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        self.models: Dict[str, Dict[str, Any]] = {m["model_id"]: m for m in data.get("models", [])}

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        if model_id not in self.models:
            raise KeyError(f"Unknown model_id '{model_id}'. Available: {list(self.models.keys())}")
        return self.models[model_id]

    def list_models(self, laboratory: str | None = None) -> List[Dict[str, Any]]:
        items = list(self.models.values())
        if laboratory:
            items = [m for m in items if m.get("laboratory") == laboratory]
        return items

    def run_isolated_worker(self, payload: Dict[str, Any], worker_script: str | Path) -> Dict[str, Any]:
        proc = subprocess.run(
            [sys.executable, str(worker_script)],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            check=True,
        )
        return json.loads(proc.stdout.strip().splitlines()[-1])
