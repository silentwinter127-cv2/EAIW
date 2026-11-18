"""
Node-Based Workflow Pipeline Orchestrator.
Supports chaining:
INPUT IMAGE -> PREPROCESSING -> LAB NODE(S) (DeepDream / Inpainting / GAN / CLIP) -> POST-PROCESSING -> OUTPUT & ARCHIVE
"""
import json
from pathlib import Path
from typing import Dict, Any, List
from core.engine import EAIWEngine


class WorkflowPipeline:
    def __init__(self, engine: EAIWEngine):
        self.engine = engine

    def run_workflow(self, workflow_spec: Dict[str, Any], work_dir: str | Path) -> Dict[str, Any]:
        work_path = Path(work_dir)
        work_path.mkdir(parents=True, exist_ok=True)
        nodes: List[Dict[str, Any]] = workflow_spec.get("nodes", [])
        current_image = workflow_spec.get("initial_input")
        current_mask = workflow_spec.get("initial_mask")
        node_outputs: List[Dict[str, Any]] = []

        for idx, node in enumerate(nodes, start=1):
            node_id = node.get("node_id", f"node_{idx:02d}")
            out_path = work_path / f"{idx:02d}_{node_id}_{node['model_id']}.png"
            req = {
                "model_id": node["model_id"],
                "execution_mode": node.get("execution_mode", "Simulation"),
                "operation": node.get("operation"),
                "input": node.get("input", current_image),
                "mask": node.get("mask", current_mask),
                "parameters": node.get("parameters", {}),
                "output": str(out_path),
            }
            res = self.engine.execute_request(req)
            current_image = res["output"]
            node_outputs.append({"node_id": node_id, **res})

        summary = {
            "workflow_id": workflow_spec.get("workflow_id", "custom_workflow"),
            "name": workflow_spec.get("name", "Custom Workflow"),
            "final_output": current_image,
            "node_results": node_outputs,
        }
        (work_path / "workflow_execution_record.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return summary
