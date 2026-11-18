"""
Batch Task Queue & Priority Job Scheduler for EAIW Workstation.
"""
from collections import deque
from typing import Dict, Any, Callable, Optional, List
from core.engine import EAIWEngine


class JobScheduler:
    def __init__(self, engine: EAIWEngine):
        self.engine = engine
        self.queue: deque = deque()
        self.history: List[Dict[str, Any]] = []

    def submit(self, request: Dict[str, Any], priority: str = "P0") -> int:
        job_id = len(self.queue) + len(self.history) + 1
        item = {"job_id": job_id, "priority": priority, "request": request, "status": "QUEUED"}
        if priority == "P0":
            self.queue.appendleft(item)
        else:
            self.queue.append(item)
        return job_id

    def run_all(self, on_progress: Optional[Callable[[Dict[str, Any]], None]] = None) -> List[Dict[str, Any]]:
        completed = []
        while self.queue:
            job = self.queue.popleft()
            job["status"] = "RUNNING"
            if on_progress:
                on_progress(job)
            result = self.engine.execute_request(job["request"])
            job["status"] = "COMPLETED"
            job["result"] = result
            self.history.append(job)
            completed.append(job)
            if on_progress:
                on_progress(job)
        return completed
