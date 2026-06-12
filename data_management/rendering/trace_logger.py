from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, List


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "unnamed"


@dataclass
class RenderingTraceLogger:
    """Logger for rendering traces that saves as a formatted JSON list.

    This version rewrites the JSON file on every entry to ensure the file
    is always a valid, human-readable list of dictionaries.
    """

    dataset_root: str
    instance_name: str
    level: Optional[str] = None
    log_dir: Optional[str] = None
    extra_metadata: Optional[Dict[str, Any]] = None
    log_path: str = field(init=False)

    def __post_init__(self) -> None:
        base_dir = self.log_dir or os.path.join(
            os.path.abspath(self.dataset_root), "renders", "traces"
        )
        if self.level:
            base_dir = os.path.join(base_dir, self.level)

        os.makedirs(base_dir, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_name = _slug(self.instance_name)
        # Changed extension to .json
        self.log_path = os.path.join(base_dir, f"{safe_name}_{timestamp}.json")

        self.log_stage(
            "logger_initialized",
            dataset_root=os.path.abspath(self.dataset_root),
            level=self.level,
            **(self.extra_metadata or {}),
        )

    def _update_log(self, payload: Dict[str, Any]) -> None:
        """Reads the existing list, appends the new record, and rewrites the file."""
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **payload,
        }

        # 1. Load existing data if file exists
        data: List[Dict[str, Any]] = []
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                # Fallback if file is corrupted or empty
                data = []

        # 2. Add new record
        data.append(record)

        # 3. Write the entire list back to disk with indentation
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def log_stage(self, stage: str, message: Optional[str] = None, **details: Any) -> None:
        payload: Dict[str, Any] = {"stage": stage}
        if message:
            payload["message"] = message
        if details:
            payload["details"] = details
        self._update_log(payload)

    def log_llm_interaction(
        self,
        *,
        stage: str,
        prompt: str,
        response: Optional[str],
        model: Optional[str] = None,
        **metadata: Any,
    ) -> None:
        self._update_log(
            {
                "stage": stage,
                "llm_model": model,
                "prompt": prompt,
                "response": response,
                "metadata": metadata or None,
            }
        )

    def log_error(self, stage: str, error: Exception | str, **context: Any) -> None:
        message = str(error)
        payload: Dict[str, Any] = {
            "stage": stage,
            "error": message,
        }
        if isinstance(error, Exception):
            payload["error_type"] = type(error).__name__
        if context:
            payload["context"] = context
        self._update_log(payload)


__all__ = ["RenderingTraceLogger"]