import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


def _slug_for_path(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-+=" else "_" for c in str(s))[:120] or "unnamed"


def _safe_write_text(path: str, text: Any) -> None:
    """Persist text-like content, gracefully handling non-string inputs."""

    if not isinstance(text, str):
        try:
            text = json.dumps(text, indent=2)
        except Exception:
            text = str(text)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


@dataclass
class DebugConfig:
    enabled: bool = False
    outdir: str = "debug_runs"
    keep_prompts: bool = True
    keep_theory_text: bool = True
    keep_classifier_code: bool = True
    keep_report: bool = True
    max_error_snippet: int = 160  # chars to show in tqdm postfix

    def to_dict(self) -> Dict[str, Any]:
        # asdict handles nested dataclasses too, if you ever add them
        return asdict(self)



def _persist_debug_bundle(
    *,
    debug: DebugConfig,
    model: str,
    level: str,
    instance: str,
    iteration: int,
    prompt: Optional[str] = None,
    theory_text: Optional[str] = None,
    classifier_code: Optional[Any] = None,
    report: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> None:
    if not debug.enabled:
        return
    task_dir = os.path.join(
        debug.outdir,
        _slug_for_path(model),
        _slug_for_path(level),
        _slug_for_path(instance),
        f"it_{iteration}",
    )
    if prompt and debug.keep_prompts:
        _safe_write_text(os.path.join(task_dir, "prompt_theory.txt"), prompt)
    if theory_text and debug.keep_theory_text:
        _safe_write_text(os.path.join(task_dir, "llm_theory.txt"), theory_text)
    if classifier_code and debug.keep_classifier_code:
        _safe_write_text(os.path.join(task_dir, "compiled_classifier.py"), classifier_code)
    if report and debug.keep_report:
        os.makedirs(task_dir, exist_ok=True)
        with open(os.path.join(task_dir, "report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    if error:
        os.makedirs(task_dir, exist_ok=True)
        with open(os.path.join(task_dir, "error.json"), "w", encoding="utf-8") as f:
            json.dump(error, f, indent=2)


