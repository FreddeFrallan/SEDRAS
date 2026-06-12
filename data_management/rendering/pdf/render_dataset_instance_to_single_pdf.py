#!/usr/bin/env python3
from __future__ import annotations

"""
Render a single DatasetInstance (all its samples) into one PDF, using the
LLM-driven template system from `generate_pdf_template.PdfTemplate`.

If you DON'T pass an explicit `template_spec`, this function will invoke an LLM
(to design a JSON layout spec) and then render that.

Usage (programmatic):
    render_dataset_instance_to_pdf(
        dataset_root="/path/to/dataset_root",
        instance_name="My Instance",
        out_pdf="/tmp/out.pdf",
        # Either pass an existing spec ...
        template_spec=my_spec,
        # ...or let an LLM design one by passing an LLM wrapper or model enum:
        llm_model=LLMModel.GPT_5,            # or provide `llm` directly
        style_hint="Concise, academic style",
    )

CLI example (minimal wrapper at bottom):
    python render_dataset_instance_to_single_pdf.py \
        --dataset /path \
        --instance "My Instance" \
        --out /tmp/out.pdf \
        --model gpt-5 \
        --style "Short intro, label table, per-sample paragraphs"

Dependencies:
  - reportlab
  - generate_pdf_template.py in your PYTHONPATH
  - llm_wrapper.py (for LLM wrappers)
"""

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# --- Import the new template engine ---
# from data_management.rendering.pdf.programmatic_PDF_template_generation import ProgrammaticPdfEngine
from data_management.rendering.pdf.new_programmatic_PDF_template_generation import ProgrammaticPdfEngine
from data_management.rendering.trace_logger import RenderingTraceLogger

# --- Optional LLM plumbing ---

from inference.model_wrappers.llm_wrapper import get_llm_wrapper, LLMModel, LLMWrapper

# ---------- Helpers (kept local) ----------
def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "unnamed"


def _find_instance_json(dataset_root: str, instance_name: str, level: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """
    Return (path, parsed_json) for the desired instance.

    Tries:
      1) instances/<level>/<name or slug>.json (if level provided)
      2) instances/*/<name or slug>.json
      3) instances/<name or slug>.json (legacy)
    """
    instances_root = os.path.join(dataset_root, "instances")
    if not os.path.isdir(instances_root):
        raise FileNotFoundError(f"No 'instances' folder in {dataset_root}")

    candidates: List[str] = []
    name_slug = _slug(instance_name)

    def _try(path: str):
        if os.path.isfile(path):
            candidates.append(path)

    if level:
        lvl_dir = os.path.join(instances_root, level)
        _try(os.path.join(lvl_dir, f"{instance_name}.json"))
        _try(os.path.join(lvl_dir, f"{name_slug}.json"))

    for maybe_level in os.listdir(instances_root):
        lvl_dir = os.path.join(instances_root, maybe_level)
        if not os.path.isdir(lvl_dir):
            continue
        _try(os.path.join(lvl_dir, f"{instance_name}.json"))
        _try(os.path.join(lvl_dir, f"{name_slug}.json"))

    _try(os.path.join(instances_root, f"{instance_name}.json"))
    _try(os.path.join(instances_root, f"{name_slug}.json"))

    seen, uniq = set(), []
    for p in candidates:
        if p not in seen:
            seen.add(p); uniq.append(p)

    if not uniq:
        raise FileNotFoundError(
            f"Could not find instance '{instance_name}'. Tried exact and slug across levels and legacy layout."
        )

    path = uniq[0]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return path, data


def render_dataset_instance_to_pdf(
    dataset_root: str,
    instance_name: str,
    # out_pdf: Optional[str] = None,
    *,
    level: Optional[str] = None,
    instance_title: Optional[str] = None,
    # LLM control
    llm: Optional["LLMWrapper"] = None,
    llm_model: Optional[str] = LLMModel.GEMINI_3_PRO,  # accepts LLMModel or raw string (e.g., "gpt-5")
    # Prompt & style
    theme_hint: Optional[str] = None,
    max_preview_samples: int = 1,
    # PDF look
    page_size: str = "A4",
    font_path: Optional[str] = None,
    margins: Tuple[int, int, int, int] = (36, 36, 42, 36),  # L, R, T, B
    print_layout: bool = True,
    instance_data: Optional[Dict[str, Any]] = None,
    samples: Optional[List[Dict[str, Any]]] = None,
    error_feedback: Optional[str] = None,
    previous_response: Optional[str] = None,
    trace_logger: Optional[RenderingTraceLogger] = None,
    raw_response_tracker=None,
) -> str:
    """
    Render a DatasetInstance to a single PDF using the new ProgrammaticPdfEngine.

    This path *always* drives the LLM with the external prompt at `prompt_md_path`
    (e.g., prompts/programmatic_template_prompt.md), which should contain the full
    instructions you curated. The LLM returns a JSON layout + optional block functions,
    which the engine compiles and executes.

    Returns:
        str: Path to the written PDF.
    """

    # ---- Load instance JSON (reuse helper) ----
    dataset_root = os.path.abspath(dataset_root)
    inst = instance_data
    if inst is None:
        _, inst = _find_instance_json(dataset_root, instance_name, level)

    # ---- Extract data ----
    name = inst.get("name") or instance_name
    title = instance_title or name
    samples = samples if samples is not None else inst.get("dataset", [])

    # ---- Decide output path ----
    renders_dir = os.path.join(dataset_root, "renders", "files")
    os.makedirs(renders_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = _slug(name)
    parts = [safe_name]
    if level:
        parts.append(_slug(level))
    parts.append(timestamp)
    out_pdf = os.path.join(renders_dir, f"{'_'.join(parts)}.pdf")
    out_tempalte = out_pdf.replace(".pdf", "_template.json")
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)

    if trace_logger:
        trace_logger.log_stage(
            "pdf_render_setup",
            message="Prepared PDF render paths",
            output_path=out_pdf,
            template_path=out_tempalte,
            samples=len(samples),
        )

    # # ---- Prepare default prompt path ----
    # prompt_md_path = prompt_md_path or os.path.join(
    #     os.path.dirname(__file__), "prompts", "programmatic_template_prompt.md"
    # )

    # ---- Prepare LLM wrapper if only model id was given ----
    effective_llm = llm
    if effective_llm is None:
        if llm_model is None:
            raise ValueError(
                "To auto-generate a layout with ProgrammaticPdfEngine, pass `llm` or `llm_model`."
            )
        model_enum = llm_model
        if LLMModel is not None and not isinstance(llm_model, LLMModel):
            try:
                model_enum = next(m for m in LLMModel if m.value == str(llm_model))  # type: ignore
            except StopIteration:
                model_enum = llm_model  # type: ignore
        effective_llm = get_llm_wrapper(model_enum)  # type: ignore

    if trace_logger:
        trace_logger.log_stage(
            "pdf_llm_resolved",
            message="Resolved PDF LLM model",
            llm_model=str(llm_model) if llm_model is not None else None,
            # prompt_md_path=prompt_md_path,
        )

    # ---- Build engine (uses your external prompt) ----
    left, right, top, bottom = margins
    engine = ProgrammaticPdfEngine(
        font_path=font_path,
        page_size=page_size,
        left_margin=left,
        right_margin=right,
        top_margin=top,
        bottom_margin=bottom,
        # default_prompt_md=prompt_md_path,   # ensure the external prompt is used by default
        raw_response_tracker=raw_response_tracker,
    )

    # ---- Render via engine (always uses external prompt) ----
    # We pass `layout=None` to force LLM generation using the provided prompt file.
    # `theme_hint` can be provided explicitly; if not, we fall back to instance title.
    _, layout = engine.render(
        samples=samples,
        out_pdf=out_pdf,
        layout=None,
        theme_hint=theme_hint or title,
        max_preview_samples=max_preview_samples,
        print_layout=print_layout,
        # layout_prompt_md_path=prompt_md_path,  # ALWAYS use the external prompt
        error_feedback=error_feedback,
        previous_response=previous_response,
        trace_logger=trace_logger,
    )

    # ---- Save the layout JSON for reference ----
    with open(out_tempalte, "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2)
    print(f"✅ Saved generated template JSON to {out_tempalte}")
    if trace_logger:
        trace_logger.log_stage(
            "pdf_layout_written",
            message="Saved generated PDF layout",
            template_path=out_tempalte,
        )

    return out_pdf
