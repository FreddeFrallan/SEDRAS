from __future__ import annotations

"""Utility entrypoint for rendering dataset instances to disk files.

This module consolidates rendering to different file formats so that callers
(especially the data creation pipeline) only need to depend on a single
interface. Use :class:`RenderingMode` to select the desired output type and
call :func:`render_dataset_instance_to_file` to dispatch to the underlying
renderer.
"""

import inspect
import json
import os
import random
import re
import shutil
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from data_management.rendering.errors import InvalidTemplateError
from data_management.rendering.pdf import render_dataset_instance_to_pdf
from data_management.rendering.xlsx import render_dataset_instance_to_xlsx
from data_management.rendering.multi_format_pipeline import convert_xlsx_to_pdf
from data_management.rendering.trace_logger import RenderingTraceLogger
from inference.model_wrappers.llm_wrapper import LLMModel


class RenderingMode(str, Enum):
    """Supported rendering targets."""

    PDF = "PDF"
    XLSX = "XLSX"
    PPT = "PPT"
    RANDOM = "RANDOM"
    MULTIPLE_FILES = "MULTIPLE_FILES"


class RawRespnseTracker:

    def __init__(self):
        self.response = None

def get_random_rendering_mode() -> RenderingMode:
    return random.choice([
        RenderingMode.PDF,
        RenderingMode.XLSX,
    ])

def _normalize_rendering_mode(mode: Union[RenderingMode, str]) -> RenderingMode:
    if isinstance(mode, RenderingMode):
        return mode
    try:
        return RenderingMode(str(mode).upper())
    except ValueError as exc:  # pragma: no cover - defensive path
        raise ValueError(f"Unsupported rendering mode: {mode}") from exc


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "unnamed"


def _build_final_pdf_path(
    *,
    dataset_root: str,
    level: Optional[str],
    instance_name: str,
    source_type: str,
) -> str:
    base_dir = os.path.join(dataset_root, "renders", "files")
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [_slug(instance_name)]
    if level:
        parts.append(_slug(level))
    parts.append(timestamp)
    if source_type:
        parts.append(source_type.lower())
    filename = "_".join(parts) + ".pdf"
    return os.path.join(base_dir, filename)


def _load_instance_data(
    dataset_root: str, instance_name: str, level: Optional[str]
) -> Tuple[str, Dict[str, Any]]:
    instances_root = os.path.join(dataset_root, "instances")
    if not os.path.isdir(instances_root):
        raise FileNotFoundError(f"No 'instances' folder in {dataset_root}")

    candidates: List[str] = []
    slugged = _slug(instance_name)

    def _maybe_add(path: str) -> None:
        if os.path.isfile(path):
            candidates.append(path)

    if level:
        _maybe_add(os.path.join(instances_root, level, f"{instance_name}.json"))
        _maybe_add(os.path.join(instances_root, level, f"{slugged}.json"))

    for root, _, files in os.walk(instances_root):
        for filename in files:
            if filename in (f"{instance_name}.json", f"{slugged}.json"):
                _maybe_add(os.path.join(root, filename))

    if not candidates:
        raise FileNotFoundError(
            f"Could not find instance '{instance_name}' under {instances_root}"
        )

    path = candidates[0]
    with open(path, "r", encoding="utf-8") as f:
        return path, json.load(f)


def _random_bucketize_samples(
    samples: List[Dict[str, Any]], num_files: int
) -> List[Tuple[List[Dict[str, Any]], List[int]]]:
    if num_files < 1:
        raise ValueError("num_files must be at least 1")
    if num_files > len(samples):
        raise ValueError(
            "num_files cannot exceed the number of available samples when rendering multiple files"
        )

    indices = list(range(len(samples)))
    random.shuffle(indices)

    buckets: List[List[Dict[str, Any]]] = [[] for _ in range(num_files)]
    bucket_indices: List[List[int]] = [[] for _ in range(num_files)]

    for bucket_idx, sample_idx in enumerate(indices[:num_files]):
        buckets[bucket_idx].append(samples[sample_idx])
        bucket_indices[bucket_idx].append(sample_idx)

    for sample_idx in indices[num_files:]:
        chosen = random.randrange(num_files)
        buckets[chosen].append(samples[sample_idx])
        bucket_indices[chosen].append(sample_idx)

    return list(zip(buckets, bucket_indices))


def _convert_intermediate_file(
    *,
    converter_fn,
    source_path: str,
    dataset_root: str,
    level: Optional[str],
    instance_name: str,
    source_type: str,
) -> str:
    pdf_path = _build_final_pdf_path(
        dataset_root=dataset_root,
        level=level,
        instance_name=instance_name,
        source_type=source_type,
    )
    converter_fn(source_path, pdf_path)

    dest_base = os.path.splitext(pdf_path)[0]
    manifest_src = None
    if source_type.lower() == "ppt":
        manifest_src = os.path.splitext(source_path)[0] + "_blocks.json"
        manifest_dest = dest_base + "_blocks.json"
    elif source_type.lower() == "xlsx":
        manifest_src = os.path.splitext(source_path)[0] + "_layout.json"
        manifest_dest = dest_base + "_layout.json"
    else:
        manifest_dest = None

    if manifest_src and os.path.isfile(manifest_src):
        try:
            shutil.move(manifest_src, manifest_dest)
        except Exception:
            shutil.copy(manifest_src, manifest_dest)
    return pdf_path


def render_dataset_instance_to_file(
    *,
    dataset_root: str,
    instance_name: str,
    level: Optional[str] = None,
    rendering_mode: Union[RenderingMode, str] = RenderingMode.PDF,
    llm_model: Optional[LLMModel] = LLMModel.GEMINI_3_PRO,
    instance_title: Optional[str] = None,
    num_files: int = 1,
    maximum_num_samples: Optional[int] = None,
    max_retries: int = 3,
    trace_logging: bool = True,
    trace_log_dir: Optional[str] = None,
) -> Union[str, List[str]]:
    """Render a dataset instance to a file based on ``rendering_mode``.

    Args:
        dataset_root: Path to the dataset root folder.
        instance_name: Name of the dataset instance to render.
        level: Optional representation level name used for namespacing outputs.
        rendering_mode: Target output format.
        prompt_md_path: Prompt path forwarded to the PDF renderer.
        llm_model: LLM model forwarded to the PDF renderer.
        instance_title: Optional title override for the rendered output.
        num_files: Number of files to create when using ``MULTIPLE_FILES`` rendering.
        maximum_num_samples: Optional maximum number of samples to render.
        max_retries: Maximum attempts when the LLM returns an invalid template.
        trace_logging: Whether to persist rendering traces to disk.
        trace_log_dir: Optional override for where trace logs are written.

    Returns:
        Path to the rendered file or list of paths when rendering multiple files.
    """

    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")

    if maximum_num_samples is not None and maximum_num_samples < 1:
        raise ValueError("maximum_num_samples must be at least 1 when provided")

    print(f"[INFO] Rendering instance '{instance_name}' to file(s) with mode '{rendering_mode}'")
    rendering_mode = RenderingMode.PDF

    if rendering_mode == rendering_mode.RANDOM:
        rendering_mode = get_random_rendering_mode()

    trace_logger = RenderingTraceLogger(
        dataset_root=dataset_root,
        instance_name=instance_name,
        level=level,
        log_dir=trace_log_dir,
    ) if trace_logging else None
    raw_response_tracker = RawRespnseTracker()

    instance_data: Optional[Dict[str, Any]] = None
    samples: Optional[List[Dict[str, Any]]] = None
    if maximum_num_samples is not None:
        _, instance_data = _load_instance_data(dataset_root, instance_name, level)
        samples = instance_data.get("dataset", [])[:maximum_num_samples]
        instance_data = {**instance_data, "dataset": samples}

    def _render_with_retries(render_fn, *, description: str, **kwargs):
        last_exc: Optional[Exception] = None
        retry_kwargs = dict(kwargs)

        try:
            signature = inspect.signature(render_fn)
            accepts_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in signature.parameters.values()
            )
            accepted_params = set(signature.parameters.keys())
        except (TypeError, ValueError):
            accepts_kwargs = True
            accepted_params = set()

        def _maybe_add_retry_param(name: str, value: Any) -> None:
            if accepts_kwargs or name in accepted_params:
                retry_kwargs[name] = value

        for attempt in range(1, max_retries + 1):
            try:
                return render_fn(**retry_kwargs)
            except InvalidTemplateError as exc:
                last_exc = exc
                trace_logger.log_error(
                    stage="invalid_template",
                    message=f"{description} attempt {attempt} failed due to invalid template.",
                    error=str(exc),
                )
                if attempt >= max_retries:
                    raise
                _maybe_add_retry_param("previous_response", getattr(exc, "llm_response", None))
                _maybe_add_retry_param("error_feedback", str(exc))
                print(
                    f"[WARN] {description} attempt {attempt} failed due to invalid template; retrying with feedback..."
                )
            except Exception as exc:  # For safety, catch all exceptions from the renderer
                last_exc = exc
                trace_logger.log_error(
                    stage="render_error",
                    message=f"{description} attempt {attempt} failed due to error.",
                    error=str(exc),
                )
                if attempt >= max_retries:
                    raise
                _maybe_add_retry_param("previous_response", raw_response_tracker.response)
                _maybe_add_retry_param("error_feedback", str(exc))

                print(
                    f"[WARN] {description} attempt {attempt} failed; retrying..."
                )
                print(f"Error details: {exc}")

        if last_exc:
            raise last_exc

    mode = _normalize_rendering_mode(rendering_mode)

    if trace_logger:
        trace_logger.log_stage(
            "render_start",
            rendering_mode=str(mode),
            message="Rendering dataset instance to file",
        )

    if mode is RenderingMode.PDF:
        return _render_with_retries(
            render_dataset_instance_to_pdf,
            description="PDF render",
            dataset_root=dataset_root,
            instance_name=instance_name,
            level=level,
            llm_model=llm_model,
            instance_title=instance_title,
            instance_data=instance_data,
            samples=samples,
            trace_logger=trace_logger,
            raw_response_tracker=raw_response_tracker,
        )

    if mode is RenderingMode.PPT:
        raise ValueError("PPT rendering is temporarily disabled.")

    if mode is RenderingMode.XLSX:
        xlsx_path = _render_with_retries(
            render_dataset_instance_to_xlsx,
            description="XLSX render",
            dataset_root=dataset_root,
            instance_name=instance_name,
            level=level,
            instance_title=instance_title,
            instance_data=instance_data,
            samples=samples,
            trace_logger=trace_logger,
        )
        return _convert_intermediate_file(
            converter_fn=convert_xlsx_to_pdf,
            source_path=xlsx_path,
            dataset_root=dataset_root,
            level=level,
            instance_name=instance_name,
            source_type="xlsx",
        )

    if mode is RenderingMode.MULTIPLE_FILES:
        if instance_data is None or samples is None:
            _, instance_data = _load_instance_data(dataset_root, instance_name, level)
            samples = instance_data.get("dataset", [])

        buckets = _random_bucketize_samples(samples, num_files)
        outputs: List[str] = []
        covered_indices: Set[int] = set()

        for bucket_idx, (bucket_samples, bucket_source_indices) in enumerate(buckets):
            covered_indices.update(bucket_source_indices)
            part_mode = random.choice([RenderingMode.PDF, RenderingMode.XLSX])
            part_title = (
                instance_title
                or instance_data.get("name")
                or instance_name
                or "Dataset Instance"
            )
            part_title = f"{part_title} (Part {bucket_idx + 1}/{num_files})"

            bucket_instance_name = f"{instance_name}_part_{bucket_idx + 1}"
            bucket_instance_data = {**instance_data, "dataset": bucket_samples}

            part_logger = (
                RenderingTraceLogger(
                    dataset_root=dataset_root,
                    instance_name=bucket_instance_name,
                    level=level,
                    log_dir=trace_log_dir,
                )
                if trace_logging
                else None
            )
            if part_logger:
                part_logger.log_stage(
                    "render_start",
                    rendering_mode=str(part_mode),
                    message=f"Rendering bucket {bucket_idx + 1}/{num_files}",
                )

            if part_mode is RenderingMode.PDF:
                output_path = _render_with_retries(
                    render_dataset_instance_to_pdf,
                    description=f"PDF render for part {bucket_idx + 1}",
                    dataset_root=dataset_root,
                    instance_name=bucket_instance_name,
                    level=level,
                    llm_model=llm_model,
                    instance_title=part_title,
                    instance_data=bucket_instance_data,
                    samples=bucket_samples,
                    trace_logger=part_logger,
                    raw_response_tracker=raw_response_tracker,
                )
                outputs.append(output_path)
                continue

            if part_mode is RenderingMode.XLSX:
                interim_path = _render_with_retries(
                    render_dataset_instance_to_xlsx,
                    description=f"XLSX render for part {bucket_idx + 1}",
                    dataset_root=dataset_root,
                    instance_name=bucket_instance_name,
                    level=level,
                    instance_title=part_title,
                    instance_data=bucket_instance_data,
                    samples=bucket_samples,
                    trace_logger=part_logger,
                )
                final_pdf = _convert_intermediate_file(
                    converter_fn=convert_xlsx_to_pdf,
                    source_path=interim_path,
                    dataset_root=dataset_root,
                    level=level,
                    instance_name=bucket_instance_name,
                    source_type="xlsx",
                )
                outputs.append(final_pdf)

        missing = set(range(len(samples))) - covered_indices
        if missing:
            raise ValueError(f"Multiple-file rendering missing source indices: {sorted(missing)}")

        return outputs

    raise ValueError(f"Unsupported rendering mode: {rendering_mode}")


__all__ = ["RenderingMode", "render_dataset_instance_to_file"]
