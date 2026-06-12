from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from evaluation.backbone.task_and_theory import Task, InducedTheory
from evaluation.backbone import static_evaluation

from tqdm import tqdm

from inference.model_wrappers.llm_wrapper import LLMBackend, LLMModel, get_llm_wrapper, UploadedFileHandle
from data_management.dataset import RepresentationLevel, AbstractDataset
from evaluation.backbone.evaluate_on_unseen_samples import evaluate_on_unseen_samples
from data_management.utils import _schema_from_dataset, _infer_variable_order, get_output_label_names
from evaluation.backbone.compile_and_evaluate_textual_theory import compile_and_evaluate_theory
from evaluation.backbone.debugging_utils import DebugConfig, _persist_debug_bundle
from evaluation.backbone.evaluation_initalization import prepare_evaluation_jobs
from interactive.interactive_session import run_interactive_theory_induction_session
from configs.interactive_config import InteractiveConfig, InteractiveMode


_FILE_UPLOAD_CACHE: Dict[Tuple[str, str], UploadedFileHandle] = {}


def _prepare_file_handles(
    llm_wrapper,
    dataset_instance,
    dataset_root: str,
) -> Tuple[List[UploadedFileHandle], List[Dict[str, Any]]]:
    docs = getattr(dataset_instance, "rendered_documents", None) or []
    if not docs:
        return [], []
    if not llm_wrapper.supports_file_upload:
        raise RuntimeError(
            f"Model wrapper '{llm_wrapper.__class__.__name__}' does not support file uploads "
            "but the dataset instance provides rendered documents."
        )

    handles: List[UploadedFileHandle] = []
    metadata: List[Dict[str, Any]] = []

    for doc in docs:
        rel_path = doc.get("path")
        abs_path = doc.get("absolute_path")
        if not abs_path and rel_path:
            abs_path = os.path.join(dataset_root, rel_path)
        if not abs_path or not os.path.isfile(abs_path):
            print(f"⚠️ Skipping missing rendered document for evaluation: {rel_path or abs_path}")
            continue

        model_marker = getattr(llm_wrapper, "model", None) or getattr(llm_wrapper, "model_name", None)
        key = (llm_wrapper.provider_label, model_marker, abs_path)
        handle = _FILE_UPLOAD_CACHE.get(key)
        if handle is None:
            handle = llm_wrapper.upload_file(
                abs_path,
                display_name=os.path.basename(abs_path),
                mime_type=doc.get("mime_type") or "application/pdf",
            )
            _FILE_UPLOAD_CACHE[key] = handle
        handles.append(handle)
        metadata.append(
            {
                "path": abs_path,
                "relative_path": rel_path,
                "provider": handle.provider,
                "file_id": handle.file_id,
                "display_name": handle.display_name,
                "doc_metadata": {k: v for k, v in doc.items() if k != "absolute_path"},
            }
        )

    return handles, metadata


def _run_evaluation_task(
    dataset_jobs: List[Tuple[str, AbstractDataset]],
    task: Task,
    *,
    return_per_sample: bool,
    debug: DebugConfig,
    interactive_config: InteractiveConfig,
    intro_sample_limit = None,
    ) -> Dict[str, Any]:
    """
    Worker for a single evaluation task.

    Responsible for:
      - (optional) running interactive tool-calling session
      - inducing a textual theory
      - compiling & evaluating it
      - evaluating on unseen samples
      - persisting debug bundles
      - returning a standardized result dict (including dataset_idx & dataset_path)
    """
    ds_idx, model_enum, level_enum, instance_name, iteration = task
    dataset_path, abstractDataset = dataset_jobs[ds_idx]

    t0 = time.time()
    model_name = model_enum.name
    level_val = level_enum.value
    session_data: Optional[Dict[str, Any]] = None

    try:
        # Choose backend based on the interactive_config's oracle backend, if present.
        backend = None
        if interactive_config is not None and interactive_config.oracle_config is not None:
            backend = interactive_config.oracle_config.backend

        print(f"Representation level: {level_enum}, initial backend: {backend}")
        if level_enum == RepresentationLevel.FILES and backend != LLMBackend.NATIVE:
            backend = LLMBackend.NATIVE
            print(
                f"Switching backend to native for file-based representation: {model_name}"
            )

        print(f"Backend for model {model_name}: {backend}")

        llm = get_llm_wrapper(model_enum, backend=backend)
        dataset_instance = abstractDataset.dataset_instances[level_enum][instance_name]
        num_output_labels = int(getattr(dataset_instance.udd, "num_output_labels", 2))
        label_names = get_output_label_names(dataset_instance)
        file_handles, file_metadata = _prepare_file_handles(llm, dataset_instance, dataset_path)

        # --- interactive session (tool-calling) ---
        interactive_summary: Optional[Dict[str, Any]] = None
        print(f"Interactive mode: {interactive_config.mode}")
        if interactive_config.mode != InteractiveMode.STATIC:
            try:
                interactive_summary = run_interactive_theory_induction_session(
                    dataset_instance,
                    interactive_config=interactive_config,
                    llm_model=model_enum,
                    max_turns=interactive_config.max_turns,
                    file_handles=file_handles,
                )

                theory_obj = InducedTheory(
                    theory_text=interactive_summary["final_message"],
                    variable_order=_infer_variable_order(dataset_instance),
                    schema=_schema_from_dataset(dataset_instance, _infer_variable_order(dataset_instance)),
                    guidance_prompt=""
                )

                session_data = interactive_summary.get("session_data")
                if file_metadata:
                    interactive_summary.setdefault("file_inputs", file_metadata)

            except Exception as e:
                # Don't fail the whole task; capture a minimal interactive error summary

                _persist_debug_bundle(
                    debug=debug,
                    model=model_name,
                    level=level_val,
                    instance=instance_name,
                    iteration=iteration,
                    prompt=None,
                    theory_text=None,
                    classifier_code=None,
                    report=None,
                    error={
                        "stage": "interactive_session",
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                    },
                )

                raise
        else: # Non-interactive mode
            try:
                theory_obj, static_summary = static_evaluation.induce_textual_theory_from_dataset(
                    dataset_instance,
                    llm,
                    verbose=interactive_config.verbose,
                    max_intro_samples=interactive_config.number_of_intro_samples,
                    file_handles=file_handles,
                )
                interactive_summary = static_summary
                if file_metadata:
                    interactive_summary.setdefault("file_inputs", file_metadata)
            except Exception as e:
                err = {
                    "stage": "induce_theory",
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
                _persist_debug_bundle(
                    debug=debug,
                    model=model_name, level=level_val,
                    instance=instance_name, iteration=iteration,
                    prompt=None, theory_text=None, classifier_code=None,
                    report=None, error=err,
                )
                raise


        try:

            dataset_samples = dataset_instance.samples
            current_num_samples = len(dataset_instance.samples)
            if interactive_config is not None:
                intro_sample_limit = getattr(interactive_config, "number_of_intro_samples", 0) or 0
                if current_num_samples > interactive_config.number_of_intro_samples:
                    dataset_samples = dataset_instance.samples[:intro_sample_limit]

            # Compile and evaluate the already-induced theory
            result, report, classifier_func = compile_and_evaluate_theory(
                theory=theory_obj,
                dataset_samples=dataset_samples,
                llm=llm,
                num_output_labels=num_output_labels,
                label_names=label_names,
                return_per_sample=return_per_sample,
            )

            # Evaluate on unseen samples
            us_result, us_report = evaluate_on_unseen_samples(
                evaluation_func=classifier_func,
                llm=llm,
                dataset_instance=dataset_instance,
                existing_samples=dataset_samples,
                num_output_labels=num_output_labels,
                label_names=label_names,
            )

            # add timing tags compatible with rest of pipeline
            report["theory_time"] = report.get("theory_time")  # may be None

            # Attach interactive session summary (if any)
            if interactive_summary is not None:
                report["interactive_session"] = interactive_summary
                if "usage" in interactive_summary:
                    report["usage"] = interactive_summary["usage"]
                if session_data is None:
                    session_data = interactive_summary.get("session_data")

            # Add unseen samples report under a different key
            report["unseen_samples_evaluation"] = us_report

        except Exception as e:
            err = {
                "stage": "compile_or_evaluate",
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            _persist_debug_bundle(
                debug=debug,
                model=model_name,
                level=level_val,
                instance=instance_name,
                iteration=iteration,
                prompt=theory_obj.guidance_prompt,
                theory_text=theory_obj.theory_text,
                classifier_code=None,
                report=None,
                error=err,
            )
            raise

        # Persist debug artifacts for successful run too (optional)
        _persist_debug_bundle(
            debug=debug,
            model=model_name, level=level_val,
            instance=instance_name, iteration=iteration,
            prompt=theory_obj.guidance_prompt,
            theory_text=theory_obj.theory_text,
            classifier_code=report.get("classifier_code"),
            report=report, error=None,
        )

        t1 = time.time()
        result = {
            "ok": True,
            "dataset_idx": ds_idx,
            "dataset_path": dataset_path,
            "model": model_enum,
            "instance": instance_name,
            "level": level_val,
            "iteration": iteration,
            "report": report,
            "session_data": session_data,
            "timings": {
                "total_seconds": t1 - t0,
                "theory_time": report.get("theory_time"),
                "compilation_time": report.get("compilation_time"),
            },
        }

        return result

    except Exception as e:
        t1 = time.time()
        short = (str(e) or type(e).__name__)[: debug.max_error_snippet]

        # Persist debug bundle for failure
        _persist_debug_bundle(
            debug=debug,
            model=model_name,
            level=level_val,
            instance=instance_name,
            iteration=iteration,
            prompt=None,
            theory_text=None,
            classifier_code=None,
            report=None,
            error={
                "stage": "overall_task_failure",
                "message": str(e),
                "traceback": traceback.format_exc(),
            },
        )


        result = {
            "ok": False,
            "dataset_idx": ds_idx,
            "dataset_path": dataset_path,
            "model": model_enum,
            "instance": instance_name,
            "level": level_val,
            "iteration": iteration,
            "error": short,
            "session_data": session_data,
            "timings": {"total_seconds": t1 - t0},
        }

        return result


# =======================
# Unified parallel evaluation
# =======================

def evaluate_parallel(
    configs: List[Dict[str, Any]],
    *,
    max_workers: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run induction+compilation+evaluation for all tasks defined in in-memory config dicts.

    Tasks are created for all configs and evaluated in a shared thread pool.
    """
    if not configs:
        raise ValueError("configs must be a non-empty list of config dictionaries.")

    config_jobs = prepare_evaluation_jobs(configs, verbose=verbose)
    task_queue: List[Tuple[int, Task, Optional[float]]] = []

    if not config_jobs:
        if verbose:
            print("⚠️ No evaluation tasks constructed across configs.")
        return {"by_config": {}, "successes": [], "failures": []}

    for job_idx, job in enumerate(config_jobs):
        for task in job["tasks"]:
            task_queue.append((job_idx, task, job["timeout_per_task"]))

    if max_workers is None:
        max_workers = min(8, (os.cpu_count() or 4))

    if verbose:
        total_tasks = sum(len(job["tasks"]) for job in config_jobs)
        print(f"🔹 Launching {total_tasks} tasks across {len(config_jobs)} config(s).")

    successes: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    by_config: Dict[str, Dict[str, Any]] = {}

    for job in config_jobs:
        by_config[job["config_key"]] = {
            "by_dataset": {
                path: {"successes": [], "failures": []} for path, _ in job["dataset_jobs"]
            },
            "successes": [],
            "failures": [],
        }

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {
            ex.submit(
                _run_evaluation_task,
                config_jobs[job_idx]["dataset_jobs"],
                task,
                return_per_sample=config_jobs[job_idx]["return_per_sample"],
                debug=config_jobs[job_idx]["debug"],
                interactive_config=config_jobs[job_idx]["interactive_config"],
            ): (job_idx, task, timeout)
            for job_idx, task, timeout in task_queue
        }

        total = len(future_map)
        progress = tqdm(total=total, desc="Evaluating", ncols=120) if verbose else None

        for fut in as_completed(future_map):
            job_idx, task, timeout = future_map[fut]
            ds_idx, m, lv, inst_name, it = task
            job = config_jobs[job_idx]
            dataset_jobs = job["dataset_jobs"]
            try:
                res = fut.result(timeout=timeout)
            except Exception as e:
                dataset_path, _ = dataset_jobs[ds_idx]
                res = {
                    "ok": False,
                    "dataset_idx": ds_idx,
                    "dataset_path": dataset_path,
                    "model": m,
                    "instance": inst_name,
                    "level": lv.value,
                    "iteration": it,
                    "error": f"timeout: {e}",
                }

            res["full_evaluation_run_args"] = job["evaluation_config"]
            res["config_path"] = job.get("config_path")
            res["config_key"] = job["config_key"]

            config_entry = by_config[job["config_key"]]

            if res["ok"]:
                successes.append(res)
                config_entry["successes"].append(res)
                ds_idx = res["dataset_idx"]
                ds_path, ds = dataset_jobs[ds_idx]
                config_entry["by_dataset"][ds_path]["successes"].append(res)
                ds.add_eval_results(
                    model_name=res["model"].name,
                    instance_name=res["instance"],
                    level=RepresentationLevel.parse(res["level"]),
                    interactive_mode=job["interactive_config"].mode if job["interactive_config"] else None,
                    results=res["report"],
                    full_evaluation_config=res["full_evaluation_run_args"],
                    session_data=res.get("session_data"),
                )
            else:
                failures.append(res)
                config_entry["failures"].append(res)
                ds_idx = res["dataset_idx"]
                ds_path, _ = dataset_jobs[ds_idx]
                config_entry["by_dataset"][ds_path]["failures"].append(res)

            if progress:
                progress.update(1)
                status = "✅" if res["ok"] else f"❌ {res.get('error','')[:job['debug'].max_error_snippet]}"
                progress.set_postfix_str(
                    f"{m.name}/{res['dataset_path']}/{lv.value}/{inst_name} [it={it}] {status}"
                )

        if progress:
            progress.close()

    if verbose:
        print(f"✅ Done. {len(successes)} succeeded, {len(failures)} failed.")

    if failures:
        print("\n—— Failure summary (top 8) ——")
        for i, fail in enumerate(failures[:8]):
            print(
                f"[{i+1}] {fail['model'].name} | {fail['dataset_path']} | "
                f"{fail['level']} | {fail['instance']}"
            )
            print(f"    error: {fail.get('error','<no message>')}")
        debug_paths = {job["debug"].outdir for job in config_jobs}
        print(
            f"Total failures: {len(failures)} • See {sorted(debug_paths)} "
            "for per-task error bundles (if debugging enabled)."
        )

    return {
        "by_config": by_config,
        "successes": successes,
        "failures": failures,
    }
