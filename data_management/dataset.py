from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Any, Sequence, Union
import os, json, re, uuid
from datetime import datetime

from data_management.underlying_data.underlying_data import UnderlyingDataDistribution, Variable, Rule
from data_management.creation.dataset_creation_statistics import (
    DatasetCreationStatistics,
    DATASET_CREATION_STATISTICS_FILENAME,
)
from configs.utils import convert_config_to_json_default
from configs.interactive_config import InteractiveMode
from enum import Enum


# =========================
# Core sample representation
# =========================

@dataclass
class DatasetSample:
    """
    Represents one dataset sample instance.
    """
    assignment: Dict[str, int]
    score: float
    label: int
    instance_text: Optional[str] = None
    instance_label_text: Optional[Dict[str, Any]] = None
    instance_type: Optional[str] = None
    numerical_values: Dict[str, float] = field(default_factory=dict)
    property_labels: Dict[str, Optional[int]] = field(default_factory=dict)
    property_scores: Dict[str, Optional[float]] = field(default_factory=dict)

    # ---------------------------------------
    # Serialization
    # ---------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert the sample to a JSON-serializable dictionary."""
        return {
            "assignment": self.assignment,
            "score": self.score,
            "label": self.label,
            "instance_text": self.instance_text,
            "instance_label_text": self.instance_label_text,
            "instance_type": self.instance_type,
            "numerical_values": self.numerical_values,
            "property_labels": self.property_labels,
            "property_scores": self.property_scores,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetSample":
        """Create a DatasetSample from a dict (e.g., loaded from JSON)."""
        return cls(
            assignment=data.get("assignment", {}),
            score=float(data.get("score", 0.0)),
            label=int(data.get("label", 0)),
            instance_text=data.get("instance_text"),
            instance_label_text=data.get("instance_label_text"),
            instance_type=data.get("instance_type"),
            numerical_values={str(k): float(v) for k, v in (data.get("numerical_values", {}) or {}).items()},
            property_labels={str(k): (None if v is None else int(v)) for k, v in (data.get("property_labels", {}) or {}).items()},
            property_scores={str(k): (None if v is None else float(v)) for k, v in (data.get("property_scores", {}) or {}).items()},
        )

    def __repr__(self) -> str:
        score = round(self.score, 2)
        return (
            f"DatasetSample(assignment={self.assignment}, score={score}, label={self.label}, "
            f"has_text={self.instance_text is not None}, has_label_text={self.instance_label_text is not None}, numerical_values={self.numerical_values}, "
            f"property_labels={self.property_labels})"
        )


class RepresentationLevel(Enum):
    RAW = "raw"
    TEMPLATE_BASED = "template_based"
    FREE_TEXT = "free_text"
    FREE_TEXT_LONG = "free_text_long"
    FREE_TEXT_LONG_FILES = "free_text_long"
    FILES = "files"
    UNKNOWN = "unknown"

    @classmethod
    def parse(cls, value: str | RepresentationLevel | None) -> "RepresentationLevel":
        if isinstance(value, RepresentationLevel):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            for m in cls:
                if m.value == v:
                    return m
        return cls.UNKNOWN

OUTPUT_PROPERTIES_FILE = "output_properties.json"


# =========================
# GeneratedDataset container
# =========================

@dataclass
class DatasetInstance:
    """
    Single in-memory dataset bundle.
    """
    name: str
    udd: Any
    samples: List["DatasetSample"]
    raw_samples: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    text_mapping: Optional[Dict[str, Any]] = None
    instance_hint_used: Optional[str] = None
    rendered_documents: List[Dict[str, Any]] = field(default_factory=list)

    # ✅ NEW FIELD
    representation_level: RepresentationLevel = RepresentationLevel.UNKNOWN
    output_properties: Dict[str, "UnderlyingDataDistribution"] = field(default_factory=dict)
    property_label_support: Dict[str, List[int]] = field(default_factory=dict)

    # ---------- Serialization helpers ----------
    @staticmethod
    def _udd_to_dict(udd: "UnderlyingDataDistribution") -> Dict[str, Any]:
        return {
            "num_output_labels": udd.num_output_labels,
            "variables": {name: var.to_dict() for name, var in udd.variables.items()},
            "rules": [r.to_dict() for r in udd.rules],
        }

    @staticmethod
    def _udd_from_dict(data: Dict[str, Any]) -> "UnderlyingDataDistribution":
        num_output_labels = int(data.get("num_output_labels", 2))
        variables = {name: Variable.from_dict(vd) for name, vd in data["variables"].items()}
        for var in variables.values():
            if var.num_output_labels != num_output_labels:
                var.num_output_labels = num_output_labels
                var.__post_init__()
        rules = [Rule.from_dict(rd) for rd in data["rules"]]
        return UnderlyingDataDistribution(
            num_output_labels=num_output_labels, variables=variables, rules=rules
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        v3 bundle format.
        Backward compatible with v2 and v1 loaders.
        """
        return {
            "version": 3,
            "name": self.name,
            "metadata": self.metadata or {},
            "udd": self._udd_to_dict(self.udd),
            "dataset": [s.to_dict() for s in self.samples],
            "text_mapping": self.text_mapping,
            "instance_hint_used": self.instance_hint_used,

            # ✅ NEW SERIALIZED FIELD
            "representation_level": self.representation_level.value,
            "rendered_documents": self._serialize_rendered_documents(),
        }

    def _serialize_rendered_documents(self) -> List[Dict[str, Any]]:
        """
        Runtime helpers may attach non-portable information (e.g., absolute paths).
        Strip those keys before persisting to disk.
        """
        serialized: List[Dict[str, Any]] = []
        for doc in self.rendered_documents or []:
            if not isinstance(doc, dict):
                continue
            entry = {k: v for k, v in doc.items() if k != "absolute_path"}
            if entry:
                serialized.append(entry)
        return serialized

    def assign_property_data(
        self,
        output_properties: Optional[Dict[str, "UnderlyingDataDistribution"]] = None,
        property_label_support: Optional[Dict[str, List[int]]] = None,
    ) -> None:
        """
        Attach property metadata to this dataset instance for easy access.
        """
        self.output_properties = dict(output_properties or {})
        self.property_label_support = {k: list(v) for k, v in (property_label_support or {}).items()}

    @classmethod
    def from_dict(cls, blob: Dict[str, Any], fallback_name: Optional[str] = None) -> "DatasetInstance":
        version = int(blob.get("version", 1))

        if "udd" not in blob or "dataset" not in blob:
            raise ValueError("Invalid file: missing 'udd' or 'dataset'.")

        udd = cls._udd_from_dict(blob["udd"])
        samples = [DatasetSample.from_dict(d) for d in blob["dataset"]]
        metadata = blob.get("metadata", {}) or {}

        text_mapping = blob.get("text_mapping", None) if version >= 2 else None
        instance_hint_used = blob.get("instance_hint_used", None) if version >= 2 else None

        # ✅ NEW: load representation_level (fallback to UNKNOWN)
        level_raw = blob.get("representation_level", "unknown")
        representation_level = RepresentationLevel.parse(level_raw)

        # name handling
        name = blob.get("name", None) if version >= 2 else None
        if name is None:
            if not fallback_name:
                raise ValueError("Instance name missing and no fallback_name provided.")
            name = fallback_name

        return cls(
            name=name,
            udd=udd,
            samples=samples,
            metadata=metadata,
            text_mapping=text_mapping,
            instance_hint_used=instance_hint_used,
            representation_level=representation_level,
            rendered_documents=[
                d for d in (blob.get("rendered_documents") or []) if isinstance(d, dict)
            ],
        )

    # ---------- File IO ----------
    def save(self, json_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✅ Saved DatasetInstance('{self.name}') to {json_path}")

    @classmethod
    def load(cls, json_path: str) -> "DatasetInstance":
        with open(json_path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        fallback_name = os.path.splitext(os.path.basename(json_path))[0]
        inst = cls.from_dict(blob, fallback_name=fallback_name)
        print(f"✅ Loaded DatasetInstance('{inst.name}') from {json_path}")
        return inst


class AbstractDataset:
    def __init__(
        self,
        udd: "UnderlyingDataDistribution",
        *,
        dataset_instances: Optional[Dict[RepresentationLevel, Dict[str, "DatasetInstance"]]] = None,
        raw_samples: Optional[List[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        output_properties: Optional[Dict[str, "UnderlyingDataDistribution"]] = None,
        property_label_support: Optional[Dict[str, List[int]]] = None,
        creation_statistics: Optional[DatasetCreationStatistics] = None,
    ):
        self.udd = udd
        # Nested: level -> { name -> instance }
        self.dataset_instances: Dict[RepresentationLevel, Dict[str, DatasetInstance]] = dataset_instances or {}
        self.raw_samples: List[Any] = raw_samples or []
        self.metadata: Dict[str, Any] = metadata or {}
        self.output_properties: Dict[str, UnderlyingDataDistribution] = output_properties or {}
        self.property_label_support: Dict[str, List[int]] = property_label_support or {}
        self.creation_statistics: Optional[DatasetCreationStatistics] = creation_statistics

        # Noise theories generated for this dataset: {name -> payload}
        self.noise_theories: Dict[str, Dict[str, Any]] = {}

        # evaluation results: now include representation_level for disambiguation
        # { model_name: { level.value: { instance_name: [ run_dict, ... ] } } }
        self.evaluation_results: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {}

        self._root_folder: Optional[str] = None

    def get_instance(self, level: RepresentationLevel, name: str) -> Optional["DatasetInstance"]:
        """
        Retrieve a DatasetInstance by (level, name), or None if not found.
        """
        lvl = RepresentationLevel.parse(level)
        return self.dataset_instances.get(lvl, {}).get(name, None)

    def add_instance(
            self,
            name: str,
            samples: List["DatasetSample"],
            *,
            level: RepresentationLevel,
            metadata: Optional[Dict[str, Any]] = None,
            text_mapping: Optional[Dict[str, Any]] = None,
            instance_hint_used: Optional[Dict[str, Any] | str] = None,
            rendered_documents: Optional[List[Dict[str, Any]]] = None,
            udd: Optional["UnderlyingDataDistribution"] = None,
            # NEW:
            folder_path: Optional[str] = None,
            persist: bool = True,
    ) -> "DatasetInstance":
        """
        Create and store a DatasetInstance under (level, name) and, by default,
        immediately persist both the instance and global metadata to disk.

        If `folder_path` is provided, it becomes/updates the dataset root.
        If not provided, we require that `self._root_folder` is already known
        (e.g., via .save(...) or a prior add_dataset_instance call).
        """
        lvl = RepresentationLevel.parse(level)

        inst = DatasetInstance(
            name=name,
            udd=udd or self.udd,
            samples=samples,
            raw_samples=list(self.raw_samples),
            metadata=metadata or {},
            text_mapping=text_mapping,
            instance_hint_used=instance_hint_used,
            representation_level=lvl,  # ✅ ensure it’s set on creation
            rendered_documents=list(rendered_documents or []),
        )

        # Register in memory
        self.dataset_instances.setdefault(lvl, {})[name] = inst

        if not persist:
            return inst

        # Determine/ensure root folder
        if folder_path:
            self._root_folder = os.path.abspath(folder_path)

        if not self._root_folder:
            raise RuntimeError(
                "No dataset root folder known. Provide folder_path or call .save(folder_path) once to set it."
            )

        root = self._root_folder

        # Ensure instances/<level> exists
        instances_dir = os.path.join(root, "instances", self._slug(lvl.value))
        os.makedirs(instances_dir, exist_ok=True)

        # Save the instance itself
        instance_path = os.path.join(instances_dir, f"{self._slug(name)}.json")
        inst.save(instance_path)
        print(f"💾 Added and saved DatasetInstance('{inst.name}') at level '{lvl.value}'")

        # Persist global components
        with open(os.path.join(root, "UDD.json"), "w", encoding="utf-8") as f:
            json.dump(self._udd_to_dict(self.udd), f, indent=2)

        with open(os.path.join(root, "raw_samples.json"), "w", encoding="utf-8") as f:
            json.dump(self._serialize_raw_samples(self.raw_samples), f, indent=2)

        self._persist_metadata()

        serialized_props = self._serialize_output_properties()
        if serialized_props:
            with open(os.path.join(root, OUTPUT_PROPERTIES_FILE), "w", encoding="utf-8") as f:
                json.dump(serialized_props, f, indent=2)

        # Persist any in-memory eval results
        self._persist_all_eval_results(root)

        # Persist noise theories
        self._persist_all_noise_theories(root)

        # Persist creation statistics (if available)
        self._persist_creation_statistics(root)

        print(f"✅ AbstractDataset updated and persisted to {root}")
        return inst

    def get_all_instance_names(self) -> List[str]:
        """
        Get a list of all dataset instance names across all levels.
        """
        names = []
        for by_name in self.dataset_instances.values():
            names.extend(by_name.keys())

        # Return unique names only
        names = list(set(names))

        return names

    # ---------- NEW ----------
    def add_dataset_instance(
        self,
        instance: "DatasetInstance",
        folder_path: str,
        *,
        level: RepresentationLevel,
    ) -> None:
        """
        Add a DatasetInstance to memory under the given level and
        immediately persist both the instance and global metadata to disk.
        """
        folder_path = os.path.abspath(folder_path)
        self._root_folder = folder_path  # remember for future writes
        lvl = RepresentationLevel.parse(level)

        instances_dir = os.path.join(folder_path, "instances", self._slug(lvl.value))
        os.makedirs(instances_dir, exist_ok=True)

        # --- Register in memory ---
        instance.raw_samples = list(self.raw_samples)
        self.dataset_instances.setdefault(lvl, {})[instance.name] = instance

        # --- Save instance ---
        instance_path = os.path.join(instances_dir, f"{instance.name}.json")
        instance.save(instance_path)
        print(f"💾 Added and saved DatasetInstance('{instance.name}') at level '{lvl.value}'")

        # --- Save global components ---
        with open(os.path.join(folder_path, "UDD.json"), "w", encoding="utf-8") as f:
            json.dump(self._udd_to_dict(self.udd), f, indent=2)

        if self.raw_samples:
            with open(os.path.join(folder_path, "raw_samples.json"), "w", encoding="utf-8") as f:
                json.dump(self._serialize_raw_samples(self.raw_samples), f, indent=2)

        self._persist_metadata()

        # Persist creation statistics (if available)
        self._persist_creation_statistics(folder_path)

        # Persist any evaluation results we already had
        self._persist_all_eval_results(folder_path)

        # Persist any noise theories we already had
        self._persist_all_noise_theories(folder_path)

        print(f"✅ AbstractDataset updated and persisted to {folder_path}")

    def update_dataset_instance(self, level: RepresentationLevel, name: str, instance: "DatasetInstance") -> None:
        # Update an existing DatasetInstance in memory and on disk.
        lvl = RepresentationLevel.parse(level)
        if lvl not in self.dataset_instances or name not in self.dataset_instances[lvl]:
            raise ValueError(f"DatasetInstance with name '{name}' at level '{lvl.value}' does not exist.")

        self.dataset_instances[lvl][name] = instance
        if not self._root_folder:
            raise RuntimeError("No dataset root folder known. Cannot persist changes.")

        instances_dir = os.path.join(self._root_folder, "instances", self._slug(lvl.value))
        instance_path = os.path.join(instances_dir, f"{self._slug(name)}.json")
        instance.save(instance_path)

    def register_rendered_artifacts(
        self,
        *,
        level: RepresentationLevel,
        instance_name: str,
        artifact_paths: Union[str, Sequence[str]],
        rendering_mode: Optional[Any] = None,
        persist: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Track rendered files for a dataset instance. Updates the in-memory instance,
        metadata.json (via ``rendered_instances``), and persists the instance JSON
        when a root folder is known.
        """
        lvl = RepresentationLevel.parse(level)
        inst = self.get_instance(lvl, instance_name)
        if inst is None:
            raise ValueError(f"DatasetInstance '{instance_name}' not found at level '{lvl.value}'.")

        if isinstance(artifact_paths, str):
            paths = [artifact_paths]
        else:
            paths = [p for p in artifact_paths if p]

        normalized_entries: List[Dict[str, Any]] = []
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        for raw in paths:
            if not raw:
                continue
            abs_path = os.path.abspath(raw)
            rel_path = abs_path
            if self._root_folder:
                try:
                    rel_candidate = os.path.relpath(abs_path, self._root_folder)
                    if not rel_candidate.startswith(".."):
                        rel_path = rel_candidate
                except Exception:
                    pass
            entry = {
                "path": rel_path,
                "absolute_path": abs_path,
                "created_at": timestamp,
            }
            mode_value = self._detect_render_mode(rendering_mode, abs_path)
            if mode_value:
                entry["mode"] = mode_value
            normalized_entries.append(entry)

        def _merge_entries(existing: Sequence[Dict[str, Any]], new_entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
            merged = list(existing or [])
            index_map: Dict[str, int] = {}
            for idx, entry in enumerate(merged):
                key = entry.get("absolute_path") or entry.get("path")
                if key:
                    index_map[key] = idx
            for entry in new_entries:
                key = entry.get("absolute_path") or entry.get("path")
                if not key:
                    continue
                if key in index_map:
                    merged[index_map[key]] = entry
                else:
                    index_map[key] = len(merged)
                    merged.append(entry)
            return merged

        merged_entries = _merge_entries(inst.rendered_documents, normalized_entries)
        inst.rendered_documents = merged_entries
        self._update_render_metadata(lvl, inst.name, merged_entries)
        if lvl is not RepresentationLevel.FILES:
            self._update_render_metadata(RepresentationLevel.FILES, inst.name, merged_entries)
        files_alias = self._ensure_files_alias_instance(lvl, inst.name)
        if files_alias:
            files_alias.rendered_documents = list(merged_entries)

        if persist and self._root_folder:
            self.update_dataset_instance(lvl, instance_name, inst)
            if files_alias and files_alias is not inst:
                self.update_dataset_instance(RepresentationLevel.FILES, instance_name, files_alias)
            self._persist_metadata()
        elif persist and not self._root_folder:
            print("ℹ️ Render metadata recorded in memory; call .save() to persist once a root folder is known.")

        return normalized_entries

    # ---------- NEW: Eval results API (now aware of level) ----------
    def add_eval_results(
            self,
            model_name: str,
            instance_name: str,
            results: Dict[str, Any],
            *,
            level: RepresentationLevel = RepresentationLevel.UNKNOWN,
            interactive_mode: InteractiveMode = InteractiveMode.STATIC,
            full_evaluation_config: Dict[str, Any] = None,
            session_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store one evaluation 'run' for (model_name, level, instance_name) and write it
        to disk under:
            evaluation_results/<model>/<interactive_mode>/<level>/<instance>/<run_id>.json

        Backward-compatible loader handles old layouts.
        """
        lvl = RepresentationLevel.parse(level)

        if full_evaluation_config:
            full_evaluation_config = convert_config_to_json_default(full_evaluation_config)

        run = {
            "run_id": str(uuid.uuid4()),
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model_name": model_name,
            "instance_name": instance_name,
            "representation_level": lvl.value,
            "interactive_mode": interactive_mode.name,
            "results": results,
            "full_evaluation_config": full_evaluation_config or {},
            "session_data": session_data,
        }

        # In-memory registration
        self.evaluation_results \
            .setdefault(model_name, {}) \
            .setdefault(interactive_mode.name, {}) \
            .setdefault(lvl.value, {}) \
            .setdefault(instance_name, []) \
            .append(run)

        if self._root_folder:
            self._persist_single_eval_run(self._root_folder, run)
        else:
            print("ℹ️ No dataset root known yet; eval run stored in memory and will be written on the next .save().")

    # ---------- Helpers ----------
    @staticmethod
    def _slug(s: str) -> str:
        """Filesystem-safe name (keeps letters, digits, ., _, -)."""
        return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "unnamed"

    @staticmethod
    def _detect_render_mode(rendering_mode: Optional[Any], path: Optional[str] = None) -> Optional[str]:
        if isinstance(rendering_mode, Enum):
            return str(rendering_mode.value)
        if hasattr(rendering_mode, "value"):
            try:
                return str(rendering_mode.value)
            except Exception:
                pass
        if rendering_mode:
            return str(rendering_mode)
        if path:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".pdf":
                return "PDF"
            if ext == ".xlsx":
                return "XLSX"
        return None

    def _persist_single_eval_run(self, folder_path: str, run: Dict[str, Any]) -> None:
        base = os.path.join(os.path.abspath(folder_path), "evaluation_results")
        model_dir = os.path.join(base, self._slug(run["model_name"]))

        # interactive_mode dimension (uses enum value)
        interactive_dir = os.path.join(model_dir, self._slug(run.get("interactive_mode", "unknown")))

        # representation_level dimension
        level_dir = os.path.join(interactive_dir, self._slug(run.get("representation_level", "unknown")))

        inst_dir = os.path.join(level_dir, self._slug(run["instance_name"]))
        os.makedirs(inst_dir, exist_ok=True)

        fname = f'{run["run_id"]}.json'
        fpath = os.path.join(inst_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(run, f, indent=2)

    def _persist_all_eval_results(self, folder_path: str) -> None:
        """Write all in-memory runs to files named by run_id (idempotent)."""
        base = os.path.join(os.path.abspath(folder_path), "evaluation_results")
        for model_name, by_level in self.evaluation_results.items():
            for level_value, by_instance in by_level.items():
                for instance_name, runs in by_instance.items():
                    model_dir = os.path.join(base, self._slug(model_name))
                    level_dir = os.path.join(model_dir, self._slug(level_value))
                    inst_dir = os.path.join(level_dir, self._slug(instance_name))
                    os.makedirs(inst_dir, exist_ok=True)
                    for run in runs:
                        fpath = os.path.join(inst_dir, f'{run["run_id"]}.json')
                        with open(fpath, "w", encoding="utf-8") as f:
                            json.dump(run, f, indent=2)

    @staticmethod
    def _udd_to_dict(udd: "UnderlyingDataDistribution") -> Dict[str, Any]:
        return DatasetInstance._udd_to_dict(udd)

    @staticmethod
    def _udd_from_dict(data: Dict[str, Any]) -> "UnderlyingDataDistribution":
        return DatasetInstance._udd_from_dict(data)

    @staticmethod
    def _serialize_raw_samples(raw_samples: List[Any]) -> List[Dict[str, Any]]:
        out = []
        for s in raw_samples:
            if isinstance(s, DatasetSample):
                out.append(s.to_dict())
            else:
                out.append(s)
        return out

    def _serialize_output_properties(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        meta_props: Dict[str, Any] = {}
        try:
            if isinstance(self.metadata, dict):
                meta_props = self.metadata.get("output_properties", {}) or {}
        except Exception:
            meta_props = {}

        def _apply_metadata(name: str, entry: Dict[str, Any]) -> Dict[str, Any]:
            meta = meta_props.get(name, {})
            if isinstance(meta, dict):
                for key in ("type", "num_output_labels", "min_value", "max_value", "unit", "description"):
                    if key in meta and meta[key] is not None:
                        entry[key] = meta[key]
            if "type" not in entry:
                entry["type"] = "categorical"
            return entry

        # Properties with explicit UDDs (categorical)
        for name, udd in self.output_properties.items():
            entry = {
                "udd": self._udd_to_dict(udd),
                "available_for_labels": self.property_label_support.get(name, []),
            }
            data[name] = _apply_metadata(name, entry)

        # Purely metadata-driven properties (e.g., numerical)
        for name, meta in meta_props.items():
            if name in data or not isinstance(meta, dict):
                continue
            entry = {
                "available_for_labels": self.property_label_support.get(name, []),
            }
            data[name] = _apply_metadata(name, entry)

        return data

    def _validate_noise_theory(self, theory: Dict[str, Any]) -> Dict[str, Any]:
        theory_with_defaults = {**theory}
        theory_with_defaults.setdefault("textualized_versions", {})

        required_keys = {
            "accuracy",
            "llm",
            "textual_llm_theory",
            "textual_dt_theory",
            "textualized_versions",
        }
        missing = required_keys - set(theory_with_defaults.keys())
        if missing:
            raise ValueError(f"Noise theory missing required keys: {', '.join(sorted(missing))}")

        textualized_versions_raw = theory_with_defaults.get("textualized_versions", {})
        if not isinstance(textualized_versions_raw, dict):
            raise ValueError("textualized_versions must be a dictionary")

        textualized_versions: Dict[str, str] = {}
        for level, text in textualized_versions_raw.items():
            textualized_versions[str(level)] = str(text)

        normalized = {
            "accuracy": float(theory_with_defaults.get("accuracy", 0.0)),
            "llm": str(theory_with_defaults.get("llm", "")),
            "textual_llm_theory": str(theory_with_defaults.get("textual_llm_theory", "")),
            "textual_dt_theory": str(theory_with_defaults.get("textual_dt_theory", "")),
            "textualized_versions": textualized_versions,
        }

        target_accuracy = theory_with_defaults.get("target_accuracy")
        normalized["target_accuracy"] = (
            float(target_accuracy) if target_accuracy is not None else None
        )

        return normalized

    def _register_noise_theory(self, name: str, theory: Dict[str, Any], *, persist: bool = True) -> None:
        normalized = self._validate_noise_theory(theory)
        safe_name = self._slug(name)
        self.noise_theories[safe_name] = normalized

        if persist and self._root_folder:
            self._persist_all_noise_theories(self._root_folder)

    def add_noise_theory(
        self,
        name: str,
        *,
        accuracy: float,
        target_accuracy: float,
        llm: str,
        textual_llm_theory: str,
        textual_dt_theory: str,
        textualized_versions: Optional[Dict[str, str]] = None,
    ) -> None:
        theory = {
            "accuracy": accuracy,
            "target_accuracy": target_accuracy,
            "llm": llm,
            "textual_llm_theory": textual_llm_theory,
            "textual_dt_theory": textual_dt_theory,
            "textualized_versions": textualized_versions or {},
        }
        self._register_noise_theory(name, theory, persist=True)

    def _persist_all_noise_theories(self, folder_path: str) -> None:
        if not self.noise_theories:
            return

        base = os.path.join(os.path.abspath(folder_path), "noise_theories")
        os.makedirs(base, exist_ok=True)

        for name, theory in self.noise_theories.items():
            fpath = os.path.join(base, f"{self._slug(name)}.json")
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(theory, f, indent=2)

    @staticmethod
    def _maybe_ds_from_dict(d: Dict[str, Any]) -> Any:
        if all(k in d for k in ("assignment", "score", "label")):
            try:
                return DatasetSample.from_dict(d)
            except Exception:
                pass
        return d

    def _persist_creation_statistics(self, folder_path: str) -> None:
        if not self.creation_statistics:
            return
        try:
            self.creation_statistics.save(folder_path)
        except Exception as exc:
            print(f"Failed to persist dataset creation statistics: {exc}")

    def _persist_metadata(self) -> None:
        """
        Persist the in-memory metadata dict if we know the dataset root.
        """
        if not self._root_folder:
            return
        os.makedirs(self._root_folder, exist_ok=True)
        data = self.metadata or {}
        with open(os.path.join(self._root_folder, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _normalize_rendered_documents(
        self,
        documents: Optional[Sequence[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        if not documents:
            return normalized
        for doc in documents:
            if not isinstance(doc, dict):
                continue
            path = doc.get("path")
            if not path:
                continue
            entry = dict(doc)
            abs_path = entry.get("absolute_path")
            if not abs_path:
                if os.path.isabs(path):
                    abs_path = path
                elif self._root_folder:
                    abs_path = os.path.abspath(os.path.join(self._root_folder, path))
                else:
                    abs_path = os.path.abspath(path)
                entry["absolute_path"] = abs_path
            normalized.append(entry)
        return normalized

    def _normalize_all_instance_render_data(self) -> None:
        for by_name in self.dataset_instances.values():
            for inst in by_name.values():
                inst.rendered_documents = self._normalize_rendered_documents(inst.rendered_documents)

    def _update_render_metadata(
        self,
        level: RepresentationLevel,
        instance_name: str,
        artifacts: Sequence[Dict[str, Any]],
    ) -> None:
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        render_meta = self.metadata.setdefault("rendered_instances", {})
        if not isinstance(render_meta, dict):
            render_meta = {}
            self.metadata["rendered_instances"] = render_meta
        level_key = RepresentationLevel.parse(level).value
        level_bucket = render_meta.setdefault(level_key, {})
        sanitized: List[Dict[str, Any]] = []
        for doc in artifacts:
            if not isinstance(doc, dict):
                continue
            entry = {k: v for k, v in doc.items() if k != "absolute_path"}
            if entry:
                sanitized.append(entry)
        if sanitized:
            level_bucket[instance_name] = sanitized
        elif instance_name in level_bucket:
            del level_bucket[instance_name]

    def _apply_render_metadata(self) -> None:
        if not isinstance(self.metadata, dict):
            return
        render_meta = self.metadata.get("rendered_instances")
        if not isinstance(render_meta, dict):
            return
        for level_key, mapping in render_meta.items():
            lvl = RepresentationLevel.parse(level_key)
            by_name = self.dataset_instances.get(lvl)
            if not by_name or not isinstance(mapping, dict):
                continue
            for inst_name, docs in mapping.items():
                inst = by_name.get(inst_name)
                if not inst:
                    continue
                entries = docs if isinstance(docs, list) else []
                inst.rendered_documents = self._normalize_rendered_documents(entries)

    def _register_file_level_aliases(self) -> None:
        """
        If any dataset instance has rendered documents attached, register an alias
        under RepresentationLevel.FILES so callers can explicitly request file-based
        evaluations without mutating the original representation level.
        """
        has_documents = any(
            inst.rendered_documents
            for by_name in self.dataset_instances.values()
            for inst in by_name.values()
        )
        if not has_documents:
            return

        for lvl, by_name in list(self.dataset_instances.items()):
            if lvl == RepresentationLevel.FILES:
                continue
            for inst_name, inst in by_name.items():
                if not inst.rendered_documents:
                    continue
                self._ensure_files_alias_instance(lvl, inst_name)

    def _ensure_files_alias_instance(
        self,
        source_level: RepresentationLevel,
        instance_name: str,
    ) -> Optional[DatasetInstance]:
        source_level = RepresentationLevel.parse(source_level)
        source = self.dataset_instances.get(source_level, {}).get(instance_name)
        if source is None:
            return None

        alias_bucket = self.dataset_instances.setdefault(RepresentationLevel.FILES, {})
        existing = alias_bucket.get(instance_name)
        if existing:
            return existing

        alias = replace(
            source,
            representation_level=RepresentationLevel.FILES,
            rendered_documents=list(source.rendered_documents),
        )
        alias.raw_samples = list(source.raw_samples)
        alias.assign_property_data(source.output_properties, source.property_label_support)
        alias_bucket[instance_name] = alias
        return alias

    # ---------- Save / Load ----------
    def save(self, folder_path: str) -> None:
        folder_path = os.path.abspath(folder_path)
        os.makedirs(folder_path, exist_ok=True)
        self._root_folder = folder_path  # remember

        # Save UDD
        with open(os.path.join(folder_path, "UDD.json"), "w", encoding="utf-8") as f:
            json.dump(self._udd_to_dict(self.udd), f, indent=2)

        # Save raw samples
        with open(os.path.join(folder_path, "raw_samples.json"), "w", encoding="utf-8") as f:
            json.dump(self._serialize_raw_samples(self.raw_samples), f, indent=2)

        # Save metadata
        self._persist_metadata()

        # Save any noise theories we have
        self._persist_all_noise_theories(folder_path)

        # Save creation statistics (if available)
        self._persist_creation_statistics(folder_path)

        serialized_props = self._serialize_output_properties()
        if serialized_props:
            with open(os.path.join(folder_path, OUTPUT_PROPERTIES_FILE), "w", encoding="utf-8") as f:
                json.dump(serialized_props, f, indent=2)

        # Save instances by level
        instances_root = os.path.join(folder_path, "instances")
        os.makedirs(instances_root, exist_ok=True)
        for lvl, by_name in self.dataset_instances.items():
            level_dir = os.path.join(instances_root, self._slug(lvl.value))
            os.makedirs(level_dir, exist_ok=True)
            for name, inst in by_name.items():
                inst.name = name
                inst.save(os.path.join(level_dir, f"{name}.json"))

        # Persist evaluation results
        self._persist_all_eval_results(folder_path)

        print(f"✅ Saved AbstractDataset to {folder_path}")

    @classmethod
    def load(cls, folder_path: str) -> "AbstractDataset":
        folder_path = os.path.abspath(folder_path)

        udd_path = os.path.join(folder_path, "UDD.json")
        if not os.path.isfile(udd_path):
            raise FileNotFoundError(f"Missing UDD.json in {folder_path}")
        with open(udd_path, "r", encoding="utf-8") as f:
            udd_blob = json.load(f)
        udd = cls._udd_from_dict(udd_blob)

        obj = cls(udd=udd)
        obj._root_folder = folder_path  # remember

        # raw samples
        raw_path = os.path.join(folder_path, "raw_samples.json")
        if os.path.isfile(raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                raw_list = json.load(f)
            obj.raw_samples = [cls._maybe_ds_from_dict(d) for d in raw_list]

        # metadata
        meta_path = os.path.join(folder_path, "metadata.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                obj.metadata = json.load(f)

        # creation statistics (optional)
        stats_path = os.path.join(folder_path, DATASET_CREATION_STATISTICS_FILENAME)
        if os.path.isfile(stats_path):
            try:
                obj.creation_statistics = DatasetCreationStatistics.load(folder_path)
            except Exception as exc:
                print(f"⚠️ Failed to load dataset creation statistics: {exc}")

        # noise theories (optional)
        noise_dir = os.path.join(folder_path, "noise_theories")
        if os.path.isdir(noise_dir):
            for fname in sorted(os.listdir(noise_dir)):
                if not fname.lower().endswith(".json"):
                    continue
                theory_path = os.path.join(noise_dir, fname)
                try:
                    with open(theory_path, "r", encoding="utf-8") as f:
                        theory_blob = json.load(f)
                    obj._register_noise_theory(os.path.splitext(fname)[0], theory_blob, persist=False)
                except Exception as exc:
                    print(f"⚠️ Failed to load noise theory {fname}: {exc}")

        # optional output properties
        props_path = os.path.join(folder_path, OUTPUT_PROPERTIES_FILE)
        if os.path.isfile(props_path):
            try:
                with open(props_path, "r", encoding="utf-8") as f:
                    props_blob = json.load(f)
                meta_props = obj.metadata.setdefault("output_properties", {}) if isinstance(obj.metadata, dict) else None
                for name, entry in (props_blob or {}).items():
                    udd_blob = entry.get("udd") if isinstance(entry, dict) else None
                    if udd_blob:
                        obj.output_properties[name] = cls._udd_from_dict(udd_blob)
                    availability = entry.get("available_for_labels") if isinstance(entry, dict) else []
                    try:
                        obj.property_label_support[name] = [int(v) for v in (availability or [])]
                    except Exception:
                        obj.property_label_support[name] = []

                    if isinstance(entry, dict) and isinstance(meta_props, dict):
                        existing = meta_props.get(name, {}) if isinstance(meta_props.get(name, {}), dict) else {}
                        merged = {**existing}
                        for key in ("type", "num_output_labels", "min_value", "max_value", "unit", "description"):
                            if key in entry and entry[key] is not None:
                                merged[key] = entry[key]
                        meta_props[name] = merged
            except Exception as exc:
                print(f"⚠️ Failed to load output properties: {exc}")

        # instances (support both new and legacy layouts)
        instances_root = os.path.join(folder_path, "instances")
        obj.dataset_instances = {}
        if os.path.isdir(instances_root):
            # Detect if legacy: JSON files directly under instances/
            legacy_files = [
                fn for fn in os.listdir(instances_root)
                if fn.lower().endswith(".json") and os.path.isfile(os.path.join(instances_root, fn))
            ]
            if legacy_files:
                lvl = RepresentationLevel.UNKNOWN
                obj.dataset_instances.setdefault(lvl, {})
                for fname in sorted(legacy_files):
                    inst = DatasetInstance.load(os.path.join(instances_root, fname))
                    obj.dataset_instances[lvl][inst.name] = inst

            # New layout: instances/<level>/<name>.json
            for maybe_level in os.listdir(instances_root):
                level_dir = os.path.join(instances_root, maybe_level)
                if not os.path.isdir(level_dir):
                    continue
                lvl = RepresentationLevel.parse(maybe_level)
                for fname in sorted(os.listdir(level_dir)):
                    if fname.lower().endswith(".json"):
                        inst = DatasetInstance.load(os.path.join(level_dir, fname))
                        obj.dataset_instances.setdefault(lvl, {})[inst.name] = inst

        # Propagate property metadata to instances
        for by_name in obj.dataset_instances.values():
            for inst in by_name.values():
                inst.assign_property_data(obj.output_properties, obj.property_label_support)
                inst.raw_samples = list(obj.raw_samples)

        obj._normalize_all_instance_render_data()
        obj._apply_render_metadata()
        obj._register_file_level_aliases()

        # Load evaluation_results from the canonical layout:
        # evaluation_results/<model>/<interactive_mode>/<level>/<instance>/*.json
        eval_root = os.path.join(folder_path, "evaluation_results")
        obj.evaluation_results = {}
        if os.path.isdir(eval_root):
            def _register_run(
                model_name: str,
                interactive_mode: str,
                level_value: str,
                instance_name: str,
                run_file_path: str,
            ) -> None:
                try:
                    with open(run_file_path, "r", encoding="utf-8") as f:
                        run = json.load(f)
                    run.setdefault("run_id", os.path.splitext(os.path.basename(run_file_path))[0])
                    run.setdefault("model_name", model_name)
                    run.setdefault("instance_name", instance_name)
                    run.setdefault("representation_level", level_value)
                    run.setdefault("interactive_mode", interactive_mode)
                    mode_key = str(run.get("interactive_mode", interactive_mode))
                    level_key = RepresentationLevel.parse(run.get("representation_level", level_value)).value
                    obj.evaluation_results \
                        .setdefault(model_name, {}) \
                        .setdefault(mode_key, {}) \
                        .setdefault(level_key, {}) \
                        .setdefault(instance_name, []) \
                        .append(run)
                except Exception as e:
                    print(f"⚠️ Skipped unreadable eval run {os.path.basename(run_file_path)}: {e}")

            for model in sorted(os.listdir(eval_root)):
                model_dir = os.path.join(eval_root, model)
                if not os.path.isdir(model_dir):
                    continue

                for mode_name in sorted(os.listdir(model_dir)):
                    mode_dir = os.path.join(model_dir, mode_name)
                    if not os.path.isdir(mode_dir):
                        continue
                    if mode_name not in InteractiveMode.__members__:
                        continue

                    for level_name in sorted(os.listdir(mode_dir)):
                        level_dir = os.path.join(mode_dir, level_name)
                        if not os.path.isdir(level_dir):
                            continue
                        level_key = RepresentationLevel.parse(level_name).value

                        for inst in sorted(os.listdir(level_dir)):
                            inst_dir = os.path.join(level_dir, inst)
                            if not os.path.isdir(inst_dir):
                                continue
                            for run_file in sorted(os.listdir(inst_dir)):
                                if run_file.lower().endswith(".json"):
                                    _register_run(
                                        model,
                                        mode_name,
                                        level_key,
                                        inst,
                                        os.path.join(inst_dir, run_file),
                                    )

        # Summary
        num_instances = sum(len(v) for v in obj.dataset_instances.values())
        num_models = len(obj.evaluation_results)
        print(
            f"✅ Loaded AbstractDataset from {folder_path} "
            f"({num_instances} instances across {len(obj.dataset_instances)} levels, "
            f"{len(obj.raw_samples)} raw samples, {num_models} models with evals)"
        )
        return obj
