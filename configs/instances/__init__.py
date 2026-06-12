"""
Lazily exposed config instances to avoid importing heavy dependencies
when they are not needed.
"""
from __future__ import annotations

import importlib
from typing import Dict, Tuple

__all__ = [
    "SEDRAS_2026_PAPER_DATA_CREATION_CONFIG",
    "SEDRAS_2026_MAIN_EVALUATION_CONFIG",
    # Modules commonly imported directly
    "SEDRAS_2026PaperDataCreation",
    "SEDRAS_2026MainEvaluation",
]

_CONFIG_EXPORTS: Dict[str, Tuple[str, str]] = {
    "SEDRAS_2026_PAPER_DATA_CREATION_CONFIG": ("configs.instances.SEDRAS_2026PaperDataCreation", "SEDRAS_2026_PAPER_DATA_CREATION_CONFIG"),
    "SEDRAS_2026_MAIN_EVALUATION_CONFIG": ("configs.instances.SEDRAS_2026MainEvaluation", "SEDRAS_2026_MAIN_EVALUATION_CONFIG"),
}

_MODULE_EXPORTS: Dict[str, str] = {
    "SEDRAS_2026PaperDataCreation": "configs.instances.SEDRAS_2026PaperDataCreation",
    "SEDRAS_2026MainEvaluation": "configs.instances.SEDRAS_2026MainEvaluation",
}


def __getattr__(name: str):
    if name in _CONFIG_EXPORTS:
        module_name, attr_name = _CONFIG_EXPORTS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    if name in _MODULE_EXPORTS:
        module = importlib.import_module(_MODULE_EXPORTS[name])
        globals()[name] = module
        return module

    raise AttributeError(f"module 'configs.instances' has no attribute '{name}'")
