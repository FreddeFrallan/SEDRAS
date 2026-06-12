"""Facade module for LLM wrappers.

This file re-exports the primary entrypoints for interacting with language models
while delegating implementation details to smaller modules within the
``inference`` package.
"""
from __future__ import annotations

from inference.model_wrappers.base_wrapper import LLMWrapper, UploadedFileHandle
from inference.enums import LLMBackend, LLMModel
from inference.factory import get_llm_wrapper
from inference.helpers import _extract_last_code_block, _load_classify_function_safely
from inference.model_wrappers import (
    DeepSeekWrapper,
    GeminiWrapper,
    LiteLLMChatWrapper,
    OpenAIChatWrapper,
    RemoteHTTPWrapper,
    CustomHTTPBackendWrapper,
)
from inference.prompts import CLASSIFIER_SYSTEM_PROMPT, render_classifier_user_prompt
from inference.types import ChatMessage

__all__ = [
    "LLMWrapper",
    "UploadedFileHandle",
    "LLMBackend",
    "LLMModel",
    "get_llm_wrapper",
    "DeepSeekWrapper",
    "GeminiWrapper",
    "LiteLLMChatWrapper",
    "OpenAIChatWrapper",
    "RemoteHTTPWrapper",
    "CustomHTTPBackendWrapper",
    "CLASSIFIER_SYSTEM_PROMPT",
    "render_classifier_user_prompt",
    "ChatMessage",
    "_extract_last_code_block",
    "_load_classify_function_safely",
]
