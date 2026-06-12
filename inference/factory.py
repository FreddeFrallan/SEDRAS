from __future__ import annotations

from typing import Optional

from .enums import LLMBackend, LLMModel
from .model_wrappers import (
    DeepSeekWrapper,
    GeminiWrapper,
    LiteLLMChatWrapper,
    OpenAIChatWrapper,
    ClaudeWrapper,
    RemoteHTTPWrapper,
    XaiWrapper,
)


_NATIVE_MODELS = {
    LLMModel.GPT_4O,
    LLMModel.GPT_5,
    LLMModel.GPT_5_2,
    LLMModel.GPT_5_2_LOW,
    LLMModel.GPT_5_2_MEDIUM,
    LLMModel.GPT_5_2_HIGH,
    LLMModel.GPT_5_2_XHIGH,
    LLMModel.GPT_5_2_XHIGH,
    LLMModel.GEMINI_2_5_FLASH,
    LLMModel.GEMINI_2_5_PRO,
    LLMModel.GEMINI_3_PRO,
    LLMModel.GEMINI_3_FLASH_PREVIEW,
    LLMModel.CLAUDE_OPUS_4_5,
    LLMModel.CLAUDE_SONNET_4_5,
    LLMModel.CLAUDE_HAIKU_3,
    LLMModel.GROK_4_1,
    LLMModel.GROK_4_1_reasoning,
    LLMModel.DEEPSEEK_3_2_THINKING,
    LLMModel.CUSTOM_HTTP_BACKEND,
    LLMModel.GEMINI_AGENT,
}


def _resolve_backend(model: LLMModel, backend: Optional[LLMBackend]) -> LLMBackend:
    """
    Resolve the backend to use for a given ``LLMModel``.

    We default to LiteLLM for backwards compatibility while incrementally adding
    native wrappers. Callers may request ``LLMBackend.NATIVE`` for models that
    already have a dedicated implementation.
    """
    if model in {LLMModel.CUSTOM_HTTP_BACKEND, LLMModel.GEMINI_AGENT}:
        return LLMBackend.NATIVE

    if backend == LLMBackend.NATIVE:
        if model in _NATIVE_MODELS:
            return LLMBackend.NATIVE
        raise ValueError(f"Native backend is not implemented for model: {model.name}")
    return LLMBackend.LITELLM


def get_llm_wrapper(
    model: LLMModel,
    api_key: Optional[str] = None,
    backend: Optional[LLMBackend] = LLMBackend.LITELLM,
    remote_endpoint_url: Optional[str] = None,
    remote_model: Optional[str] = None,
    timeout: Optional[float] = None,
):
    resolved_backend = _resolve_backend(model, backend)
    reasoning_effort = model.reasoning_effort

    if resolved_backend == LLMBackend.LITELLM:
        return LiteLLMChatWrapper(
            model=model.litellm_id,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
        )

    if model == LLMModel.GPT_4O:
        return OpenAIChatWrapper(
            model=LLMModel.GPT_4O.native_id,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
        )
    elif model == LLMModel.GPT_5:
        return OpenAIChatWrapper(
            model=LLMModel.GPT_5.native_id,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
        )
    elif model == LLMModel.GPT_5_2:
        return OpenAIChatWrapper(
            model=LLMModel.GPT_5_2.native_id,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
        )
    elif model == LLMModel.GPT_5_2_LOW:
        return OpenAIChatWrapper(
            model=LLMModel.GPT_5_2_LOW.native_id,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
        )
    elif model == LLMModel.GPT_5_2_MEDIUM:
        return OpenAIChatWrapper(
            model=LLMModel.GPT_5_2_MEDIUM.native_id,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
        )
    elif model == LLMModel.GPT_5_2_HIGH:
        return OpenAIChatWrapper(
            model=LLMModel.GPT_5_2_HIGH.native_id,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
        )
    elif model == LLMModel.GPT_5_2_XHIGH:
        return OpenAIChatWrapper(
            model=LLMModel.GPT_5_2_XHIGH.native_id,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
        )
    elif model == LLMModel.GEMINI_2_5_FLASH:
        return GeminiWrapper(model=LLMModel.GEMINI_2_5_FLASH.native_id, api_key=api_key)
    elif model == LLMModel.GEMINI_2_5_PRO:
        return GeminiWrapper(model=LLMModel.GEMINI_2_5_PRO.native_id, api_key=api_key)
    elif model == LLMModel.GEMINI_3_PRO:
        return GeminiWrapper(model=LLMModel.GEMINI_3_PRO.native_id, api_key=api_key)
    elif model == LLMModel.GEMINI_3_FLASH_PREVIEW:
        return GeminiWrapper(model=LLMModel.GEMINI_3_FLASH_PREVIEW.native_id, api_key=api_key)
    elif model in {LLMModel.CLAUDE_OPUS_4_5, LLMModel.CLAUDE_SONNET_4_5, LLMModel.CLAUDE_HAIKU_3}:
        return ClaudeWrapper(model=model.native_id, api_key=api_key)
    elif model in {LLMModel.GROK_4_1, LLMModel.GROK_4_1_reasoning}:
        return XaiWrapper(model=model.native_id, api_key=api_key)
    elif model == LLMModel.DEEPSEEK_3_2_THINKING:
        return DeepSeekWrapper(model=model.native_id, api_key=api_key)
    elif model == LLMModel.CUSTOM_HTTP_BACKEND:
        return RemoteHTTPWrapper(
            endpoint_url=remote_endpoint_url,
            model=remote_model,
            api_key=api_key,
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unsupported model enum: {model}")
