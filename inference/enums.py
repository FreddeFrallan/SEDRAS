from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


@dataclass(frozen=True)
class LLMIdentifier:
    litellm: str
    native: str | None = None
    variant: str | None = None
    reasoning_effort: str | None = None

    def to_dict(self):
        return asdict(self)

    def __str__(self):
        # Often useful for logging or simple IDs
        return self.litellm

class LLMModel(Enum):
    """Supported LLM model families."""

    GPT_4O = LLMIdentifier(litellm="gpt-4o", native="gpt-4o")
    GPT_5 = LLMIdentifier(litellm="gpt-5", native="gpt-5")
    GPT_5_2 = LLMIdentifier(litellm="gpt-5.2", native="gpt-5.2")
    GPT_5_2_LOW = LLMIdentifier(
        litellm="gpt-5.2",
        native="gpt-5.2",
        variant="low",
        reasoning_effort="low",
    )
    GPT_5_2_MEDIUM = LLMIdentifier(
        litellm="gpt-5.2",
        native="gpt-5.2",
        variant="medium",
        reasoning_effort="medium",
    )
    GPT_5_2_HIGH = LLMIdentifier(
        litellm="gpt-5.2",
        native="gpt-5.2",
        variant="high",
        reasoning_effort="high",
    )
    GPT_5_2_XHIGH = LLMIdentifier(
        litellm="gpt-5.2",
        native="gpt-5.2",
        variant="xhigh",
        reasoning_effort="xhigh",
    )

    GEMINI_2_5_FLASH = LLMIdentifier(
        litellm="gemini/gemini-2.5-flash",
        native="models/gemini-2.5-flash",
    )
    GEMINI_2_5_PRO = LLMIdentifier(
        litellm="gemini/gemini-2.5-pro",
        native="models/gemini-2.5-pro",
    )
    GEMINI_3_PRO = LLMIdentifier(
        litellm="gemini/gemini-3-pro-preview",
        native="models/gemini-3-pro-preview",
    )
    GEMINI_3_FLASH_PREVIEW = LLMIdentifier(
        litellm="gemini/gemini-3-flash-preview",
        native="models/gemini-3-flash-preview",
    )

    CLAUDE_OPUS_4_5 = LLMIdentifier("claude-opus-4-5-20251101")
    CLAUDE_SONNET_4_5 = LLMIdentifier("claude-sonnet-4-5-20250929")
    CLAUDE_HAIKU_3 = LLMIdentifier("claude-3-haiku-20240307")

    GROK_4_1 = LLMIdentifier("xai/grok-4-1-fast-non-reasoning", native="grok-4-1-fast-non-reasoning")
    GROK_4_1_reasoning = LLMIdentifier("xai/grok-4-1-fast-reasoning", native="grok-4-1-fast-reasoning")

    MAGISTRAL_SMALL = LLMIdentifier("mistral/magistral-small-2506")
    MAGISTRAL_MEDIUM = LLMIdentifier("mistral/magistral-medium-2506")

    DEEPSEEK_3_2_THINKING = LLMIdentifier(
        litellm="deepseek/deepseek-reasoner",
        native="deepseek-reasoner",
    )

    CUSTOM_HTTP_BACKEND = LLMIdentifier(
        litellm="custom_http_backend",
        native="custom_http_backend",
    )

    GEMINI_AGENT = LLMIdentifier(
        litellm="gemini_agent",
        native="models/gemini-2.5-pro",
    )

    @property
    def litellm_id(self) -> str:
        return self.value.litellm

    @property
    def native_id(self) -> str:
        return self.value.native or self.value.litellm

    @property
    def reasoning_effort(self) -> str | None:
        return self.value.reasoning_effort



class LLMBackend(Enum):
    """Supported backends for interacting with models."""

    NATIVE = "native"  # Use direct OpenAI / Gemini SDKs
    LITELLM = "litellm"  # Use LiteLLM as a unified backend
