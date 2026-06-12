from __future__ import annotations

import time
import json
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import litellm
except Exception:
    litellm = None

from interactive.tool_management import ToolDefinition
from inference.model_wrappers.base_wrapper import LLMWrapper, UploadedFileHandle
from ..helpers import _extract_last_code_block, _load_classify_function_safely
from ..prompts import CLASSIFIER_SYSTEM_PROMPT
from ..types import ChatMessage


class LiteLLMChatWrapper(LLMWrapper):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 1.0,
        reasoning_effort: Optional[str] = None,
    ):
        if litellm is None:
            raise RuntimeError("litellm is not installed. `pip install litellm`")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        if self.reasoning_effort:
            print(f"Initialized LiteLLM with reasoning_effort={self.reasoning_effort}")

    def _completion(self, messages: List[ChatMessage], **extra_kwargs: Any):
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        kwargs.update(extra_kwargs)
        if self.reasoning_effort and "reasoning_effort" not in kwargs:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key

        try:
            return self.rate_limit_safe_call(**kwargs)
        except Exception as e:
            print("LiteLLM API Error (_completion):", e)
            raise e

    def rate_limit_safe_call(self, **kwargs: Any):
        wait_time = 30.0
        while True:
            try:
                return litellm.completion(**kwargs)
            except litellm.RateLimitError as e:
                print(f"----\nRate limit hit: {e}\n Retrying in {wait_time}s...\n----")
                time.sleep(wait_time)
                wait_time *= 2

    def _extract_usage_dict(self, resp: Any) -> Dict[str, Any]:
        """Converts LiteLLM Usage object to a JSON-serializable dictionary."""
        usage = getattr(resp, "usage", None)
        if usage is None:
            return {}

        # LiteLLM usage is a Pydantic model; model_dump() is the standard conversion
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        elif hasattr(usage, "dict"):
            return usage.dict()
        return dict(usage)

    def _extract_message_content(self, resp) -> str:
        try:
            return resp.choices[0].message["content"] or ""
        except Exception:
            return ""

    def _extract_message_and_tool_calls(self, resp) -> ChatMessage:
        try:
            message = resp.choices[0].message
            content = message.get("content") or None
            tool_calls = message.get("tool_calls") or None
        except Exception:
            content, tool_calls = None, None

        tool_calls_payload = None
        if tool_calls:
            tool_calls_payload = []
            for tc in tool_calls:
                tool_calls_payload.append({
                    "id": tc.get("id"),
                    "type": tc.get("type"),
                    "function": {
                        "name": tc.get("function", {}).get("name"),
                        "arguments": tc.get("function", {}).get("arguments"),
                    },
                })
        return {"role": "assistant", "content": content, "tool_calls": tool_calls_payload}

    # --- Standard Methods (Proxies to _with_info) ---

    def make_call(self, input_text: str) -> str:
        content, _ = self.make_call_with_info(input_text)
        return content

    def make_call_to_python_code(self, input_text: str, verbose: bool = False):
        func, code, _ = self.make_call_to_python_code_with_info(input_text, verbose)
        return func, code

    def chat_with_tools(self, messages: List[ChatMessage], tools: List[ToolDefinition]) -> ChatMessage:
        msg, _ = self.chat_with_tools_with_info(messages, tools)
        return msg

    # --- JSON Serializable Methods with Info ---

    def make_call_with_info(self, input_text: str) -> Tuple[str, Dict[str, Any]]:
        resp = self._completion(messages=[{"role": "user", "content": input_text}])
        return self._extract_message_content(resp), self._extract_usage_dict(resp)

    def make_call_to_python_code_with_info(
            self, input_text: str, verbose: bool = False
    ) -> Tuple[Callable[[List[float]], Dict[str, Any]], str, Dict[str, Any]]:
        resp = self._completion(
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": input_text},
            ],
        )
        text = self._extract_message_content(resp)
        if verbose:
            print(f"\nRAW RESPONSE:\n{text.strip()}\n")

        code = _extract_last_code_block(text)
        if not code:
            raise ValueError("No code block found in LLM response.")

        return _load_classify_function_safely(code), code, self._extract_usage_dict(resp)

    def chat_with_tools_with_info(
            self,
            messages: List[ChatMessage],
            tools: List[ToolDefinition],
    ) -> Tuple[ChatMessage, Dict[str, Any]]:
        tool_kwargs: Dict[str, Any] = {}
        if tools:
            tool_kwargs["tools"] = tools
            tool_kwargs["tool_choice"] = "auto"

        resp = self._completion(messages=messages, **tool_kwargs)
        return self._extract_message_and_tool_calls(resp), self._extract_usage_dict(resp)

    def chat_with_tools_with_info_and_files(
            self,
            messages: List[ChatMessage],
            tools: List[ToolDefinition],
            file_handles: Optional[List[UploadedFileHandle]] = None,
    ) -> Tuple[ChatMessage, Dict[str, Any]]:
        if file_handles:
            raise NotImplementedError("LiteLLM wrapper does not support file uploads.")
        return self.chat_with_tools_with_info(messages, tools)
