from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None

from interactive.tool_management import ToolDefinition

from inference.model_wrappers.base_wrapper import LLMWrapper, UploadedFileHandle
from ..helpers import _extract_last_code_block, _load_classify_function_safely
from ..prompts import CLASSIFIER_SYSTEM_PROMPT
from ..types import ChatMessage


class DeepSeekWrapper(LLMWrapper):
    provider_label = "deepseek"

    def __init__(
        self,
        model: str = "deepseek-3.2-thinking",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        if OpenAI is None:
            raise RuntimeError("openai SDK not installed. `pip install openai`")
        key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("DEEPSEEK_API_KEY not set and no api_key provided.")
        self.model = model
        self.client = OpenAI(
            api_key=key,
            base_url=base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
        )

    def _extract_usage_dict(self, resp: Any) -> Dict[str, Any]:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return {}
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        if hasattr(usage, "dict"):
            return usage.dict()
        return dict(usage)

    def make_call(self, input_text: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": input_text}],
        )
        return resp.choices[0].message.content or ""

    def make_call_with_info(self, input_text: str) -> Tuple[str, Dict[str, Any]]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": input_text}],
        )
        return resp.choices[0].message.content or "", self._extract_usage_dict(resp)

    def make_call_to_python_code(
        self, input_text: str, verbose: bool = False
    ) -> [Callable[[List[float]], Dict[str, Any]], str]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": input_text},
            ],
        )
        text = resp.choices[0].message.content or ""

        if verbose:
            print("\n====================== LLM RAW RESPONSE ======================")
            print(text.strip())
            print("==============================================================\n")

        code = _extract_last_code_block(text)
        if not code:
            raise ValueError("No code block found in LLM response.")

        if verbose:
            print("✅ Extracted code block:\n")
            print(code)
            print("--------------------------------------------------------------\n")

        return _load_classify_function_safely(code), code

    def chat_with_tools(
        self,
        messages: List[ChatMessage],
        tools: List[ToolDefinition],
    ) -> ChatMessage:
        tool_kwargs: Dict[str, Any] = {}
        if tools:
            tool_kwargs["tools"] = tools
            tool_kwargs["tool_choice"] = "auto"

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **tool_kwargs,
        )

        response_message = resp.choices[0].message
        message_dict: ChatMessage = {
            "role": "assistant",
            "content": response_message.content,
            "tool_calls": None,
        }

        if response_message.tool_calls:
            message_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in response_message.tool_calls
            ]

        return message_dict

    def chat_with_tools_with_info(
        self,
        messages: List[ChatMessage],
        tools: List[ToolDefinition],
    ) -> Tuple[ChatMessage, Dict[str, Any]]:
        tool_kwargs: Dict[str, Any] = {}
        if tools:
            tool_kwargs["tools"] = tools
            tool_kwargs["tool_choice"] = "auto"

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **tool_kwargs,
        )

        response_message = resp.choices[0].message
        message_dict: ChatMessage = {
            "role": "assistant",
            "content": response_message.content,
            "tool_calls": None,
        }

        if response_message.tool_calls:
            message_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in response_message.tool_calls
            ]

        return message_dict, self._extract_usage_dict(resp)

    def chat_with_tools_with_info_and_files(
        self,
        messages: List[ChatMessage],
        tools: List[ToolDefinition],
        file_handles: Optional[List[UploadedFileHandle]] = None,
    ) -> Tuple[ChatMessage, Dict[str, Any]]:
        if file_handles:
            raise NotImplementedError("DeepSeek wrapper does not support file uploads.")
        return self.chat_with_tools_with_info(messages, tools)
