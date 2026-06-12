from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import pprint

try:
    import xai_sdk
    from xai_sdk.chat import (
        user as xai_user,
        system as xai_system,
        tool as xai_tool,
        file as xai_file,
        tool_result as xai_tool_result
    )
except Exception:
    xai_sdk = None
    xai_user = None
    xai_system = None
    xai_tool = None
    xai_file = None
    xai_tool_result = None

from interactive.tool_management import ToolDefinition
from inference.model_wrappers.base_wrapper import LLMWrapper, UploadedFileHandle
from ..helpers import _extract_last_code_block, _load_classify_function_safely
from ..prompts import CLASSIFIER_SYSTEM_PROMPT
from ..types import ChatMessage


class XaiWrapper(LLMWrapper):
    provider_label = "xai"

    def __init__(self, model: str = "grok-4-1-fast-non-reasoning", api_key: Optional[str] = None):
        if xai_sdk is None:
            raise RuntimeError("xai-sdk not installed. `pip install xai-sdk`")
        key = api_key or os.getenv("XAI_API_KEY")
        if not key:
            raise ValueError("XAI_API_KEY not set and no api_key provided.")

        self.model = model
        self.client = xai_sdk.Client(api_key=key)
        self._convo: Optional[Any] = None
        # We keep track of the last response object to append it when needed
        self._last_response_obj: Optional[Any] = None

    def _get_or_create_convo(self, tools: Optional[List[ToolDefinition]] = None) -> Any:
        if self._convo is None:
            formatted_tools = self._format_tools(tools or [])
            self._convo = self.client.chat.create(
                model=self.model,
                tools=formatted_tools if formatted_tools else None
            )
        return self._convo

    def _format_tools(self, tools: List[ToolDefinition]) -> List[Any]:
        formatted = []
        for t in tools:
            raw = t.to_dict() if hasattr(t, "to_dict") else t
            if isinstance(raw, dict) and "function" in raw:
                f = raw["function"]
                formatted.append(xai_tool(
                    name=f.get("name"),
                    description=f.get("description"),
                    parameters=f.get("parameters")
                ))
        return formatted

    def _sync_messages_to_convo(self, messages: List[ChatMessage],
                                file_handles: Optional[List[UploadedFileHandle]] = None):
        convo = self._get_or_create_convo()
        existing_count = len(convo.messages)
        new_messages = messages[existing_count:]

        if not new_messages:
            return

        file_parts = [self.build_file_prompt_part(h) for h in (file_handles or [])]

        for i, m in enumerate(new_messages):
            role = m.get("role")
            content = m.get("content") or ""

            if role == "system":
                convo.append(xai_system(content))
            elif role == "user":
                is_last_in_total = (len(convo.messages) + 1 == len(messages))
                if is_last_in_total and file_parts:
                    convo.append(xai_user(content, *file_parts))
                else:
                    convo.append(xai_user(content))
            elif role == "assistant":
                # OFFICIAL SDK WAY: Append the response object itself if we have it
                # If we don't (e.g., first turn or re-playing history), use the text helper
                if self._last_response_obj and i == 0:  # Usually the first 'new' message is the last response
                    convo.append(self._last_response_obj)
                else:
                    convo.append(xai_sdk.chat.assistant(content))
            elif role == "tool":
                # Signature: tool_result(content)
                # The SDK pairs this with the preceding tool call in the convo state automatically
                convo.append(xai_tool_result(str(content)))

    def chat_with_tools_with_info_and_files(
            self,
            messages: List[ChatMessage],
            tools: List[ToolDefinition],
            file_handles: Optional[List[UploadedFileHandle]] = None,
    ) -> Tuple[ChatMessage, Dict[str, Any]]:

        convo = self._get_or_create_convo(tools)
        self._sync_messages_to_convo(messages, file_handles)

        response = convo.sample()
        # Store this object for the next turn's synchronization
        self._last_response_obj = response

        tool_calls = None
        if response.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in response.tool_calls
            ]

        msg_dict: ChatMessage = {
            "role": "assistant",
            "content": response.content,
            "tool_calls": tool_calls,
        }

        # Safe extraction for Protobuf usage objects
        usage_dict = {}
        u_obj = getattr(response, "usage", None)
        if u_obj:
            for attr in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                val = getattr(u_obj, attr, None)
                if val is not None:
                    usage_dict[attr] = val

        return msg_dict, usage_dict

    def build_file_prompt_part(self, handle: UploadedFileHandle) -> Any:
        return xai_file(handle.file_id)

    def make_call_with_files(
            self,
            prompt: str,
            file_handles: List[UploadedFileHandle],
    ) -> Tuple[str, Dict[str, Any]]:
        messages = [{"role": "user", "content": prompt}]
        msg_dict, usage = self.chat_with_tools_with_info_and_files(messages, [], file_handles)
        return msg_dict.get("content") or "", usage

    def make_call_to_python_code(
            self, input_text: str, verbose: bool = False
    ) -> [Callable[[List[float]], Dict[str, Any]], str]:
        messages = [
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": input_text}
        ]
        msg_dict, _ = self.chat_with_tools_with_info_and_files(messages, [])
        text = msg_dict.get("content") or ""

        if verbose:
            print(f"\n--- xAI CODE GEN ---\n{text.strip()}\n")

        code = _extract_last_code_block(text)
        if not code:
            raise ValueError("No code block found in xAI response.")

        return _load_classify_function_safely(code), code

    def make_call(self, input_text: str) -> str:
        msg, _ = self.chat_with_tools_with_info_and_files([{"role": "user", "content": input_text}], [])
        return msg.get("content") or ""

    def make_call_with_info(self, input_text: str) -> Tuple[str, Dict[str, Any]]:
        return self.chat_with_tools_with_info_and_files([{"role": "user", "content": input_text}], [])

    @property
    def supports_file_upload(self) -> bool:
        return True

    def upload_file(
            self,
            file_path: str,
            *,
            display_name: Optional[str] = None,
            mime_type: Optional[str] = None,
    ) -> UploadedFileHandle:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        uploaded = self.client.files.upload(str(path))
        return UploadedFileHandle(
            provider=self.provider_label,
            file_id=uploaded.id,
            display_name=display_name or path.name,
            mime_type=mime_type or "application/pdf",
            raw_response={"file_id": uploaded.id, "filename": uploaded.filename},
        )

    def build_file_prompt_part(self, handle: UploadedFileHandle) -> Any:
        return xai_file(handle.file_id)