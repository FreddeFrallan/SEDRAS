from __future__ import annotations

import json
import os
import base64
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from anthropic import Anthropic
except Exception:
    Anthropic = None

from interactive.tool_management import ToolDefinition
from inference.model_wrappers.base_wrapper import LLMWrapper, UploadedFileHandle
from ..helpers import _extract_last_code_block, _load_classify_function_safely
from ..prompts import CLASSIFIER_SYSTEM_PROMPT
from ..types import ChatMessage


class ClaudeWrapper(LLMWrapper):
    provider_label = "claude"

    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None):
        if Anthropic is None:
            raise RuntimeError("anthropic SDK not installed. `pip install anthropic`")
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set and no api_key provided.")
        self.client = Anthropic(api_key=key)
        self.model = model

    def _extract_usage(self, response: Any) -> Dict[str, Any]:
        usage = getattr(response, "usage", None)
        if not usage:
            return {}
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        return usage.dict() if hasattr(usage, "dict") else dict(usage)

    def _format_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        formatted = []
        for t in tools:
            raw = t.to_dict() if hasattr(t, "to_dict") else t
            if isinstance(raw, dict) and "function" in raw:
                f = raw["function"]
                formatted.append({
                    "name": f.get("name"),
                    "description": f.get("description"),
                    "input_schema": f.get("parameters"),
                })
            else:
                formatted.append(raw)
        return formatted

    def chat_with_tools_with_info_and_files(
            self,
            messages: List[ChatMessage],
            tools: List[ToolDefinition],
            file_handles: Optional[List[UploadedFileHandle]] = None,
    ) -> Tuple[ChatMessage, Dict[str, Any]]:

        # 1. Format Messages for Anthropic
        anthropic_msgs = []
        system_content = None

        # Anthropic separates 'system' as a top-level parameter
        filtered_history = []
        for m in messages:
            if m.get("role") == "system":
                system_content = m.get("content")
            else:
                filtered_history.append(m)

        for i, m in enumerate(filtered_history):
            role = m.get("role")
            content = m.get("content") or ""
            tool_calls = m.get("tool_calls")

            msg_blocks = []
            if content:
                msg_blocks.append({"type": "text", "text": content})

            # Interleave files into the final user message
            if role == "user" and i == len(filtered_history) - 1 and file_handles:
                for h in file_handles:
                    msg_blocks.append(self.build_file_prompt_part(h))

            # Handle existing tool uses in history
            if tool_calls:
                for tc in tool_calls:
                    msg_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id"),
                        "name": tc.get("function", {}).get("name"),
                        "input": json.loads(tc.get("function", {}).get("arguments", "{}"))
                        if isinstance(tc.get("function", {}).get("arguments"), str)
                        else tc.get("function", {}).get("arguments"),
                    })

            # Handle tool results
            if role == "tool":
                anthropic_msgs.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.get("tool_call_id"),
                        "content": content
                    }]
                })
            else:
                anthropic_msgs.append({"role": role, "content": msg_blocks})

        # 2. Call Anthropic using the BETA namespace if files are present
        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": anthropic_msgs,
        }
        if system_content:
            kwargs["system"] = system_content
        if tools:
            kwargs["tools"] = self._format_tools(tools)

        # FIX: Use client.beta when PDF support is needed
        if file_handles:
            kwargs["betas"] = ["pdfs-2024-09-25"]
            response = self.client.beta.messages.create(**kwargs)
        else:
            response = self.client.messages.create(**kwargs)

        # 3. Process Response
        text_content = ""
        tool_use_blocks = []
        for block in response.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "tool_use":
                tool_use_blocks.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                })

        msg_out: ChatMessage = {
            "role": "assistant",
            "content": text_content.strip() or None,
            "tool_calls": tool_use_blocks if tool_use_blocks else None,
        }

        return msg_out, self._extract_usage(response)

    def make_call_to_python_code(
            self, input_text: str, verbose: bool = False
    ) -> [Callable[[List[float]], Dict[str, Any]], str]:
        messages = [
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": input_text}
        ]
        msg, _ = self.chat_with_tools_with_info_and_files(messages, [])
        text = msg.get("content") or ""

        if verbose:
            print(f"\n--- Claude Code Gen ---\n{text}\n")

        code = _extract_last_code_block(text)
        if not code:
            raise ValueError("No code block found in Claude response.")

        return _load_classify_function_safely(code), code

    def make_call_with_files(
            self,
            prompt: str,
            file_handles: List[UploadedFileHandle],
    ) -> Tuple[str, Dict[str, Any]]:
        msg, usage = self.chat_with_tools_with_info_and_files(
            [{"role": "user", "content": prompt}],
            [],
            file_handles
        )
        return msg.get("content") or "", usage

    def make_call_with_info(self, input_text: str) -> Tuple[str, Dict[str, Any]]:
        return self.make_call_with_files(input_text, [])

    def make_call(self, input_text: str) -> str:
        text, _ = self.make_call_with_info(input_text)
        return text

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
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Claude wrapper only supports PDF uploads. Got: {path.suffix}")

        if not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Claude handles PDFs via base64 in the message body, not a separate upload endpoint
        # We store the local path in the file_id for later base64 encoding
        return UploadedFileHandle(
            provider=self.provider_label,
            file_id=str(path.absolute()),
            display_name=display_name or path.name,
            mime_type=mime_type or "application/pdf"
        )

    def build_file_prompt_part(self, handle: UploadedFileHandle) -> Dict[str, Any]:
        """Encodes the PDF as base64 for inclusion in the message block."""
        with open(handle.file_id, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")

        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": data,
            },
        }