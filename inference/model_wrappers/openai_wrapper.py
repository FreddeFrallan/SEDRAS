from __future__ import annotations

import json
import os
from pathlib import Path
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

_ALLOWED_REASONING_EFFORTS = {"minimal", "low", "medium", "high", "xhigh"}


class OpenAIChatWrapper(LLMWrapper):
    provider_label = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ):
        if OpenAI is None:
            raise RuntimeError("openai SDK not installed. `pip install openai`")
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.reasoning_effort = self._normalize_reasoning_effort(reasoning_effort)
        if self.reasoning_effort:
            print(f"Initialized OpenAIChatWrapper with reasoning_effort={self.reasoning_effort}")

    def _normalize_reasoning_effort(self, requested: Optional[str]) -> Optional[str]:
        if not requested:
            return None
        normalized = requested.lower()
        if normalized in _ALLOWED_REASONING_EFFORTS:
            return normalized
        fallback = "high"
        print(
            f"[OpenAIChatWrapper] Requested reasoning_effort '{requested}' is not supported "
            f"by the native API (allowed: {sorted(_ALLOWED_REASONING_EFFORTS)}). "
            f"Falling back to '{fallback}'."
        )
        return fallback

    def _extract_reasoning_effort_from_resp(self, resp: Any) -> Optional[str]:
        if resp is None:
            return None
        direct = getattr(resp, "reasoning_effort", None)
        if direct is not None:
            return direct

        for attr_name in ("model_dump", "to_dict"):
            accessor = getattr(resp, attr_name, None)
            if not callable(accessor):
                continue
            try:
                payload = accessor()
            except Exception:
                continue
            if isinstance(payload, dict):
                if "reasoning_effort" in payload:
                    return payload["reasoning_effort"]
                reasoning = payload.get("reasoning")
                if isinstance(reasoning, dict):
                    effort = reasoning.get("effort")
                    if effort is not None:
                        return effort
        return None

    def _log_reasoning_effort(self, resp: Any):
        effort = self._extract_reasoning_effort_from_resp(resp)
        print(f"Reasoning with Native '{effort}'...")

    def _reasoning_kwargs(self) -> Dict[str, Any]:
        if not self.reasoning_effort:
            return {}
        return {"reasoning_effort": self.reasoning_effort}

    def _responses_reasoning_kwargs(self) -> Dict[str, Any]:
        if not self.reasoning_effort:
            return {}
        return {"reasoning": {"effort": self.reasoning_effort}}

    def make_call(self, input_text: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": input_text},
            ],
            **self._reasoning_kwargs(),
        )
        self._log_reasoning_effort(resp)
        return resp.choices[0].message.content or ""

    def _extract_usage_dict(self, resp: Any) -> Dict[str, Any]:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return {}
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        if hasattr(usage, "dict"):
            return usage.dict()
        return dict(usage)

    def make_call_with_info(self, input_text: str) -> Tuple[str, Dict[str, Any]]:

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": input_text}],
            **self._reasoning_kwargs(),
        )
        self._log_reasoning_effort(resp)
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
            **self._reasoning_kwargs(),
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
        tool_kwargs = {}
        if tools:
            tool_kwargs["tools"] = tools
            tool_kwargs["tool_choice"] = "auto"

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **tool_kwargs,
            **self._reasoning_kwargs(),
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
            raise ValueError(
                f"{self.provider_label} wrapper only supports PDF uploads. Got: {path.suffix or 'no extension'}"
            )
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        with path.open("rb") as fh:
            uploaded = self.client.files.create(
                file=fh,
                purpose="assistants",
            )
        raw = uploaded.model_dump() if hasattr(uploaded, "model_dump") else uploaded
        return UploadedFileHandle(
            provider=self.provider_label,
            file_id=uploaded.id,
            display_name=display_name or path.name,
            mime_type=mime_type,
            raw_response=raw,
        )

    def build_file_prompt_part(self, handle: UploadedFileHandle) -> Dict[str, Any]:
        if handle.provider != self.provider_label:
            raise ValueError(
                f"Cannot build OpenAI file part for provider '{handle.provider}'."
            )
        return {"type": "input_file", "file_id": handle.file_id}

    def _collect_responses_text(self, response: Any) -> str:
        # DEBUG PRINT: Inspect the overall response structure
        # print("\n--- DEBUG: OpenAI Response Object ---")
        # print(f"Type: {type(response)}")
        # # Attempt to print as dict for readability, fallback to repr
        # try:
        #     print(json.dumps(response.model_dump(), indent=2) if hasattr(response, "model_dump") else response)
        # except Exception:
        #     print(response)
        # print("-------------------------------------\n")

        texts = getattr(response, "output_text", None)
        if texts:
            if isinstance(texts, str):
                return texts.strip()
            combined = "\n\n".join(t.strip() for t in texts if t and t.strip())
            if combined:
                return combined

        outputs = getattr(response, "output", None) or []
        collected: List[str] = []

        for item in outputs:
            # Convert Pydantic object to dict safely
            if hasattr(item, "model_dump"):
                item_dict = item.model_dump()
            elif isinstance(item, dict):
                item_dict = item
            else:
                item_dict = {}

            # Safe access to 'content' list
            content_blocks = item_dict.get("content") or []

            for block in content_blocks:
                # Blocks can also be objects or dicts
                text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
                if text:
                    collected.append(text)

        return "\n\n".join(segment.strip() for segment in collected if segment.strip())

    def make_call_with_files(
        self,
        prompt: str,
        file_handles: List[UploadedFileHandle],
    ) -> Tuple[str, Dict[str, Any]]:
        if not file_handles:
            return self.make_call_with_info(prompt)

        content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        content.extend(self.build_file_prompt_part(h) for h in file_handles)
        print(f"Reasoning with Native '{self.reasoning_effort}'...")
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "type": "message",
                    "content": content,
                }
            ],
            # max_output_tokens=800,
            **self._responses_reasoning_kwargs(),
        )
        text = self._collect_responses_text(response)
        usage = getattr(response, "usage", None)
        if hasattr(usage, "model_dump"):
            usage_dict = usage.model_dump()
        elif hasattr(usage, "dict"):
            usage_dict = usage.dict()
        else:
            usage_dict = {}
        return (text or "").strip(), usage_dict

    def _extract_response_tool_calls(self, response: Any) -> Optional[List[Dict[str, Any]]]:
        output = getattr(response, "output", None) or []
        tool_calls: List[Dict[str, Any]] = []
        for item in output:
            if item.get("type") == "tool_call":
                tool_calls.append(self._normalize_tool_call(item))
            for block in item.get("content", []) or []:
                if block.get("type") == "tool_call":
                    tool_calls.append(self._normalize_tool_call(block))
        return tool_calls or None

    def _normalize_tool_call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        arguments = payload.get("arguments", {})
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        return {
            "id": payload.get("id") or payload.get("call_id") or "",
            "type": "function",
            "function": {
                "name": payload.get("name") or payload.get("tool_name") or "",
                "arguments": arguments,
            },
        }

    def chat_with_tools_with_info_and_files(
            self,
            messages: List[ChatMessage],
            tools: List[ToolDefinition],
            file_handles: Optional[List[UploadedFileHandle]] = None,
    ) -> Tuple[ChatMessage, Dict[str, Any]]:
        # 1. Fallback to standard chat logic if no files are involved
        if not file_handles:
            response_message = self.chat_with_tools(messages, tools)
            return response_message, {}

        # 2. Prepare the combined prompt for the Responses API
        prompt_text = "\n\n".join(
            f"{msg.get('role', 'unknown').upper()}: {msg.get('content', '')}"
            for msg in messages
        )
        content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt_text}]
        for handle in file_handles:
            content.append(self.build_file_prompt_part(handle))

        # 3. Format Tool Definitions (Flattened schema required by Responses API)
        formatted_tools = []
        if tools:
            for t in tools:
                raw_t = t.to_dict() if hasattr(t, "to_dict") else t
                if isinstance(raw_t, dict) and "function" in raw_t:
                    func_data = raw_t["function"]
                    formatted_tools.append({
                        "type": "function",
                        "name": func_data.get("name"),
                        "description": func_data.get("description"),
                        "parameters": func_data.get("parameters"),
                    })
                else:
                    formatted_tools.append(raw_t)

        # 4. Execute the call
        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "type": "message", "content": content}],
            tools=formatted_tools or None,
            tool_choice="auto" if tools else None,
            **self._responses_reasoning_kwargs(),
        )

        # 5. Extraction Logic
        collected_text_parts: List[str] = []
        extracted_tool_calls: List[Dict[str, Any]] = []

        outputs = getattr(response, "output", []) or []
        for i, item in enumerate(outputs):
            # Convert Pydantic objects safely to dictionaries
            item_dict = item.model_dump() if hasattr(item, "model_dump") else (item if isinstance(item, dict) else {})

            # Check for standard text content blocks
            for block in item_dict.get("content", []) or []:
                if "text" in block:
                    collected_text_parts.append(block["text"])

            # Check for Tool/Function calls (supports multiple naming conventions)
            item_type = item_dict.get("type")
            if item_type in ["function_call", "tool_call", "call"] or "call_id" in item_dict:
                args = item_dict.get("arguments")
                if isinstance(args, dict):
                    args = json.dumps(args)

                call_id = item_dict.get("id") or item_dict.get("call_id") or f"call_{i}"
                call_name = (
                        item_dict.get("name") or
                        item_dict.get("tool_name") or
                        item_dict.get("function", {}).get("name")
                )

                if call_name:
                    extracted_tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": call_name,
                            "arguments": args or "{}"
                        }
                    })

        # 6. Extract Usage and Return
        usage = getattr(response, "usage", None)
        usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else (
            usage.dict() if hasattr(usage, "dict") else {})

        final_text = "\n\n".join(collected_text_parts).strip()

        return (
            {
                "role": "assistant",
                "content": final_text if final_text else None,
                "tool_calls": extracted_tool_calls if extracted_tool_calls else None,
            },
            usage_dict,
        )
