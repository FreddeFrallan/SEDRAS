from __future__ import annotations

import json
import os
import base64
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from interactive.tool_management import ToolDefinition
from inference.model_wrappers.base_wrapper import LLMWrapper, UploadedFileHandle
from inference.helpers import _extract_last_code_block, _load_classify_function_safely
from inference.prompts import CLASSIFIER_SYSTEM_PROMPT
from inference.types import ChatMessage


class RemoteHTTPWrapper(LLMWrapper):
    """
    LLM wrapper that forwards evaluation calls to a user-provided HTTP endpoint.

    The remote endpoint is expected to accept a JSON POST body with OpenAI-like
    fields: ``model``, ``messages``, and optionally ``tools``/``tool_choice``.
    For simple prototype endpoints, the request also includes ``message`` with
    the latest user text.
    """

    provider_label = "remote_http"

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.endpoint_url = (
            endpoint_url
            or os.getenv("REMOTE_HTTP_LLM_ENDPOINT_URL")
            or os.getenv("CUSTOM_HTTP_ENDPOINT_URL")
        )
        if not self.endpoint_url:
            raise ValueError(
                "Remote HTTP endpoint URL is required. Pass endpoint_url or set "
                "REMOTE_HTTP_LLM_ENDPOINT_URL."
            )

        self.model = (
            model
            or os.getenv("REMOTE_HTTP_LLM_MODEL")
            or os.getenv("CUSTOM_HTTP_MODEL")
            or "remote-http-model"
        )
        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv("REMOTE_HTTP_LLM_API_KEY") or os.getenv("CUSTOM_HTTP_API_KEY")
        )
        self.timeout = float(
            timeout
            if timeout is not None
            else os.getenv("REMOTE_HTTP_LLM_TIMEOUT", "360")
        )

    def _latest_user_message(self, messages: List[ChatMessage]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                return content if isinstance(content, str) else json.dumps(content)
        return ""

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = Request(
            self.endpoint_url,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Remote HTTP LLM endpoint returned {error.code}: {detail}"
            ) from error
        except URLError as error:
            raise RuntimeError(f"Failed to reach remote HTTP LLM endpoint: {error}") from error

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Remote HTTP LLM endpoint returned invalid JSON: {response_body[:500]}"
            ) from error

        if not isinstance(parsed, dict):
            raise RuntimeError("Remote HTTP LLM endpoint must return a JSON object.")
        return parsed

    def _completion(
        self,
        messages: List[ChatMessage],
        *,
        tools: Optional[List[ToolDefinition]] = None,
        file_handles: Optional[List[UploadedFileHandle]] = None,
    ) -> Tuple[ChatMessage, Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "message": self._latest_user_message(messages),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if file_handles:
            payload["files"] = [self.build_file_prompt_part(handle) for handle in file_handles]

        response = self._post(payload)
        return self._extract_message(response), self._extract_usage(response)

    def _extract_usage(self, response: Dict[str, Any]) -> Dict[str, Any]:
        usage = response.get("usage")
        return usage if isinstance(usage, dict) else {}

    def _extract_message(self, response: Dict[str, Any]) -> ChatMessage:
        if isinstance(response.get("choices"), list) and response["choices"]:
            first_choice = response["choices"][0]
            if isinstance(first_choice, dict):
                message = first_choice.get("message")
                if isinstance(message, dict):
                    return self._normalize_message(message)
                text = first_choice.get("text")
                if isinstance(text, str):
                    return {"role": "assistant", "content": text, "tool_calls": None}

        message = response.get("message")
        if isinstance(message, dict):
            return self._normalize_message(message)
        if isinstance(message, str):
            return {"role": "assistant", "content": message, "tool_calls": None}

        for key in ("reply", "content", "text", "response", "output_text"):
            value = response.get(key)
            if isinstance(value, str):
                return {
                    "role": "assistant",
                    "content": value,
                    "tool_calls": self._normalize_tool_calls(response.get("tool_calls")),
                }

        return {
            "role": "assistant",
            "content": None,
            "tool_calls": self._normalize_tool_calls(response.get("tool_calls")),
        }

    def _normalize_message(self, message: Dict[str, Any]) -> ChatMessage:
        return {
            "role": message.get("role", "assistant"),
            "content": message.get("content"),
            "tool_calls": self._normalize_tool_calls(message.get("tool_calls")),
        }

    def _normalize_tool_calls(self, tool_calls: Any) -> Optional[List[Dict[str, Any]]]:
        if not tool_calls:
            return None
        if not isinstance(tool_calls, list):
            raise RuntimeError("Remote HTTP LLM response field 'tool_calls' must be a list.")

        normalized: List[Dict[str, Any]] = []
        for idx, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue

            function = tool_call.get("function")
            if isinstance(function, dict):
                name = function.get("name")
                arguments = function.get("arguments", "{}")
            else:
                name = tool_call.get("name") or tool_call.get("tool_name")
                arguments = tool_call.get("arguments", "{}")

            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)

            normalized.append(
                {
                    "id": tool_call.get("id") or tool_call.get("call_id") or f"call_{idx}",
                    "type": tool_call.get("type", "function"),
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    },
                }
            )

        return normalized or None

    def make_call(self, input_text: str) -> str:
        content, _ = self.make_call_with_info(input_text)
        return content

    def make_call_with_info(self, input_text: str) -> Tuple[str, Dict[str, Any]]:
        message, usage = self._completion([{"role": "user", "content": input_text}])
        return message.get("content") or "", usage

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

        return UploadedFileHandle(
            provider=self.provider_label,
            file_id=str(path.resolve()),
            display_name=display_name or path.name,
            mime_type=mime_type or "application/octet-stream",
            raw_response={"path": str(path.resolve())},
        )

    def build_file_prompt_part(self, handle: UploadedFileHandle) -> Dict[str, Any]:
        if handle.provider != self.provider_label:
            raise ValueError(
                f"Cannot build remote HTTP file payload for provider '{handle.provider}'."
            )

        path = Path(handle.file_id)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {handle.file_id}")

        return {
            "file_id": handle.file_id,
            "display_name": handle.display_name or path.name,
            "mime_type": handle.mime_type or "application/octet-stream",
            "content_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
        }

    def make_call_with_files(
        self,
        prompt: str,
        file_handles: List[UploadedFileHandle],
    ) -> Tuple[str, Dict[str, Any]]:
        if not file_handles:
            return self.make_call_with_info(prompt)
        message, usage = self._completion(
            [{"role": "user", "content": prompt}],
            file_handles=file_handles,
        )
        return message.get("content") or "", usage

    def make_call_to_python_code(self, input_text: str, verbose: bool = False):
        func, code, _ = self.make_call_to_python_code_with_info(input_text, verbose)
        return func, code

    def make_call_to_python_code_with_info(
        self,
        input_text: str,
        verbose: bool = False,
    ) -> Tuple[Callable[[List[float]], Dict[str, Any]], str, Dict[str, Any]]:
        message, usage = self._completion(
            [
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": input_text},
            ]
        )
        text = message.get("content") or ""
        if verbose:
            print(f"\nREMOTE HTTP RAW RESPONSE:\n{text.strip()}\n")

        code = _extract_last_code_block(text)
        if not code:
            raise ValueError("No code block found in LLM response.")
        return _load_classify_function_safely(code), code, usage

    def chat_with_tools(
        self,
        messages: List[ChatMessage],
        tools: List[ToolDefinition],
    ) -> ChatMessage:
        message, _ = self.chat_with_tools_with_info(messages, tools)
        return message

    def chat_with_tools_with_info(
        self,
        messages: List[ChatMessage],
        tools: List[ToolDefinition],
    ) -> Tuple[ChatMessage, Dict[str, Any]]:
        return self._completion(messages, tools=tools)

    def chat_with_tools_with_info_and_files(
        self,
        messages: List[ChatMessage],
        tools: List[ToolDefinition],
        file_handles: Optional[List[UploadedFileHandle]] = None,
    ) -> Tuple[ChatMessage, Dict[str, Any]]:
        return self._completion(messages, tools=tools, file_handles=file_handles)


CustomHTTPBackendWrapper = RemoteHTTPWrapper
