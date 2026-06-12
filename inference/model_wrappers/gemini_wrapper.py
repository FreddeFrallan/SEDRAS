from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None

from interactive.tool_management import ToolDefinition, convert_openai_to_gemini_tools

from inference.model_wrappers.base_wrapper import LLMWrapper, UploadedFileHandle
from ..helpers import _extract_last_code_block, _load_classify_function_safely
from ..prompts import CLASSIFIER_SYSTEM_PROMPT
from ..types import ChatMessage


class GeminiWrapper(LLMWrapper):
    provider_label = "gemini"

    def __init__(
        self,
        model: str = "gemini-1.5-flash-latest",
        api_key: Optional[str] = None,
    ):
        if genai is None:
            raise RuntimeError(
                "google-generativeai SDK not installed. `pip install google-generativeai`"
            )

        self.model_name = model

        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY not set and no api_key provided.")
        genai.configure(api_key=key)

        self.general_model = genai.GenerativeModel(self.model_name)

        self.classifier_model = genai.GenerativeModel(
            self.model_name, system_instruction=CLASSIFIER_SYSTEM_PROMPT
        )

        self.classifier_config = genai.types.GenerationConfig(temperature=0.0)
        self.chat_config = genai.types.GenerationConfig(temperature=0.7)

    @property
    def supports_file_upload(self) -> bool:
        return True

    def make_call(self, input_text: str) -> str:
        try:
            resp = self.general_model.generate_content(input_text)
            return resp.text
        except Exception as e:
            print(f"Gemini API Error (make_call): {e}")
            return f"Error: {e}"

    def _serialize_usage_metadata(self, resp: Any) -> Dict[str, Any]:
        meta = getattr(resp, "usage_metadata", None)
        if meta is None:
            return {}
        if hasattr(meta, "to_dict"):
            try:
                return meta.to_dict()
            except Exception:
                pass
        data = {}
        for attr in ("prompt_token_count", "candidates_token_count", "total_token_count"):
            value = getattr(meta, attr, None)
            if value is not None:
                data[attr] = value
        return data

    def make_call_with_info(self, input_text: str) -> Tuple[str, Dict[str, Any]]:
        try:
            resp = self.general_model.generate_content(input_text)
            return resp.text or "", self._serialize_usage_metadata(resp)
        except Exception as e:
            return f"Error: {e}", {"error": str(e)}

    def make_call_to_python_code(
        self, input_text: str, verbose: bool = False
    ) -> [Callable[[List[float]], Dict[str, Any]], str]:
        try:
            resp = self.classifier_model.generate_content(
                input_text, generation_config=self.classifier_config
            )
            text = resp.text
        except Exception as e:
            raise ValueError(f"Gemini response error: {e}. Full response: {resp}") from e

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
        resp = None
        try:
            gemini_tools = convert_openai_to_gemini_tools(tools)
            gemini_history = self._convert_openai_to_gemini_messages(messages)

            resp = self.general_model.generate_content(
                contents=gemini_history,
                tools=gemini_tools,
                generation_config=self.chat_config,
            )

            return self._convert_gemini_to_openai_response(resp.candidates[0].content)

        except Exception as e:
            print(f"Gemini API Error (chat_with_tools): {e}")
            try:
                if resp and resp.prompt_feedback.block_reason:
                    print(f"Call blocked. Reason: {resp.prompt_feedback.block_reason}")
                    return {
                        "role": "assistant",
                        "content": f"Error: Request was blocked. Reason: {resp.prompt_feedback.block_reason}",
                        "tool_calls": None,
                    }
            except Exception:
                pass

            return {
                "role": "assistant",
                "content": f"Error: {e}",
                "tool_calls": None,
            }

    def chat_with_tools_with_info_and_files(
        self,
        messages: List[ChatMessage],
        tools: List[ToolDefinition],
        file_handles: Optional[List[UploadedFileHandle]] = None,
    ) -> Tuple[ChatMessage, Dict[str, Any]]:
        resp = None
        try:
            gemini_tools = convert_openai_to_gemini_tools(tools)
            if file_handles:
                gemini_history = self._convert_openai_to_gemini_messages_with_files(
                    messages,
                    file_handles,
                )
            else:
                gemini_history = self._convert_openai_to_gemini_messages(messages)

            resp = self.general_model.generate_content(
                contents=gemini_history,
                tools=gemini_tools,
                generation_config=self.chat_config,
            )

            return self._convert_gemini_to_openai_response(resp.candidates[0].content), {}

        except Exception as e:
            print(f"Gemini API Error (chat_with_tools_with_info_and_files): {e}")
            try:
                if resp and resp.prompt_feedback.block_reason:
                    print(f"Call blocked. Reason: {resp.prompt_feedback.block_reason}")
                    return (
                        {
                            "role": "assistant",
                            "content": f"Error: Request was blocked. Reason: {resp.prompt_feedback.block_reason}",
                            "tool_calls": None,
                        },
                        {},
                    )
            except Exception:
                pass

            return {
                "role": "assistant",
                "content": f"Error: {e}",
                "tool_calls": None,
            }, {}

    def _convert_openai_to_gemini_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        gemini_history: List[Dict[str, Any]] = []
        tool_call_id_to_name_map: Dict[str, str] = {}

        for msg in messages:
            role = msg.get("role")

            if role == "user":
                gemini_history.append({"role": "user", "parts": [{"text": msg.get("content", "")}]})

            elif role == "assistant":
                if msg.get("tool_calls"):
                    parts = []
                    for tc in msg["tool_calls"]:
                        func = tc.get("function", {})
                        func_name = func.get("name")
                        if func_name:
                            tool_call_id_to_name_map[tc.get("id")] = func_name
                            try:
                                args_dict = json.loads(func.get("arguments", "{}"))
                            except Exception:
                                args_dict = {}

                            parts.append(
                                {
                                    "function_call": {
                                        "name": func_name,
                                        "args": args_dict,
                                    }
                                }
                            )
                    if parts:
                        gemini_history.append({"role": "model", "parts": parts})
                else:
                    gemini_history.append({"role": "model", "parts": [{"text": msg.get("content", "")}]})

            elif role == "tool":
                tool_call_id = msg.get("tool_call_id")
                func_name = tool_call_id_to_name_map.get(tool_call_id)
                if not func_name:
                    print(f"Warning: Could not find function name for tool_call_id {tool_call_id}")
                    continue

                gemini_history.append(
                    {
                        "role": "function",
                        "parts": [
                            {
                                "function_response": {
                                    "name": func_name,
                                    "response": {"content": msg.get("content")},
                                }
                            }
                        ],
                    }
                )
        return gemini_history

    def _convert_openai_to_gemini_messages_with_files(
        self,
        messages: List[ChatMessage],
        file_handles: List[UploadedFileHandle],
    ) -> List[Dict[str, Any]]:
        gemini_history: List[Dict[str, Any]] = []
        tool_call_id_to_name_map: Dict[str, str] = {}
        files_attached = False

        for msg in messages:
            role = msg.get("role")

            if role == "user":
                parts = [{"text": msg.get("content", "")}]
                if not files_attached:
                    parts.extend(self.build_file_prompt_part(handle) for handle in file_handles)
                    files_attached = True
                gemini_history.append({"role": "user", "parts": parts})

            elif role == "assistant":
                if msg.get("tool_calls"):
                    parts = []
                    for tc in msg["tool_calls"]:
                        func = tc.get("function", {})
                        func_name = func.get("name")
                        if func_name:
                            tool_call_id_to_name_map[tc.get("id")] = func_name
                            try:
                                args_dict = json.loads(func.get("arguments", "{}"))
                            except Exception:
                                args_dict = {}

                            parts.append(
                                {
                                    "function_call": {
                                        "name": func_name,
                                        "args": args_dict,
                                    }
                                }
                            )
                    if parts:
                        gemini_history.append({"role": "model", "parts": parts})
                else:
                    gemini_history.append({"role": "model", "parts": [{"text": msg.get("content", "")}]})

            elif role == "tool":
                tool_call_id = msg.get("tool_call_id")
                func_name = tool_call_id_to_name_map.get(tool_call_id)
                if not func_name:
                    print(f"Warning: Could not find function name for tool_call_id {tool_call_id}")
                    continue

                gemini_history.append(
                    {
                        "role": "function",
                        "parts": [
                            {
                                "function_response": {
                                    "name": func_name,
                                    "response": {"content": msg.get("content")},
                                }
                            }
                        ],
                    }
                )

        return gemini_history

    def _convert_gemini_to_openai_response(self, gemini_content) -> ChatMessage:
        openai_msg: ChatMessage = {
            "role": "assistant",
            "content": None,
            "tool_calls": None,
        }

        text_parts = []
        tool_call_parts = []

        if not gemini_content or not gemini_content.parts:
            return openai_msg

        for part in gemini_content.parts:
            if part.text:
                text_parts.append(part.text)
            elif part.function_call:
                args_dict = dict(part.function_call.args)
                args_str = json.dumps(args_dict)

                call_id = f"call_{part.function_call.name}_{len(tool_call_parts)}"

                tool_call_parts.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": part.function_call.name,
                            "arguments": args_str,
                        },
                    }
                )

        if tool_call_parts:
            openai_msg["tool_calls"] = tool_call_parts

        openai_msg["content"] = "\n".join(text_parts) if text_parts else None

        return openai_msg

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

        upload_kwargs: Dict[str, Any] = {}
        upload_kwargs["mime_type"] = mime_type or "application/pdf"

        # Gemini expects a display name; fall back to the basename for readability.
        friendly_name = display_name or path.name or "uploaded_file"

        try:
            uploaded_obj = genai.upload_file(
                path=str(path),
                display_name=friendly_name,
                **upload_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to upload file to Gemini: {exc}") from exc

        raw_payload = None
        if hasattr(uploaded_obj, "to_dict"):
            try:
                raw_payload = uploaded_obj.to_dict()
            except Exception:
                raw_payload = uploaded_obj
        else:
            raw_payload = uploaded_obj

        # The Gemini SDK returns ids like "files/abc123".
        file_id = getattr(uploaded_obj, "name", None)
        if not file_id:
            raise RuntimeError("Gemini upload response missing file identifier.")

        resolved_display_name = getattr(uploaded_obj, "display_name", friendly_name)
        resolved_mime = getattr(uploaded_obj, "mime_type", upload_kwargs["mime_type"])

        return UploadedFileHandle(
            provider=self.provider_label,
            file_id=file_id,
            display_name=resolved_display_name,
            mime_type=resolved_mime,
            raw_response=raw_payload,
        )

    def build_file_prompt_part(self, handle: UploadedFileHandle) -> Dict[str, Any]:
        if handle.provider != self.provider_label:
            raise ValueError(
                f"Cannot build Gemini file part for provider '{handle.provider}'."
            )
        file_uri = handle.raw_response.get("uri") if isinstance(handle.raw_response, dict) else None
        if not file_uri:
            file_uri = f"https://generativelanguage.googleapis.com/v1beta/{handle.file_id}"
        return {
            "file_data": {
                "file_uri": file_uri,
                **(
                    {"mime_type": handle.mime_type}
                    if handle.mime_type is not None
                    else {}
                ),
            }
        }

    def make_call_with_files(
        self,
        prompt: str,
        file_handles: List[UploadedFileHandle],
    ) -> Tuple[str, Dict[str, Any]]:
        if not file_handles:
            return self.make_call_with_info(prompt)

        parts: List[Dict[str, Any]] = []
        for handle in file_handles:
            parts.append(self.build_file_prompt_part(handle))
        parts.append({"text": prompt})

        try:
            resp = self.general_model.generate_content(
                contents=[{"role": "user", "parts": parts}],
                generation_config=self.chat_config,
            )
            return resp.text or "", self._serialize_usage_metadata(resp)
        except Exception as e:
            return f"Error: {e}", {"error": str(e)}
