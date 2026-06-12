from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from interactive.tool_management import ToolDefinition

from inference.types import ChatMessage


@dataclass
class UploadedFileHandle:
    """
    Provider-agnostic metadata describing an uploaded file artifact.
    """

    provider: str
    file_id: str
    display_name: Optional[str] = None
    mime_type: Optional[str] = None
    raw_response: Any = None


class LLMWrapper:
    provider_label: str = "unknown"

    def make_call(self, input_text: str) -> str:
        raise NotImplementedError

    def make_call_to_python_code(self, input_text: str) -> [Callable[[List[float]], Dict[str, Any]], str]:
        raise NotImplementedError

    def chat_with_tools(
        self,
        messages: List[ChatMessage],
        tools: List[ToolDefinition],
    ) -> ChatMessage:
        """
        Executes a single turn in a tool-calling conversation.

        Args:
            messages: The conversation history in OpenAI format.
                      e.g., [{"role": "user", "content": "..."}]
            tools: A list of tool definitions in OpenAI format.
                   e.g., [{"type": "function", "function": {...}}]

        Returns:
            A single ChatMessage from the assistant, which may contain
            text content, tool_calls, or both.
        """
        raise NotImplementedError

    def make_call_with_info(self, input_text: str) -> Tuple[str, Any]:
        raise NotImplementedError

    def make_call_to_python_code_with_info(
            self, input_text: str, verbose: bool = False
    ) -> Tuple[Callable[[List[float]], Dict[str, Any]], str, Any]:
        raise NotImplementedError

    def make_call_with_files(
        self,
        prompt: str,
        file_handles: List[UploadedFileHandle],
    ) -> Tuple[str, Any]:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement file-augmented calls.")

    def chat_with_tools_with_info(
            self,
            messages: List[ChatMessage],
            tools: List[ToolDefinition],
    ) -> Tuple[ChatMessage, Any]:
        raise NotImplementedError

    def chat_with_tools_with_info_and_files(
            self,
            messages: List[ChatMessage],
            tools: List[ToolDefinition],
            file_handles: Optional[List[UploadedFileHandle]] = None,
    ) -> Tuple[ChatMessage, Any]:
        raise NotImplementedError

    # --- File upload hooks -------------------------------------------------
    @property
    def supports_file_upload(self) -> bool:
        return False

    def upload_file(
        self,
        file_path: str,
        *,
        display_name: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> UploadedFileHandle:
        raise NotImplementedError(f"{self.__class__.__name__} does not support file uploads.")

    def build_file_prompt_part(self, handle: UploadedFileHandle) -> Any:
        raise NotImplementedError(f"{self.__class__.__name__} does not support file uploads.")
