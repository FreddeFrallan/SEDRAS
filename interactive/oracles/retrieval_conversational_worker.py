"""Oracle worker that supports tool-calling to fetch sample text on demand."""
from __future__ import annotations

import json
from typing import Any, Callable, List, Optional

from inference.model_wrappers import llm_wrapper
from interactive.interactive_utils import _json_dumps_safe
from interactive.oracles.conversational_worker import ConversationalWorker
from interactive.tool_management import generate_tool_definition, parse_tool_call


class RetrievalConversationalWorker(ConversationalWorker):
    """
    Conversational oracle that keeps assignments/labels in its system prompt and
    exposes a tool to fetch textual descriptions for specific samples.
    """

    def __init__(
        self,
        dataset_instance,
        llm_model: llm_wrapper.LLMModel,
        llm_system_prompt: Optional[str] = None,
        llm_system_prompt_text: Optional[str] = None,
        verbose: bool = False,
        max_internal_turns: int = 5,
        **kwargs: Any,
    ):
        logged_out_grace_turns = kwargs.pop("logged_out_grace_turns", 3)

        super().__init__(
            dataset_instance,
            llm_model,
            llm_system_prompt=llm_system_prompt,
            llm_system_prompt_text=llm_system_prompt_text,
            verbose=verbose,
            **kwargs,
        )
        self.max_internal_turns = max(1, int(max_internal_turns))
        self.logged_out_grace_turns = max(0, int(logged_out_grace_turns))
        self._tool_definitions: Optional[List[llm_wrapper.ToolDefinition]] = None
        self.message_history: List[llm_wrapper.ChatMessage] = [
            {"role": "system", "content": self.system_prompt_text.strip()}
        ]
        self._ui_event_buffer: List[str] = []
        self._pending_status_updates: List[str] = []
        self._is_client_connected = True
        self._grace_turns_remaining = 0

        # If verbose, print the system prompt
        if self.verbose:
            print("System prompt:")
            print(self.system_prompt_text)
            print("-----------------------------------------------------")

    def set_client_connected(self, connected: bool) -> None:
        """Set client connectivity state with a grace period for status syncing."""

        if connected:
            was_disconnected = not self._is_client_connected
            self._is_client_connected = True
            self._grace_turns_remaining = 0

            if was_disconnected:
                self._ui_event_buffer.append(
                    "You have reconnected. Your session is now synced with the latest status updates."
                )
                self._ui_event_buffer.extend(self._pending_status_updates)
                self._pending_status_updates.clear()
            return

        self._is_client_connected = False
        self._grace_turns_remaining = self.logged_out_grace_turns

    def pop_ui_events(self) -> List[str]:
        """Return and clear queued UI events for the client."""

        events = list(self._ui_event_buffer)
        self._ui_event_buffer.clear()
        return events

    def _publish_status_update(self, message: str) -> None:
        if not message:
            return

        if self._is_client_connected:
            self._ui_event_buffer.append(message)
            return

        if self._grace_turns_remaining > 0:
            self._pending_status_updates.append(message)

    def _consume_grace_turn(self) -> None:
        if self._is_client_connected or self._grace_turns_remaining <= 0:
            return

        self._grace_turns_remaining -= 1
        if self._grace_turns_remaining <= 0:
            self._pending_status_updates.clear()

    def fetch_sample_text(self, sample_id: int) -> str:
        """Return the textual description for a dataset sample if available."""

        idx = self._normalize_sample_id(sample_id)
        if idx is None:
            return "Invalid sample_id. Please provide a non-negative integer."
        if idx >= len(self.dataset_instance.samples):
            return (
                "Unknown sample_id. Please choose an ID from the reference samples "
                f"(0-{len(self.dataset_instance.samples) - 1})."
            )

        sample = self.dataset_instance.samples[idx]
        if sample.instance_text:
            return str(sample.instance_text)

        return (
            "No textual description is available for this sample. You may rely on "
            "the assignment/label information."
        )

    def _get_tool_definitions(self) -> List[llm_wrapper.ToolDefinition]:
        if self._tool_definitions is None:
            self._tool_definitions = [generate_tool_definition(self.fetch_sample_text)]
        return self._tool_definitions

    def ask_question(self, question: str) -> str:
        """
        Ask a question using a tool-enabled conversational loop.

        The LLM may call `fetch_sample_text` to retrieve textual details before
        producing its final JSON response containing `return_msg` and
        `shared_sample_ids`.
        """

        # Record the new user turn for downstream session data
        self.history.append({"role": "user", "content": question})
        self.message_history.append({"role": "user", "content": question})
        self._publish_status_update("Question received. Processing your request.")

        tools = self._get_tool_definitions()

        for _ in range(self.max_internal_turns):
            response = self.llm.chat_with_tools(self.message_history, tools)
            tool_calls = response.get("tool_calls") or []
            content = (response.get("content") or "").strip()

            if not tool_calls:
                # Final answer path
                self.message_history.append(response)
                answer_str = content
                parsed_response = self._parse_oracle_response(answer_str)
                return_msg = parsed_response.get("return_msg", "")
                shared_sample_ids = parsed_response.get("shared_sample_ids", [])

                for sample_id in shared_sample_ids:
                    if sample_id not in self.shared_sample_ids:
                        self.shared_sample_ids.append(sample_id)

                self.history.append({"role": "assistant", "content": return_msg})
                self._publish_status_update(return_msg)

                if self.verbose:
                    print("LLM answer:")
                    print(answer_str)
                    print("-----------------------------------------------------")

                self._consume_grace_turn()
                return return_msg

            # Handle tool calls before continuing the loop
            self.message_history.append(response)

            for tool_call in tool_calls:
                function_name, function_args, parse_error = parse_tool_call(tool_call)

                if parse_error:
                    tool_content = parse_error
                else:
                    function_to_call = getattr(self, function_name, None)
                    if not function_to_call:
                        tool_content = f"Error: Function '{function_name}' not found."
                    else:
                        try:
                            result = function_to_call(**function_args)
                            tool_content = (
                                result if isinstance(result, str) else _json_dumps_safe(result)
                            )
                        except Exception as e:  # pragma: no cover - defensive
                            tool_content = f"Error executing function {function_name}: {e}"

                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", ""),
                    "name": tool_call.get("function", {}).get("name", ""),
                    "content": tool_content,
                }
                self.message_history.append(tool_message)
                self._publish_status_update(
                    f"Processed tool call: {tool_message.get('name') or 'unknown_tool'}."
                )

                if self.verbose:
                    print("Tool call processed:")
                    try:
                        print(json.dumps(tool_message, indent=2, ensure_ascii=False))
                    except Exception:
                        print(str(tool_message))
                    print("-----------------------------------------------------")

        # Fallback if we exit the loop without a final response
        fallback_msg = (
            "Unable to produce a response after processing tool calls. Please try again."
        )
        self.history.append({"role": "assistant", "content": fallback_msg})
        self._publish_status_update(fallback_msg)
        self._consume_grace_turn()
        return fallback_msg

    def get_callable_functions(self) -> List[Callable[..., Any]]:
        """Expose only the user-facing conversational tool."""

        return [self.ask_question]
