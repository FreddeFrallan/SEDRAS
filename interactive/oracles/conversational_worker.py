"""Conversational oracle worker implementation."""
from __future__ import annotations

import json
import re  # <--- ADDED: Required for parsing Markdown code blocks
import copy
from typing import Any, Dict, List, Optional, Callable

from inference.model_wrappers import llm_wrapper
from interactive.oracles.oracle_workers import OracleWorker


class ConversationalWorker(OracleWorker):
    """
    Oracle worker intended for conversational retrieval-style interactions.

    It loads a system prompt from disk (or accepts it directly) at initialization,
    and exposes an `ask_question(question: str) -> str` tool that maintains
    conversation history with the caller.

    The conversation is stored on `self.history` as a list of dicts:
      { "role": "user" | "assistant", "content": str }
    """

    def __init__(
        self,
        dataset_instance,
        llm_model: llm_wrapper.LLMModel,
        llm_system_prompt: Optional[str] = None,
        llm_system_prompt_text: Optional[str] = None,
        verbose: bool = False,
        **kwargs: Any,
    ):
        """
        Args:
            dataset_instance: A DatasetInstance (or similar) this worker will reason over.
            llm_model: LLMModel enum used for the conversational LLM.
            llm_system_prompt: Path to a text file containing the system prompt.
            llm_system_prompt_text: A raw string containing the system prompt text.
            verbose: If True, print debug/log information.
            **kwargs: Extra configuration to be stored on the base class.
        """
        super().__init__(dataset_instance, llm_model, verbose=verbose, **kwargs)

        if llm_system_prompt_text is not None:
            self.system_prompt_text = llm_system_prompt_text
            self.system_prompt_path = None
        elif llm_system_prompt is not None:
            self.system_prompt_path = llm_system_prompt
            try:
                with open(llm_system_prompt, "r", encoding="utf-8") as f:
                    self.system_prompt_text = f.read()
            except Exception as e:
                # Fail early with a clear error if the prompt cannot be loaded
                raise RuntimeError(
                    f"Failed to load system prompt from '{llm_system_prompt}': {e}"
                ) from e
        else:
            raise ValueError(
                "ConversationalWorker requires either 'llm_system_prompt' (path) "
                "or 'llm_system_prompt_text' (raw text)."
            )

        # Conversation history: list of {"role": "user"|"assistant", "content": str}
        self.history: List[Dict[str, str]] = []
        # Track sample identifiers shared with the user across turns (dataset indices)
        self.shared_sample_ids: List[int] = []

        # Hard-copy the samples at initialization so later mutations to the dataset
        # do not affect what we consider "shown" during this session.
        original_samples = getattr(self.dataset_instance, "samples", [])
        self._snapshot_samples: List[Any] = [
            copy.deepcopy(s) for s in original_samples
        ]

    def _build_history_prompt(self) -> str:
        parts: List[str] = []
        parts.append("[SYSTEM]")
        parts.append(self.system_prompt_text.strip())
        parts.append("")

        for msg in self.history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                parts.append("[USER]")
            else:
                parts.append("[ASSISTANT]")
            parts.append(content.strip())
            parts.append("")

        full_prompt = "\n".join(parts)
        return full_prompt

    def _normalize_sample_id(self, sample_id: Any) -> Optional[int]:
        """Convert a provided sample identifier to an integer index if possible."""

        try:
            idx = int(sample_id)
        except (TypeError, ValueError):
            return None

        if idx < 0:
            return None
        return idx

    def _parse_oracle_response(self, raw_response: str) -> Dict[str, Any]:
        """
        Parse the oracle response expected to be JSON with fields:
          - return_msg: str
          - shared_sample_ids: list

        Handles raw JSON strings as well as JSON strings wrapped in Markdown
        code blocks (e.g., ```json ... ```). Falls back gracefully if parsing fails.
        """
        parsed = None

        # 1. Attempt direct JSON parsing
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            # 2. If direct parsing fails, check for Markdown code blocks.
            # Regex matches ``` followed optionally by 'json', creates a group for content, and ends with ```
            # re.DOTALL ensures '.' matches newlines.
            match = re.search(r"```(?:json)?\s*(.*?)```", raw_response, re.DOTALL | re.IGNORECASE)
            if match:
                try:
                    # Extract the content inside the code block
                    json_content = match.group(1).strip()
                    parsed = json.loads(json_content)
                except json.JSONDecodeError:
                    parsed = None

        # 3. If parsing is still successful, ensure it's a dict
        if parsed is not None and not isinstance(parsed, dict):
            # Valid JSON found, but it wasn't a dictionary (e.g., a list or string)
            parsed = None

        # 4. Fallback: If parsed is None, treat the raw response as a plain string message
        if parsed is None:
            return {"return_msg": raw_response, "shared_sample_ids": []}

        # 5. Extract fields from the valid dict
        return_msg = parsed.get("return_msg", raw_response)
        shared_sample_ids_raw = parsed.get("shared_sample_ids", [])
        if not isinstance(shared_sample_ids_raw, list):
            shared_sample_ids_raw = []

        shared_sample_ids: List[int] = []
        for sample_id in shared_sample_ids_raw:
            normalized = self._normalize_sample_id(sample_id)
            if normalized is not None:
                shared_sample_ids.append(normalized)

        # Ensure return_msg is a string, even if the JSON had complex types
        if not isinstance(return_msg, str):
            return_msg = str(return_msg)

        return {"return_msg": return_msg, "shared_sample_ids": shared_sample_ids}

    def ask_question(self, question: str) -> str:
        """
        Ask a question in an ongoing conversation with the worker's LLM.

        The worker maintains a conversation history (user/assistant turns).
        Each call to this function:
          - appends the user question to history,
          - builds a prompt from the system prompt + full history,
          - calls the underlying LLM (which responds with JSON containing
            'return_msg' and 'shared_sample_ids'),
          - appends the assistant answer to history,
          - and returns the 'return_msg' field as plain text.
        """
        # Record the new user turn
        self.history.append({"role": "user", "content": question})

        # Build conversational prompt as a single text input for make_call
        full_prompt = self._build_history_prompt()

        if self.verbose:
            print("---- ConversationalRetrievalWorker.ask_question ----")
            print("Prompt sent to LLM:")
            print(full_prompt)
            print("-----------------------------------------------------")

        # Call the LLM
        answer = self.llm.make_call(full_prompt)
        if answer is None:
            answer_str = ""
        else:
            answer_str = str(answer).strip()

        parsed_response = self._parse_oracle_response(answer_str)
        return_msg = parsed_response.get("return_msg", "")
        shared_sample_ids = parsed_response.get("shared_sample_ids", [])

        # Record any shared sample identifiers while preserving order
        for sample_id in shared_sample_ids:
            if self.verbose:
                print(f"Processing shared sample ID: {sample_id}")
            if sample_id not in self.shared_sample_ids:
                self.shared_sample_ids.append(sample_id)
                if self.verbose:
                    print("Shared sample ID recorded:", sample_id)

        # Record the assistant's response (human-facing message only)
        self.history.append({"role": "assistant", "content": return_msg})

        if self.verbose:
            print("LLM answer:")
            print(answer_str)
            print("-----------------------------------------------------")

        return return_msg

    def get_callable_functions(self) -> List[Callable[..., Any]]:
        """
        Return the list of callables to expose as tools to the LLM.

        For now, this worker exposes only a single conversational tool:
          - ask_question(question: str) -> str
        """
        return [self.ask_question]

    def _json_safe_assignment(self, assignment: dict[str, Any]) -> Dict[str, Any]:
        """Return a JSON-serializable copy of the assignment (mirrors DirectExperiment)."""

        def _convert(value: Any) -> Any:
            # Avoid numpy dependency; handle simple nested structures
            if isinstance(value, (list, tuple)):
                return [_convert(v) for v in value]
            if isinstance(value, dict):
                return {str(k): _convert(v) for k, v in value.items()}
            return value

        if not isinstance(assignment, dict):
            return {"value": _convert(assignment)}

        return {str(k): _convert(v) for k, v in assignment.items()}

    def get_session_data(self) -> Dict[str, Any]:
        """Return metadata about which samples were surfaced during the session.

        Uses a hard-copied snapshot of the samples taken at initialization, so that
        later mutations to `dataset_instance.samples` do not affect the returned data.
        """

        assignments: List[Dict[str, Any]] = []

        for sample_id in self.shared_sample_ids:
            if sample_id < 0 or sample_id >= len(self._snapshot_samples):
                continue

            sample = self._snapshot_samples[sample_id]
            # Expecting DatasetSample-like objects with .assignment and .label
            assignment_dict = getattr(sample, "assignment", None)
            label = getattr(sample, "label", None)

            # If for some reason the snapshot doesn't have these attrs,
            # skip rather than crash.
            if assignment_dict is None or label is None:
                continue

            property_labels = getattr(sample, "property_labels", None) or {}
            label_name = None if label is None else self.label_names.get(label, f"Label {label}")

            assignments.append(
                {
                    "input_values": self._json_safe_assignment(assignment_dict),
                    "label": label,
                    "label_name": label_name,
                    "property_labels": property_labels,
                }
            )

        return {"assignments": assignments}