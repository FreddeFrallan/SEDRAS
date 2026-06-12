from __future__ import annotations

import json
from typing import List, Dict, Callable, Any, Optional

from inference.model_wrappers import llm_wrapper
from inference.model_wrappers.llm_wrapper import UploadedFileHandle
from interactive.tool_management import generate_tool_definition, parse_tool_call
from data_management import dataset
from configs.interactive_config import InteractiveConfig
from interactive.interactive_utils import _json_dumps_safe, _normalize_assignment
from prompts.task_descriptions import interactive_direct_experiments_prompt, interactive_single_conversational_prompt
from prompts.task_descriptions import (
    interactive_direct_experiments_and_evaluation_prompt,
    interactive_symbolic_regression_prompt,
)
from interactive import init_oracle
from interactive.experiment_costs import get_experiment_cost


def _normalize_run_experiment_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Per-function normalizer for run_experiment:
    - Ensure 'assignment' is normalized using _normalize_assignment.
    """
    if "assignment" in args:
        args = dict(args)  # shallow copy
        args["assignment"] = _normalize_assignment(args["assignment"])
    return args


def run_interactive_theory_induction_session(
    dataset_instance: dataset.DatasetInstance,
    interactive_config: InteractiveConfig,
    *,
    llm_model: llm_wrapper.LLMModel = llm_wrapper.LLMModel.GEMINI_2_5_PRO,
    max_turns: int = 20,
    file_handles: Optional[List[UploadedFileHandle]] = None,
) -> Dict[str, Any]:
    """
    Run an interactive, tool-calling based theory-induction session on a single DatasetInstance.

    Flow:
      - Build an OracleWorker over the dataset instance.
      - Expose its tools (run_experiment, get_variable_mapping) via function calling.
      - Drive an LLM (llm_model) that uses those tools to explore and eventually propose a theory.
      - Optionally attach uploaded files when calling the LLM.
      - Return the final assistant message plus the full message history.

    Returns:
        {
          "final_message": {...},     # last assistant message
          "messages": [...],          # full conversation history
          "num_turns": int,
          "num_tool_calls": int,
          "terminated_reason": "no_tool_calls" | "max_turns_reached" | "llm_error",
        }
    """
    # --- 1) Build Oracle on this dataset instance ---
    oracle = init_oracle.init_oracle_worker(
        dataset_instance,
        interactive_config,
        llm_model=llm_model
    )
    session_settings = oracle.get_session_settings() if hasattr(oracle, "get_session_settings") else {}

    # 2) Tools
    my_functions = oracle.get_callable_functions()
    tools = [generate_tool_definition(func) for func in my_functions]

    available_functions: Dict[str, Callable[..., Any]] = {
        func.__name__: func for func in my_functions
    }

    # 3) Initialize main LLM wrapper for interactive reasoning
    if interactive_config.verbose:
        print(f"Initializing wrapper for model: {llm_model.name}...")

    # Backend selection for the main interactive LLM mirrors the oracle's backend.
    backend = None
    if interactive_config.oracle_config is not None:
        backend = init_oracle.select_backend_for_representation(
            dataset.RepresentationLevel.parse(
                getattr(dataset_instance, "representation_level", dataset.RepresentationLevel.UNKNOWN)
            ),
            interactive_config.oracle_config.backend,
        )

    if interactive_config.verbose and backend is not None:
        print(f"Using backend: {backend.name}")

    wrapper = llm_wrapper.get_llm_wrapper(llm_model, backend=backend)

    # 4) Starting prompt
    if interactive_config.mode == InteractiveConfig().mode.DIRECT_EXPERIMENT:
        raise NotImplementedError("Conversational modes not implemented yet.")
        start_prompt = interactive_direct_experiments_prompt.build_interactive_prompt(
            dataset_instance,
            interactive_config,
        )
    elif (
        interactive_config.mode
        == InteractiveConfig().mode.DIRECT_EXPERIMENT_AND_EXPLICIT_FINISH
    ):
        raise NotImplementedError("Conversational modes not implemented yet.")
        start_prompt = interactive_direct_experiments_prompt.build_interactive_prompt(
            dataset_instance,
            interactive_config,
        )
    elif (
        interactive_config.mode
        == InteractiveConfig().mode.DIRECT_EXPERIMENT_AND_EVALUATION
    ):
        start_prompt = (
            interactive_direct_experiments_and_evaluation_prompt.build_interactive_prompt(
                dataset_instance,
                interactive_config,
            )
        )
    elif interactive_config.mode in (
        InteractiveConfig().mode.SINGLE_CONVERSATIONAL,
        InteractiveConfig().mode.RETRIEVAL_CONVERSATIONAL,
    ):
        raise NotImplementedError("Conversational modes not implemented yet.")
        start_prompt = interactive_single_conversational_prompt.build_interactive_prompt(
            dataset_instance,
            interactive_config,
        )
    elif interactive_config.mode == InteractiveConfig().mode.SYMBOLIC_REGRESSION:
        start_prompt = interactive_symbolic_regression_prompt.build_direct_and_eval_symbolic_regression_prompt(
            dataset_instance,
            interactive_config,
        )
    else:
        raise NotImplementedError(
            f"Interactive mode '{interactive_config.mode}' not implemented."
        )


    # print(start_prompt)
    # input(f"Press Enter to start the interactive session...")

    # print(start_prompt)
    messages: List[llm_wrapper.ChatMessage] = [
        {"role": "user", "content": start_prompt}
    ]

    if interactive_config.verbose:
        print("\n[Interactive session start]")
        print("---------------------------------")
        print(f"User: {messages[0]['content']}\n")

    num_tool_calls = 0
    final_message: Optional[Dict[str, Any]] = None
    terminated_reason = "max_turns_reached"
    experimental_budget = (
        None if interactive_config.max_experiments is None else int(interactive_config.max_experiments)
    )

    # Per-function argument normalizers (for parse_tool_call)
    arg_normalizers = {}
    if interactive_config.mode in (
        InteractiveConfig().mode.DIRECT_EXPERIMENT,
        InteractiveConfig().mode.DIRECT_EXPERIMENT_AND_EXPLICIT_FINISH,
        InteractiveConfig().mode.DIRECT_EXPERIMENT_AND_EVALUATION,
        InteractiveConfig().mode.SYMBOLIC_REGRESSION,
    ):
        arg_normalizers["run_experiment"] = _normalize_run_experiment_args

    def _experiments_left() -> Optional[int]:
        if experimental_budget is None:
            return None
        return max(experimental_budget, 0)

    def _build_status_message(turns_left: int, experiments_left: Optional[int]) -> str:
        experiments_str = "unlimited" if experiments_left is None else str(experiments_left)
        status = (
            f"[Session status] Turns left: {turns_left}; "
            f"Experiments left: {experiments_str}."
        )
        if turns_left == 1:
            status += (
                " This is the final turn; your answer to this message will be "
                "interpreted as the final theory."
            )
        return status

    # To avoid infinite retries on backend/JSON errors
    backend_error_retries = 0
    max_backend_error_retries = 3

    # 5) Conversation loop
    usage_objects = []

    for turn_idx in range(max_turns):
        if interactive_config.verbose:
            print(f"--- Turn {turn_idx + 1} ---")

        session_finished = False
        turns_left = max_turns - turn_idx
        experiments_left = _experiments_left()

        status_message = _build_status_message(turns_left, experiments_left)
        messages.append({"role": "user", "content": status_message})

        if interactive_config.verbose:
            print("Appended status message for LLM:")
            print(status_message)
            print()

        # Ask the model with tools
        # response_message = wrapper.chat_with_tools(messages, tools)
        response_message, usage = wrapper.chat_with_tools_with_info_and_files(
            messages,
            tools,
            file_handles=file_handles,
        )
        usage_objects.append(usage)

        if interactive_config.verbose:
            print("LLM intermediate response (content):")
            print(response_message.get("content") or "[No content]")
            print()

        messages.append(response_message)
        final_message = response_message

        tool_calls = response_message.get("tool_calls")
        content = (response_message.get("content") or "").strip()

        # --- NEW: handle backend / wrapper errors as "tool-style" feedback ---
        # If the wrapper surfaced something like:
        #   "Error: Object of type MapComposite is not JSON serializable"
        # with no tool_calls, we treat it as a recoverable error:
        if content.startswith("Error:") and not tool_calls:
            backend_error_retries += 1
            if interactive_config.verbose:
                print(
                    f"Backend/LLM error detected (attempt {backend_error_retries}): {content}"
                )

            if backend_error_retries > max_backend_error_retries:
                # Give up after a few attempts
                terminated_reason = "llm_error"
                if interactive_config.verbose:
                    print("Exceeded max backend error retries; ending session.")
                break

            # Feed a tool-style error message back to the model,
            # telling it to fix its JSON / arguments and try again.
            error_tool_message = {
                "role": "tool",
                "tool_call_id": f"backend_error_retry_{backend_error_retries}",
                "name": "tool_call_error",
                "content": (
                    "Your previous response caused an internal error on the backend:\n"
                    f"{content}\n\n"
                    "This usually means that some of your tool call arguments were not valid JSON or "
                    "contained values that could not be serialized. Please try again, making sure that:\n"
                    "- The `arguments` field for each tool call is valid JSON.\n"
                    "- All values are JSON-serializable (no custom objects).\n"
                    "- Arrays and objects are well-formed.\n"
                    "Return a corrected tool call instead of plain text."
                ),
            }
            messages.append(error_tool_message)

            if interactive_config.verbose:
                print("Appended corrective tool message and continuing to next turn.\n")

            # Continue to next turn without treating this as final answer
            continue

        # --- Normal termination: no tool calls and no backend error pattern ---
        if not tool_calls:
            end_on_no_tool_calls = session_settings.get("end_on_no_tool_calls", True)
            continue_message = session_settings.get(
                "no_tool_calls_continue_message",
                session_settings.get("no_too_calls_continue_message"),
            )

            if end_on_no_tool_calls:
                terminated_reason = "no_tool_calls"
                if interactive_config.verbose:
                    print("Assistant (Final Answer):")
                    print(response_message.get("content") or "[No text content]")
                    print("\n---------------------------------")
                    print("Conversation finished (no further tool calls).")
                break

            if continue_message:
                messages.append({"role": "user", "content": continue_message})
                if interactive_config.verbose:
                    print("No tool calls; appending continue message and continuing session.")
                    print(continue_message)

            continue

        # 5a) Handle tool calls
        if interactive_config.verbose:
            print("Assistant: (wants to call tools)")
            print("Tool calls requested:")
            try:
                print(json.dumps(tool_calls, indent=2, ensure_ascii=False))
            except Exception:
                print(str(tool_calls))
            print()

        num_tool_calls += len(tool_calls)

        # Map tool_call_id -> function name for pairing in tool responses
        name_by_id: Dict[str, str] = {}
        for tc in tool_calls:
            fn = tc.get("function", {}) or {}
            name_by_id[tc.get("id", "")] = fn.get("name", "")

        for tool_call in tool_calls:
            # Centralized parsing & normalization (robust, never raises)
            function_name, function_args, parse_error = parse_tool_call(
                tool_call,
                arg_normalizers=arg_normalizers,
            )

            if parse_error:
                # Tell the model exactly what went wrong so it can fix its tool call
                err_msg = (
                    f"Error while parsing arguments for tool call "
                    f"'{function_name or 'unknown_function'}': {parse_error}"
                )
                if interactive_config.verbose:
                    print(err_msg)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", ""),
                    "name": function_name or "unknown_function",
                    "content": err_msg,
                })
                # Skip function execution for this tool call
                continue

            function_to_call = available_functions.get(function_name)

            if interactive_config.verbose:
                print(f"  -> Model wants to call: {function_name}(...)")

            if not function_to_call:
                # Unknown function: report error back as tool result
                err_msg = f"Error: Function '{function_name}' not found."
                if interactive_config.verbose:
                    print(err_msg)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", ""),
                    "name": function_name or "unknown_function",
                    "content": err_msg,
                })
                continue

            tool_cost = get_experiment_cost(function_name)
            if experimental_budget is not None and tool_cost:
                if tool_cost > experimental_budget:
                    err_msg = (
                        "Error: Experiment budget exceeded for tool "
                        f"'{function_name}'. Cost: {tool_cost}, "
                        f"budget remaining: {experimental_budget}."
                    )
                    if interactive_config.verbose:
                        print(err_msg)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", ""),
                        "name": function_name or "unknown_function",
                        "content": err_msg,
                    })
                    continue

                experimental_budget -= tool_cost

            try:
                if interactive_config.verbose:
                    print("  Parsed tool arguments (normalized):")
                    try:
                        print(json.dumps(function_args, indent=2, ensure_ascii=False))
                    except Exception:
                        print(str(function_args))

                # Execute the tool
                result = function_to_call(**function_args)

                if interactive_config.verbose:
                    print("  Tool result (pre-serialization):")
                    try:
                        print(json.dumps(result, indent=2, ensure_ascii=False))
                    except Exception:
                        print(str(result))

                # Ensure tool 'content' is a STRING
                if isinstance(result, str):
                    tool_content = result
                else:
                    tool_content = _json_dumps_safe(result)

                # Append as a tool message (include the function name!)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", ""),
                    "name": name_by_id.get(tool_call.get("id", ""), function_name or ""),
                    "content": tool_content,
                }
                messages.append(tool_message)

                if interactive_config.verbose:
                    print("  Tool message appended to history:")
                    print(json.dumps(tool_message, indent=2, ensure_ascii=False))
                    print()

                if function_name == "submit_theory":
                    if isinstance(result, dict) and "accepted" in result:
                        accepted = bool(result.get("accepted"))
                    else:
                        accepted = bool(result)

                    if accepted is True:
                        terminated_reason = "theory_submitted"
                        session_finished = True
                        break

            except Exception as e:
                err = f"Error executing function {function_name}: {e}"
                if interactive_config.verbose:
                    print(err)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", ""),
                    "name": function_name or "unknown_function",
                    "content": err,
                })

        # End turn loop; continue to next turn or exit if theory submitted
        if session_finished:
            break

    if final_message is None:
        # Should not really happen, but guard anyway
        final_message = {
            "role": "assistant",
            "content": "[No response produced]",
        }

    # If we broke due to backend retries being exceeded, we already set terminated_reason
    # Otherwise, terminated_reason is either "no_tool_calls" or "max_turns_reached".
    if interactive_config.verbose and terminated_reason in ("max_turns_reached", "llm_error"):
        print("\n---------------------------------")
        if terminated_reason == "llm_error":
            print("Interactive session ended due to repeated LLM/backend errors.")
        else:
            print("Reached max turns. Ending conversation.")

    session_data = oracle.get_session_data()
    print("Session data collected by oracle:")
    print(session_data)

    return {
        "final_message": final_message,
        "messages": messages,
        "num_turns": min(
            max_turns,
            len([m for m in messages if m.get("role") == "assistant"]),
        ),
        "num_tool_calls": num_tool_calls,
        "terminated_reason": terminated_reason,
        "session_data": session_data,
        "usage": usage_objects,
    }
