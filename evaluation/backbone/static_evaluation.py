from data_management.utils import _schema_from_dataset, _infer_variable_order
from prompts.task_descriptions.static_evaluation_prompt import build_static_prompt
from evaluation.backbone.task_and_theory import InducedTheory
from inference.model_wrappers.llm_wrapper import LLMWrapper, UploadedFileHandle
from data_management.dataset import DatasetInstance
from typing import Dict, List, Optional, Tuple

def induce_textual_theory_from_dataset(
    dataset_instance: DatasetInstance,
    llm: LLMWrapper,
    *,
    verbose: bool = False,
    max_intro_samples: Optional[int] = None,
    file_handles: Optional[List[UploadedFileHandle]] = None,
) -> Tuple[InducedTheory, Dict]:
    if not dataset_instance.samples:
        raise ValueError("Dataset has no samples.")

    variable_order = _infer_variable_order(dataset_instance)
    schema = _schema_from_dataset(dataset_instance, variable_order)

    guidance = build_static_prompt(dataset=dataset_instance, variable_order=variable_order, max_examples=max_intro_samples)

    if verbose:
        print(f"Induction prompt:\n{guidance}\n--- End of prompt ---\n")

    if file_handles:
        theory_text, usage = llm.make_call_with_files(guidance, file_handles=file_handles)
    else:
        theory_text, usage = llm.make_call_with_info(guidance)

    if verbose:
        print(f"Induced theory text:\n{theory_text}\n--- End of theory ---\n")

    static_conversation = {
        "final_message": {"role": "assistant", "content": theory_text},
        "messages": [
            {"role": "user", "content": guidance},
            {"role": "assistant", "content": theory_text},
        ],
        "num_turns": 1,
        "num_tool_calls": 0,
        "terminated_reason": "static_mode",
        "session_data": None,
        "usage": [usage],
    }

    return (
        InducedTheory(
            theory_text=theory_text.strip(),
            variable_order=variable_order,
            schema=schema,
            guidance_prompt=guidance,
        ),
        static_conversation,
    )
