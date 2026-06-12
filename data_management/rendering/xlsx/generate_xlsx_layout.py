import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from data_management.rendering.errors import InvalidTemplateError
from data_management.rendering.trace_logger import RenderingTraceLogger
from inference.model_wrappers.llm_wrapper import LLMWrapper, get_llm_wrapper, LLMModel


@dataclass
class XlsxLayoutGenerator:
    prompt_md: Optional[str] = None

    def _load_prompt(self, path: Optional[str]) -> str:
        actual_path = path or os.path.join(os.path.dirname(__file__), "prompts", "xlsx_programmatic_prompt.md")
        print(f"Loading XLSX layout prompt from: {actual_path}")
        with open(actual_path, "r", encoding="utf-8") as f:
            return f.read()

    def _generate_samples_overview(self, samples: List[Dict[str, Any]]) -> str:
        label_to_samples = {}
        for idx, sample in enumerate(samples):
            label = sample.get("label", "UNLABELED")
            if label not in label_to_samples:
                label_to_samples[label] = []
            label_to_samples[label].append(idx)

        # Generate a per-label list of the corresponding sample indices
        overview_lines = []
        for label, indices in label_to_samples.items():
            overview_lines.append(f"- Label '{label}':")
            overview_lines.append(", ".join([f"{i}" for i in indices]))  # Show up to first 10 indices
            overview_lines.append(f"\n")

        return "\n".join(overview_lines)

    def generate_layout(
            self,
            *,
            llm: LLMWrapper,
            instance_title: str,
            samples: List[Dict[str, Any]],
            style_hint: Optional[str] = None,
            max_preview_samples: int = 1,
            error_feedback: Optional[str] = None,
            previous_response: Optional[str] = None,  # Capture the actual text returned last time
            trace_logger: Optional[RenderingTraceLogger] = None,
    ) -> Dict[str, Any]:
        base_prompt = self._load_prompt(self.prompt_md)

        # Give the LLM context about indices and distribution
        context = (
            "\n\n---\n"
            f"DATASET_TITLE: {instance_title}\n"
            f"TOTAL_SAMPLES: {len(samples)}\n"
            f"STYLE_HINT: {style_hint or 'None'}\n"
            f"Sample Overview:\n"
            f"{self._generate_samples_overview(samples)}\n"
            "---\n"
            "Return JSON containing 'sheet_name' and a list of 'blocks'."
        )

        if error_feedback and previous_response:
            context += (
                "\n\n### CRITICAL: FIX PREVIOUS ERRORS ###\n"
                "Your previous output was invalid and caused a system crash. "
                "Below is your PREVIOUS RESPONSE and the resulting ERROR LOG.\n\n"
                f"--- YOUR PREVIOUS RESPONSE ---\n{previous_response}\n\n"
                f"--- ERROR LOG ---\n{error_feedback}\n\n"
                "Analyze why the engine failed to parse your response. Remember the following rules:\n"
                "- Do NOT use function calls like 'title_block()'.\n"
                "- Every block MUST look like this: {\"type\": \"table_block\", \"title\": \"...\", \"sample_indices\": [...]}\n"
                "Please correct the structure and return the full JSON."
            )

        prompt = base_prompt + context
        if trace_logger:
            trace_logger.log_llm_interaction(
                stage="xlsx_layout_generation",
                prompt=prompt,
                response=None,
                model=getattr(llm, "model_name", None) if hasattr(llm, "model_name") else None,
                total_samples=len(samples),
            )

        # print(f"Sending prompt to LLM for XLSX layout generation...")
        # print(f"{prompt}")
        raw_text = llm.make_call(prompt)

        if trace_logger:
            trace_logger.log_llm_interaction(
                stage="xlsx_layout_generation",
                prompt=prompt,
                response=raw_text,
                model=getattr(llm, "model_name", None) if hasattr(llm, "model_name") else None,
                total_samples=len(samples),
            )
        # Handle markdown fences
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if json_match:
            clean_json = json_match.group(1).strip()
        else:
            # Fallback to finding the first '{' and last '}'
            start_idx = raw_text.find('{')
            end_idx = raw_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                clean_json = raw_text[start_idx:end_idx + 1]
            else:
                clean_json = raw_text.strip()

        try:
            return json.loads(clean_json)
        except json.JSONDecodeError as exc:
            if trace_logger:
                trace_logger.log_error(
                    "xlsx_layout_generation",
                    exc,
                    response_fragment=raw_text[:5000],
                )
            raise InvalidTemplateError(
                "LLM did not return valid JSON for XLSX layout.", llm_response=raw_text
            ) from exc
