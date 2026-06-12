import ast
import json
import re
from typing import Any, Callable, Dict, List, Optional
from interactive.oracles.direct_experiment_worker import DirectExperimentWorker
from interactive.interactive_utils import _normalize_assignment
from prompts.task_descriptions.interactive_direct_experiments_prompt import (
    get_variable_mapping,
)

from prompts.task_descriptions.utils import (
    _get_property_label_names,
)


class TextualDirectExperimentWorker(DirectExperimentWorker):
    """
    Oracle worker that accepts natural language experiment requests.

    The worker first converts the textual request into a concrete assignment
    using its configured LLM, then runs the standard experiment flow.
    """

    def __init__(
        self,
        dataset_instance,
        llm_model,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(dataset_instance, llm_model, verbose=verbose, **kwargs)
        self._textual_assignment_history: List[Dict[str, Any]] = []

    def _translate_results_to_text(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate numeric labels/property labels into textual forms using the dataset's
        semantic/textual mapping when available.
        """

        tm = getattr(self.dataset_instance, "text_mapping", {}) or {}
        tm_labels = tm.get("output_labels", {}) if isinstance(tm, dict) else {}
        tm_properties = tm.get("output_properties", {}) if isinstance(tm, dict) else {}
        property_label_names = _get_property_label_names(self.dataset_instance)

        label_val = result.get("label")
        label_text = None
        if label_val is not None and isinstance(tm_labels, dict):
            label_text = tm_labels.get(str(label_val), tm_labels.get(label_val))
        if label_text is None and label_val is not None:
            label_text = self.label_names.get(label_val)

        property_texts: Dict[str, Optional[str]] = {}
        for prop, raw_val in (result.get("property_labels") or {}).items():
            if raw_val is None:
                property_texts[prop] = None
                continue

            tm_prop = tm_properties.get(prop, {}) if isinstance(tm_properties, dict) else {}
            prop_text = tm_prop.get(str(raw_val), tm_prop.get(raw_val)) if isinstance(tm_prop, dict) else None

            if prop_text is None:
                try:
                    prop_text = property_label_names.get(prop, {}).get(int(raw_val))
                except Exception:
                    prop_text = None

            property_texts[prop] = str(prop_text) if prop_text is not None else None

        return {"label_text": label_text, "property_texts": property_texts}

    def _build_assignment_prompt(self, query: str) -> str:
        variable_mapping = get_variable_mapping(self.dataset_instance.text_mapping,
        self.dataset_instance.udd)
        return (
            "You are a conversion assistant. Convert the user's natural language experiment request "
            "into a JSON object mapping each variable name to an integer category.\n\n"
            f"{variable_mapping}\n"
            "Guidelines:\n"
            "- Always return a JSON object with keys exactly matching the variable names (e.g., V0, V1, ...).\n"
            "- Values must be integers corresponding to valid categories.\n"
            "- Respond with only the JSON object and nothing else.\n\n"
            f"User request:\n{query}\n"
            "Return the JSON assignment:"
        )

    def _parse_assignment_response(self, raw_response: str) -> Dict[str, int]:
        """Parse and normalize the LLM response into an assignment dict."""

        def _try_load(candidate: str) -> Optional[Dict[str, Any]]:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            try:
                parsed = ast.literal_eval(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            return None

        parsed = _try_load(raw_response)
        if parsed is None:
            # Try to extract the first JSON object in the text
            match = re.search(r"{.*}", raw_response, flags=re.DOTALL)
            if match:
                parsed = _try_load(match.group(0))

        if parsed is None:
            raise ValueError("Could not parse assignment JSON from LLM response.")

        return _normalize_assignment(parsed)

    def _interpret_textual_query(self, query: str) -> Dict[str, int]:
        if not isinstance(query, str):
            raise ValueError("query must be a string containing the experiment request.")



        prompt = self._build_assignment_prompt(query)
        if(self.verbose):
            print("[TextualDirectExperimentWorker] Interpreting textual query:")
            print(prompt)

        response = self.llm.make_call(prompt)
        if response is None:
            raise ValueError("LLM returned no content when parsing the query.")

        if self.verbose:
            print("[TextualDirectExperimentWorker] Raw LLM response for assignment:")
            print(response)

        return self._parse_assignment_response(str(response))

    def _record_textual_assignment(
        self,
        *,
        query: str,
        assignment: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        self._textual_assignment_history.append(
            {
                "query": query,
                "translated_assignment": self._json_safe_assignment(assignment),
                "label": result.get("label"),
                "label_name": result.get("label_name"),
                "property_labels": result.get("property_labels", {}),
                "experiment_index": result.get("experiment_index"),
                "experiment_status": result.get("experiment status"),
            }
        )

    def run_textual_experiment(self, query: str) -> str:
        """
        Convert a natural language experiment request into an assignment and run it.

        Args:
            query: Natural language description of the desired experiment.

        Returns:
            The same structure as run_experiment, including labels and rendered sample.
        """
        try:
            assignment = self._interpret_textual_query(query)
        except Exception as e:
            return f"Error interpreting experiment request: {e}"

        self.experiment_counter += 1
        result = self.run_experiment(assignment)
        self._record_textual_assignment(
            query=query,
            assignment=assignment,
            result=result if isinstance(result, dict) else {},
        )
        result["resolved_assignment"] = self._json_safe_assignment(assignment)
        txt_result = self._translate_results_to_text(result)
        return txt_result

    def get_callable_functions(self) -> List[Callable[..., Any]]:
        """Expose only the textual experiment entrypoint to the interactive LLM."""

        return [self.run_textual_experiment]

    def get_session_data(self) -> Dict[str, Any]:
        session_data = super().get_session_data()
        session_data["textual_assignments"] = list(self._textual_assignment_history)
        return session_data
