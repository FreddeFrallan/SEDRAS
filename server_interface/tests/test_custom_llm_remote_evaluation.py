import importlib
import importlib.util
import json
import os
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _load_env_if_available() -> None:
    spec = importlib.util.find_spec("dotenv")
    if spec is None:
        print("[test_custom_llm_remote_evaluation] python-dotenv not installed; .env was not loaded.")
        print(
            "[test_custom_llm_remote_evaluation] GOOGLE_API_KEY present: "
            f"{bool(os.getenv('GOOGLE_API_KEY'))}"
        )
        return
    load_dotenv = importlib.import_module("dotenv").load_dotenv
    env_path = PROJECT_ROOT / ".env"
    loaded = load_dotenv(env_path)
    print(
        "[test_custom_llm_remote_evaluation] .env loaded: "
        f"{loaded} from {env_path}"
    )
    print(
        "[test_custom_llm_remote_evaluation] GOOGLE_API_KEY present: "
        f"{bool(os.getenv('GOOGLE_API_KEY'))}"
    )


_load_env_if_available()

from inference import CLASSIFIER_SYSTEM_PROMPT, LLMBackend, LLMModel, get_llm_wrapper
from server_interface.host.server import EvalHostHandler, REMOTE_HTTP_EVAL_DATASET_PATH

CUSTOM_INFERENCE_MODEL_NAME = "Custom_Inference_Model"
BACKEND_MODEL_ENV_VAR = "REMOTE_EVAL_BACKEND_MODEL"
DEFAULT_BACKEND_MODEL = LLMModel.GEMINI_3_FLASH_PREVIEW
VERBOSE_HTTP_INFERENCE = os.getenv("VERBOSE_HTTP_INFERENCE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}


def _resolve_backend_model() -> LLMModel:
    configured = os.getenv(BACKEND_MODEL_ENV_VAR)
    if not configured:
        return DEFAULT_BACKEND_MODEL

    normalized = configured.strip()
    for model in LLMModel:
        if normalized in {
            model.name,
            model.native_id,
            model.litellm_id,
            str(model),
        }:
            return model

    valid_names = ", ".join(model.name for model in LLMModel)
    raise ValueError(
        f"Unknown {BACKEND_MODEL_ENV_VAR}={configured!r}. Use one of: {valid_names}"
    )


def _resolve_model_name(model_name: str) -> LLMModel:
    for model in LLMModel:
        if model_name in {model.name, model.native_id, model.litellm_id, str(model)}:
            return model
    raise ValueError(f"Unknown model requested by remote wrapper: {model_name!r}")


def _backend_api_key_is_available(model: LLMModel) -> bool:
    if model.name.startswith("GEMINI"):
        return bool(os.getenv("GOOGLE_API_KEY"))
    if model.name.startswith("GPT"):
        return bool(os.getenv("OPENAI_API_KEY"))
    if model.name.startswith("CLAUDE"):
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    if model.name.startswith("GROK"):
        return bool(os.getenv("XAI_API_KEY"))
    if model.name.startswith("DEEPSEEK"):
        return bool(os.getenv("DEEPSEEK_API_KEY"))
    return True


BACKEND_MODEL = _resolve_backend_model()


class BackendProxyLLMHandler(BaseHTTPRequestHandler):
    """Local client endpoint that forwards remote-wrapper requests to a real LLM backend."""

    _wrappers = {}
    inference_checks = []

    @staticmethod
    def _verbose_print(label: str, payload: dict) -> None:
        if not VERBOSE_HTTP_INFERENCE:
            return
        print(f"\n[test_custom_llm_remote_evaluation] {label}")
        print(json.dumps(payload, indent=2, default=str))

    @classmethod
    def reset_checks(cls) -> None:
        cls.inference_checks = []

    @classmethod
    def _record_check(
        cls,
        *,
        request_kind: str,
        passed: bool,
        reasons: list[str],
        response_payload: dict,
    ) -> None:
        content = cls._extract_response_content(response_payload)
        cls.inference_checks.append(
            {
                "request_kind": request_kind,
                "passed": passed,
                "reasons": reasons,
                "content_preview": (content or "")[:500],
            }
        )

    @classmethod
    def print_check_summary(cls) -> None:
        print("\n[test_custom_llm_remote_evaluation] HTTP inference format checks")
        if not cls.inference_checks:
            print("  FAIL no inference calls were recorded")
            return
        for idx, check in enumerate(cls.inference_checks, start=1):
            status = "PASS" if check["passed"] else "FAIL"
            print(f"  {idx}. {status} {check['request_kind']}")
            if check["reasons"]:
                for reason in check["reasons"]:
                    print(f"     - {reason}")
            if not check["passed"] and check["content_preview"]:
                print(f"     content preview: {check['content_preview']}")

    @classmethod
    def failed_checks(cls) -> list[dict]:
        return [check for check in cls.inference_checks if not check["passed"]]

    @staticmethod
    def _extract_response_content(response_payload: dict) -> str:
        try:
            return response_payload["choices"][0]["message"].get("content") or ""
        except Exception:
            return ""

    @classmethod
    def _validate_response_payload(
        cls,
        response_payload: dict,
        *,
        request_kind: str,
    ) -> tuple[bool, list[str]]:
        reasons = []
        choices = response_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            reasons.append("response must include a non-empty choices list")
            return False, reasons

        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            reasons.append("choices[0].message must be an object")
            return False, reasons

        content = message.get("content")
        tool_calls = message.get("tool_calls")
        if request_kind == "tool_call":
            if content is None and not tool_calls:
                reasons.append("tool-call response must include content or tool_calls")
        else:
            if not isinstance(content, str) or not content.strip():
                reasons.append("assistant content must be a non-empty string")
            elif content.strip().startswith("Error:"):
                reasons.append("assistant content must not be an Error response")

        usage = response_payload.get("usage")
        if isinstance(usage, dict) and "error" in usage:
            reasons.append(f"usage contains error: {usage.get('error')!r}")

        if request_kind == "classifier_codegen":
            if not isinstance(content, str) or "def classify" not in content:
                reasons.append("classifier codegen response must include def classify")

        return not reasons, reasons

    def _send_json(self, status_code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _wrapper_for_model(self, model: str):
        backing_model = BACKEND_MODEL if model == CUSTOM_INFERENCE_MODEL_NAME else _resolve_model_name(model)
        if backing_model not in self._wrappers:
            self._wrappers[backing_model] = get_llm_wrapper(
                backing_model,
                backend=LLMBackend.NATIVE,
            )
        return self._wrappers[backing_model]

    @staticmethod
    def _latest_user_message(messages: list[dict]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return message.get("content", "")
        return ""

    @staticmethod
    def _has_classifier_system_prompt(messages: list[dict]) -> bool:
        return any(
            message.get("role") == "system"
            and message.get("content") == CLASSIFIER_SYSTEM_PROMPT
            for message in messages
        )

    @classmethod
    def _send_validated_response(
        cls,
        handler: "BackendProxyLLMHandler",
        *,
        request_kind: str,
        response_payload: dict,
    ) -> None:
        passed, reasons = cls._validate_response_payload(
            response_payload,
            request_kind=request_kind,
        )
        cls._record_check(
            request_kind=request_kind,
            passed=passed,
            reasons=reasons,
            response_payload=response_payload,
        )
        cls._verbose_print("HTTP inference response returned", response_payload)
        handler._send_json(200, response_payload)

    def do_POST(self):
        if self.path != "/llm-proxy":
            self._send_json(404, {"error": "not found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid json"})
            return

        self._verbose_print("HTTP inference request received", payload)

        model = payload.get("model") or CUSTOM_INFERENCE_MODEL_NAME
        messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
        tools = payload.get("tools") if isinstance(payload.get("tools"), list) else []
        wrapper = self._wrapper_for_model(model)

        if tools:
            message, usage = wrapper.chat_with_tools_with_info_and_files(messages, tools)
            response_payload = {"choices": [{"message": message}], "usage": usage}
            self._send_validated_response(
                self,
                request_kind="tool_call",
                response_payload=response_payload,
            )
            return

        prompt = self._latest_user_message(messages) or payload.get("message", "")
        is_classifier_codegen = self._has_classifier_system_prompt(messages)
        if is_classifier_codegen:
            try:
                _func, code = wrapper.make_call_to_python_code(prompt, verbose=VERBOSE_HTTP_INFERENCE)
                content = f"```python\n{code}\n```"
                usage = {}
            except Exception as error:
                content = f"Error: {error}"
                usage = {"error": str(error)}
        else:
            content, usage = wrapper.make_call_with_info(prompt)

        response_payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": content,
                    }
                }
            ],
            "usage": usage,
        }
        self._send_validated_response(
            self,
            request_kind="classifier_codegen" if is_classifier_codegen else "completion",
            response_payload=response_payload,
        )


@unittest.skipUnless(
    os.getenv("RUN_REMOTE_HTTP_BACKEND_EVAL_TEST") == "1"
    and _backend_api_key_is_available(BACKEND_MODEL)
    and importlib.util.find_spec("sklearn") is not None,
    f"Set RUN_REMOTE_HTTP_BACKEND_EVAL_TEST=1, {BACKEND_MODEL_ENV_VAR} if needed, and the backend model API key, with sklearn installed, to run this integration test.",
)
class TestRemoteHTTPBackendEvaluationFlow(unittest.TestCase):
    def setUp(self):
        BackendProxyLLMHandler.reset_checks()
        self.existing_eval_files = self._snapshot_eval_files()

        self.proxy_server = HTTPServer(("127.0.0.1", 0), BackendProxyLLMHandler)
        self.proxy_port = self.proxy_server.server_address[1]
        self.proxy_thread = threading.Thread(target=self.proxy_server.serve_forever, daemon=True)
        self.proxy_thread.start()

        self.host_server = HTTPServer(("127.0.0.1", 0), EvalHostHandler)
        self.host_port = self.host_server.server_address[1]
        self.host_thread = threading.Thread(target=self.host_server.serve_forever, daemon=True)
        self.host_thread.start()

        time.sleep(0.05)

    def tearDown(self):
        BackendProxyLLMHandler.print_check_summary()
        self.host_server.shutdown()
        self.host_server.server_close()
        self.proxy_server.shutdown()
        self.proxy_server.server_close()
        self._remove_new_eval_files(self.existing_eval_files)

    def _snapshot_eval_files(self) -> set[Path]:
        eval_root = REMOTE_HTTP_EVAL_DATASET_PATH / "evaluation_results"
        if not eval_root.exists():
            return set()
        return {path for path in eval_root.rglob("*.json")}

    def _remove_new_eval_files(self, before: set[Path]) -> None:
        eval_root = REMOTE_HTTP_EVAL_DATASET_PATH / "evaluation_results"
        if not eval_root.exists():
            return
        for path in eval_root.rglob("*.json"):
            if path not in before:
                path.unlink()

    def _format_failed_checks(self) -> str:
        lines = []
        for idx, check in enumerate(BackendProxyLLMHandler.failed_checks(), start=1):
            lines.append(f"{idx}. {check['request_kind']}")
            for reason in check["reasons"]:
                lines.append(f"   - {reason}")
            if check["content_preview"]:
                lines.append(f"   content preview: {check['content_preview']}")
        return "\n".join(lines)

    def test_start_eval_can_run_full_evaluation_through_backend_proxy(self):
        payload = {
            "endpoint_url": f"http://127.0.0.1:{self.proxy_port}/llm-proxy",
            "api_key": "local-test-key",
            "model": CUSTOM_INFERENCE_MODEL_NAME,
            "test_message": "ping",
            "run_evaluation": True,
        }

        request = Request(
            f"http://127.0.0.1:{self.host_port}/start-eval",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urlopen(request, timeout=600) as response:
            self.assertEqual(response.status, 200)
            body = json.loads(response.read().decode("utf-8"))

        self.assertEqual(body["status"], "eval complete")
        self.assertEqual(body["llm_backend"], "CUSTOM_HTTP_BACKEND")
        self.assertEqual(body["evaluation"]["successes"], 1)
        self.assertEqual(body["evaluation"]["failures"], 0)
        failed_checks = BackendProxyLLMHandler.failed_checks()
        self.assertFalse(
            failed_checks,
            "HTTP inference response format checks failed:\n"
            f"{self._format_failed_checks()}",
        )


if __name__ == "__main__":
    unittest.main()
