from __future__ import annotations

import base64
import importlib
import importlib.util
import json
import os
import sys
import tempfile
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
        print("[test_remote_http_file_passthrough] python-dotenv not installed; .env was not loaded.")
        print(
            "[test_remote_http_file_passthrough] OPENAI_API_KEY present: "
            f"{bool(os.getenv('OPENAI_API_KEY'))}"
        )
        return

    load_dotenv = importlib.import_module("dotenv").load_dotenv
    env_path = PROJECT_ROOT / ".env"
    loaded = load_dotenv(env_path)
    print(
        "[test_remote_http_file_passthrough] .env loaded: "
        f"{loaded} from {env_path}"
    )
    print(
        "[test_remote_http_file_passthrough] OPENAI_API_KEY present: "
        f"{bool(os.getenv('OPENAI_API_KEY'))}"
    )


_load_env_if_available()

from configs.interactive_config import InteractiveConfig, InteractiveMode, OracleConfig
from data_management.dataset import AbstractDataset, RepresentationLevel
from inference import CLASSIFIER_SYSTEM_PROMPT, LLMBackend, LLMModel, get_llm_wrapper
from server_interface.host import server as eval_server


CUSTOM_INFERENCE_MODEL_NAME = "Custom_Inference_Model"
OPENAI_BACKING_MODEL = LLMModel.GPT_4O
FILES_EVAL_DATASET_PATH = (
    PROJECT_ROOT / "Tests" / "Test_Dataset-Files-ns10_nv4_nr5_c3-4_nnv0_ol2_props0"
)
FILES_INSTANCE_NAME = "environmental_impact_assessment_summary"
VERBOSE_HTTP_INFERENCE = os.getenv("VERBOSE_HTTP_INFERENCE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}


def build_remote_http_files_evaluation_configs() -> list[dict]:
    return [
        {
            "dataset_paths": [str(FILES_EVAL_DATASET_PATH)],
            "instance_names": [FILES_INSTANCE_NAME],
            "model_names": [LLMModel.CUSTOM_HTTP_BACKEND],
            "levels": [RepresentationLevel.FILES],
            "num_iterations": 1,
            "return_per_sample": False,
            "max_workers": 1,
            "timeout_per_task": None,
            "debug": None,
            "interactive_config": InteractiveConfig(
                mode=InteractiveMode.STATIC,
                verbose=False,
                max_experiments=0,
                number_of_intro_samples=10,
                oracle_config=OracleConfig(
                    llm_model=LLMModel.CUSTOM_HTTP_BACKEND,
                    backend=LLMBackend.NATIVE,
                    verbose=False,
                    kwargs={},
                ),
            ),
        }
    ]


class OpenAIFileProxyLLMHandler(BaseHTTPRequestHandler):
    """Local client endpoint that forwards file requests through OpenAI file upload."""

    _wrapper = None
    inference_checks: list[dict] = []

    @staticmethod
    def _verbose_print(label: str, payload: dict) -> None:
        if not VERBOSE_HTTP_INFERENCE:
            return
        printable = dict(payload)
        if isinstance(printable.get("files"), list):
            printable["files"] = [
                {
                    **{k: v for k, v in file_payload.items() if k != "content_base64"},
                    "content_base64_preview": file_payload.get("content_base64", "")[:64],
                    "content_base64_length": len(file_payload.get("content_base64", "")),
                }
                for file_payload in printable["files"]
                if isinstance(file_payload, dict)
            ]
        print(f"\n[test_remote_http_file_passthrough] {label}")
        print(json.dumps(printable, indent=2, default=str))

    @classmethod
    def reset_checks(cls) -> None:
        cls.inference_checks = []
        cls._wrapper = None

    @classmethod
    def _wrapper_for_openai(cls):
        if cls._wrapper is None:
            cls._wrapper = get_llm_wrapper(
                OPENAI_BACKING_MODEL,
                backend=LLMBackend.NATIVE,
            )
        return cls._wrapper

    @classmethod
    def _record_check(
        cls,
        *,
        request_kind: str,
        request_payload: dict,
        response_payload: dict,
    ) -> None:
        passed, reasons = cls._validate_request_payload(request_payload, request_kind=request_kind)
        response_passed, response_reasons = cls._validate_response_payload(
            response_payload,
            request_kind=request_kind,
        )
        reasons.extend(response_reasons)
        cls.inference_checks.append(
            {
                "request_kind": request_kind,
                "passed": passed and response_passed,
                "reasons": reasons,
                "content_preview": cls._extract_response_content(response_payload)[:500],
            }
        )

    @classmethod
    def print_check_summary(cls) -> None:
        print("\n[test_remote_http_file_passthrough] HTTP inference format checks")
        if not cls.inference_checks:
            print("  FAIL no inference calls were recorded")
            return
        for idx, check in enumerate(cls.inference_checks, start=1):
            status = "PASS" if check["passed"] else "FAIL"
            print(f"  {idx}. {status} {check['request_kind']}")
            for reason in check["reasons"]:
                print(f"     - {reason}")
            if not check["passed"] and check["content_preview"]:
                print(f"     content preview: {check['content_preview']}")

    @classmethod
    def failed_checks(cls) -> list[dict]:
        return [check for check in cls.inference_checks if not check["passed"]]

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

    @staticmethod
    def _extract_response_content(response_payload: dict) -> str:
        try:
            return response_payload["choices"][0]["message"].get("content") or ""
        except Exception:
            return ""

    @classmethod
    def _validate_request_payload(
        cls,
        request_payload: dict,
        *,
        request_kind: str,
    ) -> tuple[bool, list[str]]:
        reasons = []
        if request_payload.get("model") != CUSTOM_INFERENCE_MODEL_NAME:
            reasons.append("request model must use the opaque custom inference model name")

        messages = request_payload.get("messages")
        if not isinstance(messages, list) or not messages:
            reasons.append("request must include non-empty messages")

        files = request_payload.get("files")
        if request_kind == "completion_with_files":
            if not isinstance(files, list) or not files:
                reasons.append("file induction request must include non-empty files")
            else:
                first_file = files[0]
                if first_file.get("mime_type") != "application/pdf":
                    reasons.append("file payload must use application/pdf")
                if not first_file.get("display_name", "").endswith(".pdf"):
                    reasons.append("file payload display_name must identify the PDF")
                try:
                    decoded = base64.b64decode(first_file.get("content_base64", ""), validate=True)
                except Exception as error:
                    reasons.append(f"file payload content_base64 is invalid: {error}")
                else:
                    if not decoded.startswith(b"%PDF"):
                        reasons.append("decoded file payload must be a PDF")
        elif files:
            reasons.append(f"{request_kind} request should not include files")

        return not reasons, reasons

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
        if not isinstance(content, str) or not content.strip():
            reasons.append("assistant content must be a non-empty string")
        elif content.strip().startswith("Error:"):
            reasons.append("assistant content must not be an Error response")

        usage = response_payload.get("usage")
        if isinstance(usage, dict) and "error" in usage:
            reasons.append(f"usage contains error: {usage.get('error')!r}")

        if request_kind == "classifier_codegen" and (
            not isinstance(content, str) or "def classify" not in content
        ):
            reasons.append("classifier codegen response must include def classify")

        return not reasons, reasons

    def _send_json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _upload_incoming_files_to_openai(self, files: list[dict]) -> list:
        wrapper = self._wrapper_for_openai()
        handles = []
        for idx, file_payload in enumerate(files):
            content = base64.b64decode(file_payload["content_base64"], validate=True)
            suffix = Path(file_payload.get("display_name") or f"upload_{idx}.pdf").suffix or ".pdf"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
                tmp.write(content)
                tmp.flush()
                handles.append(
                    wrapper.upload_file(
                        tmp.name,
                        display_name=file_payload.get("display_name") or Path(tmp.name).name,
                        mime_type=file_payload.get("mime_type") or "application/pdf",
                    )
                )
        return handles

    def _send_validated_response(
        self,
        *,
        request_kind: str,
        request_payload: dict,
        response_payload: dict,
    ) -> None:
        self._record_check(
            request_kind=request_kind,
            request_payload=request_payload,
            response_payload=response_payload,
        )
        self._verbose_print("HTTP inference response returned", response_payload)
        self._send_json(200, response_payload)

    def do_POST(self) -> None:
        if self.path != "/openai-file-proxy":
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

        messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
        files = payload.get("files") if isinstance(payload.get("files"), list) else []
        wrapper = self._wrapper_for_openai()
        prompt = self._latest_user_message(messages) or payload.get("message", "")
        is_classifier_codegen = self._has_classifier_system_prompt(messages)

        try:
            if files:
                file_handles = self._upload_incoming_files_to_openai(files)
                content, usage = wrapper.make_call_with_files(prompt, file_handles)
                request_kind = "completion_with_files"
            elif is_classifier_codegen:
                _func, code = wrapper.make_call_to_python_code(
                    prompt,
                    verbose=VERBOSE_HTTP_INFERENCE,
                )
                content = f"```python\n{code}\n```"
                usage = {}
                request_kind = "classifier_codegen"
            else:
                content, usage = wrapper.make_call_with_info(prompt)
                request_kind = "completion"
        except Exception as error:
            content = f"Error: {error}"
            usage = {"error": str(error)}
            request_kind = "completion_with_files" if files else "classifier_codegen" if is_classifier_codegen else "completion"

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
            request_kind=request_kind,
            request_payload=payload,
            response_payload=response_payload,
        )


@unittest.skipUnless(
    os.getenv("RUN_REMOTE_HTTP_OPENAI_FILE_EVAL_TEST") == "1"
    and os.getenv("OPENAI_API_KEY")
    and importlib.util.find_spec("sklearn") is not None
    and importlib.util.find_spec("openai") is not None,
    "Set RUN_REMOTE_HTTP_OPENAI_FILE_EVAL_TEST=1 and OPENAI_API_KEY, with sklearn and openai installed, to run the OpenAI file-backed evaluation integration test.",
)
class TestRemoteHTTPOpenAIFileEvaluationFlow(unittest.TestCase):
    def setUp(self) -> None:
        OpenAIFileProxyLLMHandler.reset_checks()
        self.existing_eval_files = self._snapshot_eval_files()
        self.original_build_configs = eval_server.build_remote_http_evaluation_configs
        eval_server.build_remote_http_evaluation_configs = build_remote_http_files_evaluation_configs

        self.proxy_server = HTTPServer(("127.0.0.1", 0), OpenAIFileProxyLLMHandler)
        self.proxy_port = self.proxy_server.server_address[1]
        self.proxy_thread = threading.Thread(target=self.proxy_server.serve_forever, daemon=True)
        self.proxy_thread.start()

        self.host_server = HTTPServer(("127.0.0.1", 0), eval_server.EvalHostHandler)
        self.host_port = self.host_server.server_address[1]
        self.host_thread = threading.Thread(target=self.host_server.serve_forever, daemon=True)
        self.host_thread.start()

        time.sleep(0.05)

    def tearDown(self) -> None:
        OpenAIFileProxyLLMHandler.print_check_summary()
        eval_server.build_remote_http_evaluation_configs = self.original_build_configs
        self.host_server.shutdown()
        self.host_server.server_close()
        self.proxy_server.shutdown()
        self.proxy_server.server_close()
        self._remove_new_eval_files(self.existing_eval_files)

    def _snapshot_eval_files(self) -> set[Path]:
        eval_root = FILES_EVAL_DATASET_PATH / "evaluation_results"
        if not eval_root.exists():
            return set()
        return {path for path in eval_root.rglob("*.json")}

    def _remove_new_eval_files(self, before: set[Path]) -> None:
        eval_root = FILES_EVAL_DATASET_PATH / "evaluation_results"
        if not eval_root.exists():
            return
        for path in eval_root.rglob("*.json"):
            if path not in before:
                path.unlink()

    def _format_failed_checks(self) -> str:
        lines = []
        for idx, check in enumerate(OpenAIFileProxyLLMHandler.failed_checks(), start=1):
            lines.append(f"{idx}. {check['request_kind']}")
            for reason in check["reasons"]:
                lines.append(f"   - {reason}")
            if check["content_preview"]:
                lines.append(f"   content preview: {check['content_preview']}")
        return "\n".join(lines)

    def test_fixture_points_to_files_dataset_and_representation(self) -> None:
        config = build_remote_http_files_evaluation_configs()[0]
        self.assertEqual(config["dataset_paths"], [str(FILES_EVAL_DATASET_PATH)])
        self.assertEqual(config["instance_names"], [FILES_INSTANCE_NAME])
        self.assertEqual(config["levels"], [RepresentationLevel.FILES])

        dataset = AbstractDataset.load(str(FILES_EVAL_DATASET_PATH))
        self.assertIn(RepresentationLevel.FILES, dataset.dataset_instances)
        self.assertIn(FILES_INSTANCE_NAME, dataset.dataset_instances[RepresentationLevel.FILES])
        instance = dataset.dataset_instances[RepresentationLevel.FILES][FILES_INSTANCE_NAME]
        self.assertTrue(instance.rendered_documents)
        rendered_path = FILES_EVAL_DATASET_PATH / instance.rendered_documents[0]["path"]
        self.assertEqual(rendered_path.suffix, ".pdf")
        self.assertTrue(rendered_path.is_file())

    def test_start_eval_can_run_full_files_evaluation_through_openai_file_proxy(self) -> None:
        payload = {
            "endpoint_url": f"http://127.0.0.1:{self.proxy_port}/openai-file-proxy",
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

        with urlopen(request, timeout=900) as response:
            self.assertEqual(response.status, 200)
            body = json.loads(response.read().decode("utf-8"))

        self.assertEqual(body["status"], "eval complete")
        self.assertEqual(body["llm_backend"], "CUSTOM_HTTP_BACKEND")
        self.assertEqual(body["evaluation"]["successes"], 1)
        self.assertEqual(body["evaluation"]["failures"], 0)

        request_kinds = [check["request_kind"] for check in OpenAIFileProxyLLMHandler.inference_checks]
        self.assertIn("completion", request_kinds)
        self.assertIn("completion_with_files", request_kinds)
        self.assertIn("classifier_codegen", request_kinds)

        failed_checks = OpenAIFileProxyLLMHandler.failed_checks()
        self.assertFalse(
            failed_checks,
            "HTTP inference response format checks failed:\n"
            f"{self._format_failed_checks()}",
        )


if __name__ == "__main__":
    unittest.main()
