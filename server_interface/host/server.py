import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from configs.interactive_config import InteractiveConfig, InteractiveMode, OracleConfig
from data_management.dataset import RepresentationLevel
from inference import LLMBackend, LLMModel, get_llm_wrapper


REMOTE_HTTP_EVAL_DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "Tests"
    / "Test_Dataset-ns10_nv4_nr5_c3-4_nnv0_ol2_props0"
)


def configure_remote_http_wrapper(payload: dict) -> None:
    """
    Configure the process so normal evaluation code can use CUSTOM_HTTP_BACKEND.

    The evaluation pipeline still calls get_llm_wrapper(...) as usual. When the
    selected model is LLMModel.CUSTOM_HTTP_BACKEND, the factory reads these env
    values to construct the remote wrapper.
    """

    os.environ["REMOTE_HTTP_LLM_ENDPOINT_URL"] = payload["endpoint_url"]
    os.environ["REMOTE_HTTP_LLM_MODEL"] = payload["model"]
    if payload.get("api_key") is not None:
        os.environ["REMOTE_HTTP_LLM_API_KEY"] = payload["api_key"]
    else:
        os.environ.pop("REMOTE_HTTP_LLM_API_KEY", None)


def build_remote_http_evaluation_configs() -> list[dict]:
    """Build the single evaluation config used by the online evaluation path."""

    return [
        {
            "dataset_paths": [str(REMOTE_HTTP_EVAL_DATASET_PATH)],
            "instance_names": ["raw_samples"],
            "model_names": [LLMModel.CUSTOM_HTTP_BACKEND],
            "levels": [RepresentationLevel.RAW],
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


def summarize_evaluation_result(result: dict) -> dict:
    """Return a compact JSON-safe summary for the HTTP response."""

    successes = result.get("successes", [])
    failures = result.get("failures", [])
    return {
        "successes": len(successes),
        "failures": len(failures),
        "first_failure": failures[0].get("error") if failures else None,
    }


class EvalHostHandler(BaseHTTPRequestHandler):
    """Receives LLM endpoint configuration and ends session with eval complete."""

    def _send_json(self, status_code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/start-eval":
            self._send_json(404, {"error": "not found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid json"})
            return

        required_fields = {"endpoint_url", "model", "test_message"}
        missing = required_fields.difference(payload.keys())
        if missing:
            self._send_json(400, {"error": f"missing fields: {sorted(missing)}"})
            return

        configure_remote_http_wrapper(payload)

        wrapper = get_llm_wrapper(
            LLMModel.CUSTOM_HTTP_BACKEND,
            api_key=payload.get("api_key"),
            backend=LLMBackend.NATIVE,
            remote_endpoint_url=payload["endpoint_url"],
            remote_model=payload["model"],
        )
        try:
            reply, _usage = wrapper.make_call_with_info(payload["test_message"])
        except Exception as error:
            self._send_json(400, {"error": f"failed to reach llm endpoint: {error}"})
            return

        response_payload = {
            "status": "eval complete",
            "llm_backend": LLMModel.CUSTOM_HTTP_BACKEND.name,
            "remote_reply": reply,
        }

        if payload.get("run_evaluation"):
            try:
                from evaluation.textual_llm_evaluation import evaluate_parallel

                eval_result = evaluate_parallel(
                    build_remote_http_evaluation_configs(),
                    max_workers=1,
                    verbose=False,
                )
            except Exception as error:
                self._send_json(500, {"error": f"evaluation failed: {error}"})
                return
            response_payload["evaluation"] = summarize_evaluation_result(eval_result)

        self._send_json(200, response_payload)


def run_server(host: str = "127.0.0.1", port: int = 8080):
    server = HTTPServer((host, port), EvalHostHandler)
    print(f"Host server listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
