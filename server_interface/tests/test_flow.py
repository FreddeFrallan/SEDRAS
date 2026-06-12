import json
import threading
import time
import unittest
from http.server import HTTPServer
from urllib.request import Request, urlopen

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server_interface.client.mock_llm_server import MockLLMHandler
from server_interface.host.server import EvalHostHandler
from inference import LLMBackend, LLMModel, get_llm_wrapper


class TestOnlineEvaluationFlow(unittest.TestCase):
    def setUp(self):
        self.mock_llm_server = HTTPServer(("127.0.0.1", 0), MockLLMHandler)
        self.mock_llm_port = self.mock_llm_server.server_address[1]
        self.mock_llm_thread = threading.Thread(target=self.mock_llm_server.serve_forever, daemon=True)
        self.mock_llm_thread.start()

        self.host_server = HTTPServer(("127.0.0.1", 0), EvalHostHandler)
        self.host_port = self.host_server.server_address[1]
        self.host_thread = threading.Thread(target=self.host_server.serve_forever, daemon=True)
        self.host_thread.start()

        time.sleep(0.05)

    def tearDown(self):
        self.host_server.shutdown()
        self.host_server.server_close()
        self.mock_llm_server.shutdown()
        self.mock_llm_server.server_close()

    def test_start_eval_returns_eval_complete(self):
        payload = {
            "endpoint_url": f"http://127.0.0.1:{self.mock_llm_port}/mock-llm",
            "api_key": "local-test-key",
            "model": "mock-model-v1",
            "test_message": "ping",
        }

        request = Request(
            f"http://127.0.0.1:{self.host_port}/start-eval",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 200)
            body = json.loads(response.read().decode("utf-8"))

        self.assertEqual(body["status"], "eval complete")
        self.assertEqual(body["llm_backend"], "CUSTOM_HTTP_BACKEND")
        self.assertIn("Mocked response for model=mock-model-v1", body["remote_reply"])

    def test_remote_http_wrapper_forwards_basic_call(self):
        wrapper = get_llm_wrapper(
            LLMModel.CUSTOM_HTTP_BACKEND,
            api_key="local-test-key",
            backend=LLMBackend.NATIVE,
            remote_endpoint_url=f"http://127.0.0.1:{self.mock_llm_port}/mock-llm",
            remote_model="mock-model-v1",
        )

        content, usage = wrapper.make_call_with_info("hello")

        self.assertIn("Mocked response for model=mock-model-v1: hello", content)
        self.assertEqual(usage["total_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
