import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class MockLLMHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/mock-llm":
            self._send_json(404, {"error": "not found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid json"})
            return

        message = payload.get("message", "")
        if not message and isinstance(payload.get("messages"), list):
            for entry in reversed(payload["messages"]):
                if entry.get("role") == "user":
                    message = entry.get("content", "")
                    break

        self._send_json(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"Mocked response for model={payload.get('model', 'unknown')}: {message}",
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            },
        )


def run_server(host: str = "127.0.0.1", port: int = 8090):
    server = HTTPServer((host, port), MockLLMHandler)
    print(f"Mock LLM listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
