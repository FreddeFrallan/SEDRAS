import json
from urllib.request import Request, urlopen


def start_evaluation(host_url: str, endpoint_url: str):
    payload = {
        "endpoint_url": endpoint_url,
        "api_key": "local-test-key",
        "model": "mock-model-v1",
        "test_message": "hello from online evaluation interface",
    }

    request = Request(
        f"{host_url}/start-eval",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=5) as response:
        body = json.loads(response.read().decode("utf-8"))
        print(body)


if __name__ == "__main__":
    start_evaluation("http://127.0.0.1:8080", "http://127.0.0.1:8090/mock-llm")
