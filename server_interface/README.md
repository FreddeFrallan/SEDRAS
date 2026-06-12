# Online Evaluation Interface (Prototype)

This folder contains a simple host/client split for starting online evaluations over HTTP.

## Layout

- `host/server.py`: Host endpoint (`POST /start-eval`) that accepts an LLM endpoint configuration, configures `LLMModel.CUSTOM_HTTP_BACKEND`, validates it through the normal wrapper factory, and returns `{"status": "eval complete"}`.
- `client/mock_llm_server.py`: Mock LLM endpoint (`POST /mock-llm`) to simulate a user-provided LLM server.
- `client/start_eval.py`: Simple client launcher that sends evaluation-start payload to the host.
- `tests/test_flow.py`: End-to-end local test that spins up both servers and validates `eval complete` response.

## Remote HTTP wrapper contract

The evaluation host forwards normal LLM calls to the client-provided endpoint
through `inference.model_wrappers.remote_http_wrapper.RemoteHTTPWrapper`.

Requests are JSON `POST`s shaped like:

```json
{
  "model": "mock-model-v1",
  "messages": [{"role": "user", "content": "prompt text"}],
  "message": "prompt text",
  "tools": [],
  "tool_choice": "auto"
}
```

`tools` and `tool_choice` are only present for tool-calling evaluation modes.
If an `api_key` is provided to `/start-eval`, the wrapper sends it as
`Authorization: Bearer <api_key>`.

File-based evaluation calls use the same JSON endpoint and add a `files` array:

```json
{
  "model": "mock-model-v1",
  "messages": [{"role": "user", "content": "prompt text"}],
  "message": "prompt text",
  "files": [
    {
      "file_id": "/absolute/or/client-local/path.pdf",
      "display_name": "environmental_impact_assessment_summary.pdf",
      "mime_type": "application/pdf",
      "content_base64": "JVBERi0x..."
    }
  ]
}
```

Client LLM servers should parse `files` as optional. For each entry, decode
`content_base64` using standard base64 decoding, interpret the bytes according
to `mime_type`, and use `display_name` only as a human-readable name. `file_id`
is an opaque identifier from the caller; do not assume it is a usable path on
the receiving server. Tool-calling file requests include both `tools` and
`files` in the same payload.

Responses may be OpenAI-compatible:

```json
{
  "choices": [
    {"message": {"role": "assistant", "content": "answer text"}}
  ],
  "usage": {"total_tokens": 123}
}
```

For prototype endpoints, a simpler `{"reply": "answer text"}` response is also
accepted.

To use the remote endpoint in the normal evaluation pipeline, select:

```python
"target_llm_model": LLMModel.CUSTOM_HTTP_BACKEND
```

`/start-eval` stores the provided endpoint settings in `REMOTE_HTTP_LLM_*`
environment variables so later `get_llm_wrapper(LLMModel.CUSTOM_HTTP_BACKEND)`
calls can construct the same remote wrapper.

For integration testing, `/start-eval` also accepts:

```json
{
  "run_evaluation": true
}
```

When set, the host builds one static evaluation config for:

```text
Tests/Test_Dataset-ns10_nv4_nr5_c3-4_nnv0_ol2_props0
```

and runs it through `evaluate_parallel` with `LLMModel.CUSTOM_HTTP_BACKEND`.
The focused test `test_custom_llm_remote_evaluation.py` starts a local HTTP
endpoint identified to the host as `Custom_Inference_Model`. That endpoint
forwards the remote-wrapper requests to the backend model selected by
`REMOTE_EVAL_BACKEND_MODEL`, defaulting to `GEMINI_3_FLASH_PREVIEW`; the host
only sees the opaque custom model name. Because this spends real API calls, it
is opt-in and loads API keys from the repo `.env` file when `python-dotenv` is
installed:

```bash
RUN_REMOTE_HTTP_BACKEND_EVAL_TEST=1 python -m unittest server_interface.tests.test_custom_llm_remote_evaluation
```

## Manual local flow

Terminal A:

```bash
python -m server_interface.host.server
```

Terminal B:

```bash
python -m server_interface.client.mock_llm_server
```

Terminal C:

```bash
python -m server_interface.client.start_eval
```

Expected output:

```text
{'status': 'eval complete'}
```
