from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

from server_interface.website.database.seed import get_connection, initialize_database


FRONTEND_ROOT = Path(__file__).resolve().parents[1] / "frontend"


def fetch_leaderboard() -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                rank,
                model_name,
                organization,
                representation,
                accuracy,
                full_accuracy,
                latency_seconds,
                status
            FROM leaderboard_entries
            ORDER BY rank ASC
            """
        ).fetchall()
    return [dict(row) for row in rows]


class WebsiteHandler(BaseHTTPRequestHandler):
    def _send_bytes(self, status_code: int, body: bytes, content_type: str) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status_code: int, payload: dict | list) -> None:
        self._send_bytes(
            status_code,
            json.dumps(payload).encode("utf-8"),
            "application/json",
        )

    def _serve_static(self, request_path: str) -> None:
        relative = request_path.lstrip("/") or "index.html"
        if relative == "leaderboard":
            relative = "index.html"

        file_path = (FRONTEND_ROOT / relative).resolve()
        if not str(file_path).startswith(str(FRONTEND_ROOT.resolve())):
            self._send_json(403, {"error": "forbidden"})
            return
        if not file_path.is_file():
            self._send_json(404, {"error": "not found"})
            return

        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        self._send_bytes(200, file_path.read_bytes(), content_type)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/leaderboard":
            self._send_json(200, {"entries": fetch_leaderboard()})
            return
        self._serve_static(parsed.path)


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    initialize_database()
    server = HTTPServer((host, port), WebsiteHandler)
    print(f"SEDRAS benchmark website listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()

