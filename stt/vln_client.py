
"""Client for the VLN server's HTTP API (see integration/vln_server.py).

POST /command  {"command_id", "text"}  ->  {"status": "accepted" | "busy", ...}
GET  /status                           ->  {"status": "idle" | "running" | "completed" | "failed", ...}
Uses only the Python standard library.
"""
from __future__ import annotations

import json
import urllib.request


class VlnClient:
    def __init__(self, base_url: str, request_timeout_s: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = request_timeout_s

    def _call(self, method: str, path: str, body: dict = None) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(self.base_url + path, data=data, method=method,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read() or b"{}")

    def send(self, command_id: str, text: str) -> dict:
        return self._call("POST", "/command", {"command_id": command_id, "text": text})

    def status(self) -> dict:
        return self._call("GET", "/status")