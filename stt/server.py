"""Operator console server: serves the web UI and a small JSON API.

Uses only the Python standard library. Binds to 127.0.0.1 by default, so the
console is reachable only from the machine it runs on.

API
    GET  /                          the console page
    GET  /api/status                state, system info, recent commands
    GET  /api/level                 live mic level and elapsed time (polled while listening)
    GET  /api/commands?since=<t>    accepted commands newer than unix time t, oldest first
                                    (add &include_rejected=1 to include rejected ones)
    GET  /api/latest                the most recent accepted command, or {} if none
    POST /api/start                 Start Listening
    POST /api/stop                  Stop Listening; returns the STTResult
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from stt.pipeline import STTPipeline

WEB_DIR = Path(__file__).resolve().parent / "web"


class ConsoleState:
    """Thread-safe wrapper around the pipeline for the web API."""

    def __init__(self, pipeline: STTPipeline, history_size: int = 20) -> None:
        self.pipeline = pipeline
        self.history: deque = deque(maxlen=history_size)   # newest first
        self.state = "idle"               # idle | listening | processing
        self._started_at = 0.0
        self._lock = threading.Lock()

    def _elapsed(self) -> float:
        return round(time.monotonic() - self._started_at, 2) if self.state == "listening" else 0.0

    def info(self) -> dict:
        cfg, t = self.pipeline.cfg, self.pipeline.transcriber
        return {
            "state": self.state,
            "elapsed_s": self._elapsed(),
            "model": cfg.model.size,
            "device": t.device,
            "compute_type": t.compute_type,
            "outputs": list(cfg.output.sinks),
            "ros2_topic": cfg.ros2.topic,
            "max_record_seconds": cfg.audio.max_record_seconds,
            "history": list(self.history),
        }

    def level(self) -> dict:
        return {"state": self.state, "elapsed_s": self._elapsed(),
                "level": round(self.pipeline.recorder.level, 4)}

    def commands_since(self, since: float, include_rejected: bool = False) -> list:
        """Commands newer than `since` (unix time), oldest first. Used by VLN to fetch."""
        items = list(self.history)
        items.reverse()
        return [r for r in items
                if r["unix_time"] > since and (r["accepted"] or include_rejected)]

    def start(self) -> dict:
        with self._lock:
            if self.state != "idle":
                raise RuntimeError(f"Cannot start while {self.state}")
            self.pipeline.start_listening()
            self.state = "listening"
            self._started_at = time.monotonic()
        return self.info()

    def stop(self) -> dict:
        with self._lock:
            if self.state != "listening":
                raise RuntimeError("Not listening")
            self.state = "processing"
        try:
            result = self.pipeline.stop_listening().to_dict()
        finally:
            with self._lock:
                self.state = "idle"
        self.history.appendleft(result)
        return result


def _make_handler(console: ConsoleState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args) -> None:   # keep the terminal clean
            pass

        def _send(self, status, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, status=HTTPStatus.OK) -> None:
            self._send(status, json.dumps(data).encode("utf-8"), "application/json")

        def do_GET(self) -> None:
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if url.path in ("/", "/index.html"):
                page = (WEB_DIR / "index.html").read_bytes()
                self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
            elif url.path == "/api/status":
                self._json(console.info())
            elif url.path == "/api/level":
                self._json(console.level())
            elif url.path == "/api/commands":
                try:
                    since = float(query.get("since", ["0"])[0])
                except ValueError:
                    self._json({"error": "since must be a number"}, HTTPStatus.BAD_REQUEST)
                    return
                include_rejected = query.get("include_rejected", ["0"])[0] == "1"
                self._json({"commands": console.commands_since(since, include_rejected),
                            "server_time": time.time()})
            elif url.path == "/api/latest":
                commands = console.commands_since(0.0)
                self._json(commands[-1] if commands else {})
            else:
                self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            try:
                if self.path == "/api/start":
                    self._json(console.start())
                elif self.path == "/api/stop":
                    self._json(console.stop())
                else:
                    self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            except RuntimeError as e:
                self._json({"error": str(e)}, HTTPStatus.CONFLICT)
            except Exception as e:   # report unexpected errors to the UI instead of crashing
                self._json({"error": f"{type(e).__name__}: {e}"},
                           HTTPStatus.INTERNAL_SERVER_ERROR)

    return Handler


def create_server(pipeline: STTPipeline, host: str = "127.0.0.1", port: int = 8765,
                  history_size: int = 20) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), _make_handler(ConsoleState(pipeline, history_size)))
    server.daemon_threads = True
    return server