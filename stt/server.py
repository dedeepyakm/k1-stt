"""Operator console server: serves the web UI and a small JSON API.

Uses only the Python standard library. Binds to 127.0.0.1 by default, so the
console is reachable only from the machine it runs on.

When vln.enabled is true, each accepted command is sent to the VLN server; the
console then shows "Robot executing" and stays locked until VLN reports the task
completed or failed (or the safety limit passes).

API
    GET  /                          the console page
    GET  /api/status                state, system info, current robot task, recent commands
    GET  /api/level                 live mic level and elapsed time (polled while listening)
    GET  /api/commands?since=<t>    accepted commands newer than unix time t, oldest first
    GET  /api/latest                the most recent accepted command, or {} if none
    POST /api/start                 Start Listening
    POST /api/stop                  Stop Listening; returns the STTResult (+ robot task)
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
from stt.vln_client import VlnClient

WEB_DIR = Path(__file__).resolve().parent / "web"


class ConsoleState:
    """Thread-safe wrapper around the pipeline (and VLN link) for the web API."""

    def __init__(self, pipeline: STTPipeline, history_size: int = 20) -> None:
        self.pipeline = pipeline
        self.history: deque = deque(maxlen=history_size)   # newest first
        self.state = "idle"               # idle | listening | processing | executing
        self.task = None                  # the robot task for the latest sent command
        self._started_at = 0.0
        self._lock = threading.Lock()
        v = pipeline.cfg.vln
        self.vln_cfg = v
        self.vln = VlnClient(v.url, v.request_timeout_s) if v.enabled else None

    def _elapsed(self) -> float:
        return round(time.monotonic() - self._started_at, 2) if self.state == "listening" else 0.0

    def info(self) -> dict:
        cfg, t = self.pipeline.cfg, self.pipeline.transcriber
        return {
            "state": self.state,
            "elapsed_s": self._elapsed(),
            "server_time": time.time(),
            "model": cfg.model.size,
            "device": t.device,
            "compute_type": t.compute_type,
            "outputs": list(cfg.output.sinks),
            "matching": cfg.matching.enabled,
            "vln_enabled": self.vln is not None,
            "vln_url": cfg.vln.url,
            "task": self.task,
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

    # ---- listening -----------------------------------------------------------------
    def start(self) -> dict:
        with self._lock:
            if self.state != "idle":
                raise RuntimeError("Robot is still executing the previous command"
                                   if self.state == "executing" else f"Cannot start while {self.state}")
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
        if self.vln is not None and result["accepted"]:
            self._dispatch(result)
            result["task"] = self.task      # same object: later status updates show in history too
        self.history.appendleft(result)
        return result

    # ---- robot task (VLN) ------------------------------------------------------------
    def _new_task(self, command_id: str, text: str, status: str, message: str = "") -> dict:
        now = time.time()
        return {"command_id": command_id, "text": text, "status": status, "message": message,
                "started": now, "ended": None if status == "running" else now}

    def _dispatch(self, result: dict) -> None:
        cid = result["command_id"]
        text = result.get("matched_command") or result["transcript"]
        try:
            reply = self.vln.send(cid, text)
        except Exception as e:
            self.task = self._new_task(cid, text, "unreachable", f"{type(e).__name__}: {e}")
            print(f"  [VLN] not reachable: {e}")
            return
        status = reply.get("status")
        if status == "accepted":
            with self._lock:
                self.task = self._new_task(cid, text, "running")
                self.state = "executing"
            print(f'  [VLN] accepted, robot executing: "{text}"')
            threading.Thread(target=self._watch, args=(cid,), daemon=True).start()
        elif status == "busy":
            self.task = self._new_task(cid, text, "busy", "Robot is still executing a previous command")
            print("  [VLN] busy: command not started")
        else:
            self.task = self._new_task(cid, text, "error", f"Unexpected reply from VLN: {reply}")
            print(f"  [VLN] unexpected reply: {reply}")

    def _watch(self, command_id: str) -> None:
        cfg = self.vln_cfg
        started = time.monotonic()
        while True:
            time.sleep(cfg.poll_interval_s)
            if time.monotonic() - started > cfg.max_task_seconds:
                self._finish("timeout", f"No completion reported after {cfg.max_task_seconds / 60:.0f} min. "
                                        "Check the robot.")
                return
            try:
                st = self.vln.status()
            except Exception:
                continue                     # VLN briefly unreachable: keep checking
            if st.get("command_id") == command_id and st.get("status") in ("completed", "failed"):
                self._finish(st["status"], st.get("message", ""))
                return

    def _finish(self, status: str, message: str) -> None:
        with self._lock:
            if self.task is not None:
                self.task["status"] = status
                self.task["message"] = message
                self.task["ended"] = time.time()
            self.state = "idle"
        print(f"  [VLN] {status}{': ' + message if message else ''}")


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