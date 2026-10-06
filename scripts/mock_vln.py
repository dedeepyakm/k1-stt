"""Mock VLN: stands in for the VLN module, so STT's output can be tested without it.

Two modes, matching the two ways VLN might connect:

  Push (STT sends to VLN). Needs output.sinks to include "http":
      Terminal 1:  python scripts/mock_vln.py
      Terminal 2:  python app.py

  Fetch (VLN asks STT for new commands). Works with the default sinks:
      Terminal 1:  python app.py
      Terminal 2:  python scripts/mock_vln.py --poll

Speak some commands, then press Ctrl+C here for a summary of delay and message loss.
It is also a minimal example of a receiver for the VLN team: read the JSON and act
only when "accepted" is true.
"""
import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # allow "import stt"
from stt.config import load_config  # noqa: E402

received = {}   # command_id -> delay in ms


def record(data: dict) -> None:
    """Handle one STT message: what a real VLN would do on receiving a command."""
    if data["command_id"] in received:
        return
    delay_ms = (time.time() - data["unix_time"]) * 1000
    received[data["command_id"]] = delay_ms
    status = "ACCEPTED" if data["accepted"] else "REJECTED"
    print(f'[{status}] "{data["transcript"]}"  confidence={data["confidence"]:.2f}  '
          f"delay={delay_ms:.1f} ms")
    # A real VLN would start navigation here when data["accepted"] is true.


def expected_ids(log_path: Path, since: float, include_rejected: bool) -> set:
    """Commands the STT app logged after this receiver started (what should have arrived)."""
    if not log_path.exists():
        return set()
    ids = set()
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("unix_time", 0) >= since and (d.get("accepted") or include_rejected):
            ids.add(d["command_id"])
    return ids


def summary(cfg, started: float, include_rejected: bool) -> None:
    expected = expected_ids(cfg.output.jsonl_path, started, include_rejected)
    missing = expected - set(received)
    delays = list(received.values())
    print("\n--- Summary ---")
    print(f"Received          : {len(received)}")
    print(f"Sent (from log)   : {len(expected)}")
    if expected:
        print(f"Missing           : {len(missing)}  (loss {len(missing) / len(expected):.1%})")
    if delays:
        print(f"Delay mean/median : {statistics.mean(delays):.1f} / {statistics.median(delays):.1f} ms")
        print(f"Delay min/max     : {min(delays):.1f} / {max(delays):.1f} ms")


# ---- push mode: STT sends to us ---------------------------------------------------
def make_handler(path: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args) -> None:
            pass

        def _reply(self, status, data: dict) -> None:
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path != path:
                self._reply(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            record(data)
            self._reply(HTTPStatus.OK, {"status": "received", "command_id": data["command_id"]})

    return Handler


def run_push(cfg) -> float:
    url = urlparse(cfg.http.url)
    server = ThreadingHTTPServer((url.hostname, url.port or 80), make_handler(url.path or "/"))
    started = time.time()
    print(f"Mock VLN (push) listening on {cfg.http.url}   (Ctrl+C for summary)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return started


# ---- fetch mode: we ask STT for new commands ----------------------------------------
def run_poll(cfg, interval: float) -> float:
    host = "127.0.0.1" if cfg.ui.host in ("0.0.0.0", "") else cfg.ui.host
    base = f"http://{host}:{cfg.ui.port}/api/commands"
    started = since = time.time()
    print(f"Mock VLN (fetch) polling {base} every {interval}s   (Ctrl+C for summary)")
    warned = False
    try:
        while True:
            try:
                with urllib.request.urlopen(f"{base}?since={since!r}", timeout=2) as resp:
                    data = json.loads(resp.read())
                for cmd in data["commands"]:
                    record(cmd)
                    since = max(since, cmd["unix_time"])
                warned = False
            except (urllib.error.URLError, OSError):
                if not warned:
                    print("Cannot reach the STT console. Is app.py running? Retrying...")
                    warned = True
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    return started


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock VLN receiver for testing STT output")
    parser.add_argument("--poll", action="store_true", help="fetch mode: poll the STT console")
    parser.add_argument("--interval", type=float, default=0.2, help="poll interval in seconds")
    args = parser.parse_args()

    cfg = load_config()
    if args.poll:
        started = run_poll(cfg, args.interval)
        summary(cfg, started, include_rejected=False)
    else:
        started = run_push(cfg)
        summary(cfg, started, include_rejected=cfg.http.send_rejected)


if __name__ == "__main__":
    main()