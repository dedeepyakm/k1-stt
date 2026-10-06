"""Mock VLN receiver: stands in for the VLN module, so STT's HTTP output can be tested alone.

    Terminal 1:  python scripts/mock_vln.py
    Terminal 2:  python app.py        (with output.sinks: [console, jsonl, http])

Speak some commands, then press Ctrl+C here for a summary of delivery delay and message loss.

It listens on the address set in config.yaml (http.url). It is also a minimal example of a
receiver for the VLN team: read the JSON body and act only when "accepted" is true.
"""
import json
import statistics
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # allow "import stt"
from stt.config import load_config  # noqa: E402

received = {}   # command_id -> delay in ms


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


def make_handler(path: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args) -> None:   # keep the terminal clean
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
            data = json.loads(self.rfile.read(length))        # the STT JSON message
            delay_ms = (time.time() - data["unix_time"]) * 1000
            received[data["command_id"]] = delay_ms

            status = "ACCEPTED" if data["accepted"] else "REJECTED"
            print(f'[{status}] "{data["transcript"]}"  confidence={data["confidence"]:.2f}  '
                  f"delay={delay_ms:.1f} ms")
            # A real VLN would start navigation here when data["accepted"] is true.
            self._reply(HTTPStatus.OK, {"status": "received", "command_id": data["command_id"]})

    return Handler


def main() -> None:
    cfg = load_config()
    url = urlparse(cfg.http.url)
    host, port, path = url.hostname, url.port or 80, url.path or "/"

    server = ThreadingHTTPServer((host, port), make_handler(path))
    started = time.time()
    print(f"Mock VLN listening on {cfg.http.url}   (Ctrl+C for summary)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

    # ---- summary ----
    expected = expected_ids(cfg.output.jsonl_path, started, cfg.http.send_rejected)
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


if __name__ == "__main__":
    main()