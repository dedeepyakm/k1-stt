"""VLN server: connects the VLN module to the STT operator console.

Copy this file into the VLN project, replace the placeholder main() below with the
real one, then run:

    pip install fastapi uvicorn
    uvicorn vln_server:app --host 127.0.0.1 --port 8000

STT sends   POST /command  {"command_id": "...", "text": "<approved instruction>"}
and checks  GET  /status   every second until the task is completed or failed.

Requirement on main(text): return when the task finishes, raise an error if it fails.
"""
import threading
import time
import traceback
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel


# ---- REPLACE THIS PLACEHOLDER with the real VLN entry point, e.g.:  from vln.run import main
def main(text: str) -> None:
    """Placeholder: pretends the robot walks for 10 seconds."""
    time.sleep(10)
# ------------------------------------------------------------------------------------------


app = FastAPI(title="VLN server for STT")
_lock = threading.Lock()
_state = {"status": "idle", "command_id": None, "text": None,
          "started": None, "finished": None, "message": ""}


class Command(BaseModel):
    text: str
    command_id: Optional[str] = None


def _run(cmd: Command) -> None:
    try:
        main(cmd.text)
        status, message = "completed", ""
    except Exception as e:                      # main() raised: the task failed
        traceback.print_exc()
        status, message = "failed", f"{type(e).__name__}: {e}"
    with _lock:
        _state.update(status=status, message=message, finished=time.time())
    print(f"[VLN] {status}: {cmd.command_id} {message}")


@app.post("/command")
def command(cmd: Command):
    with _lock:
        if _state["status"] == "running":       # one task at a time, no queuing
            return {"status": "busy", "command_id": _state["command_id"]}
        _state.update(status="running", command_id=cmd.command_id, text=cmd.text,
                      started=time.time(), finished=None, message="")
    print(f"[VLN] running {cmd.command_id}: {cmd.text}")
    threading.Thread(target=_run, args=(cmd,), daemon=True).start()
    return {"status": "accepted", "command_id": cmd.command_id}


@app.get("/status")
def status():
    with _lock:
        s = dict(_state)
    end = s["finished"] or time.time()
    elapsed = round(end - s["started"], 1) if s["started"] else 0.0
    return {"status": s["status"], "command_id": s["command_id"], "text": s["text"],
            "elapsed_s": elapsed, "message": s["message"]}