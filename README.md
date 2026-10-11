# K1 Speech-to-Text

Local, offline speech-to-text for voice-based Vision-Language Navigation (VLN) on the Booster K1 humanoid robot.
Part of ICT Capstone PG-S2-43, AIML Robotics Lab, University of Adelaide.

An operator presses **Start Listening**, speaks a navigation command, and presses **Stop Listening**.
The command is transcribed on the local machine, checked against the approved command list, and sent to the
VLN server. The console then shows the robot executing and the result before allowing the next command.

## Features

- **Fully local and offline.** faster-whisper runs on the machine itself; no audio leaves it and no API keys are needed.
- **Operator console.** Start/Stop Listening, live mic level, transcript, per-word confidence, robot status and command history.
- **Approved commands only.** Each transcript is matched to the VLN team's command list; the approved wording is sent,
  which also corrects small transcription errors. Unclear or unlisted commands are blocked.
- **One command at a time.** After sending, the console locks until VLN reports the task completed or failed.
- **Push-to-talk with pre-roll and post-roll**, so first and last words are not clipped.
- **Pre-Start filter** removes speech said before Start was pressed.
- **Voice activity detection** trims silence around each command.
- **Safety gate.** Empty audio and noise are rejected and never sent.
- **Cross-platform.** The same code runs on Windows and Ubuntu; GPU is used automatically when available, with CPU fallback.

## How it connects to VLN

STT and VLN are two separate programs on the lab PC, talking over HTTP:

1. STT sends `POST /command` with `{"command_id": "...", "text": "<approved command>"}`.
2. The VLN server replies at once with `accepted` (or `busy`) and runs `main(text)` in the background.
3. STT checks `GET /status` every second until it reports `completed` or `failed`.

The VLN side only needs `integration/vln_server.py`, with its placeholder `main()` replaced by the real one.
Full details are in the STT and VLN Integration Guide shared with the team.

### Running the integration

1. In `config.yaml`, set `enabled: true` under `vln:` (it is off by default, so the console also works without VLN).
2. Start the VLN server:
```bash
   uvicorn vln_server:app --host 127.0.0.1 --port 8000          # inside the VLN project
```
   To test without the real VLN, from this project (placeholder `main()` that "walks" for 10 seconds):
```bash
   pip install fastapi uvicorn
   uvicorn vln_server:app --app-dir integration --host 127.0.0.1 --port 8000
```
3. Start the console: `python app.py`.

Tested end to end on 12 Oct 2026 with the placeholder: commands were sent, the console stayed locked while
"executing", and showed "Completed in 0:10"; unapproved commands and an unreachable VLN server were handled.

## Evaluation

### Accuracy (28 Sep 2026)

20 scripted navigation commands per speaker, 3 speakers (A, B, C), quiet room, laptop microphone.
Word accuracy = 1 - WER, after lowercasing and removing punctuation. Target: at least 80% per speaker.
Participants gave consent; recordings stay local and are referred to only by letter.

| Speaker | small.en (default) | medium.en |
|---|---|---|
| A | 91.1% | 93.5% |
| B | 95.2% | 96.0% |
| C | 99.2% | 100% |
| **Combined, 60 trials** | **95.2%** | **96.5%** |
| Mean processing time | 0.13 s | 0.27 s |

`small.en` is the default: medium.en is only 1.3 points better at twice the processing time and GPU memory.
Evaluation on the VLN team's commands in the lab will follow.

### Findings

- A 0.4 s post-roll fixed last-word clipping (single-speaker baseline rose from 96.0% to 99.2%).
- Approved-command matching corrects transcription errors before they reach the robot: "It's sure to keep safe
  distance" was sent as the approved "Ensure to keep safe distance" (12 Oct).
- Commands that differ only by object ("blue bag" vs "blue bottle") are kept apart: a transcript equally close to
  two approved commands is blocked rather than guessed.
- The noise filter originally rejected short, clear commands; it now requires both a high no-speech estimate and
  uncertain words (the same rule Whisper uses).
- A vocabulary hint was tested and rejected (6 Oct): it inserted words the operator never said.
- Short commands can lose their first word; the long VLN commands are less affected.
- Numbers are transcribed as digits ("2 meters").

## Setup

### Windows

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Ubuntu (lab PC)

```bash
sudo apt install python3-venv libportaudio2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On CPU-only machines, the two `nvidia-*` packages in `requirements.txt` can be skipped.
The first run downloads the Whisper model into `models/` (about 480 MB for small.en).

## Usage

| Command | What it does |
|---|---|
| `python app.py` | Operator console at http://127.0.0.1:8765 |
| `python demo.py` | Terminal demo: Enter to start, Enter to stop |
| `python evaluate.py record --speaker a` | Record the evaluation commands for speaker A (resumable) |
| `python evaluate.py run --all` | Evaluate every speaker; report written to `outputs/eval/` |
| `python evaluate.py run --all --model medium.en` | Compare another model size |
| `uvicorn vln_server:app --app-dir integration --port 8000` | Placeholder VLN server for testing |
| `python scripts/check_env.py` | Check Python, GPU and microphone |
| `python -m pytest` | Run unit tests (15) |

## Configuration

All settings are in `config.yaml`. Relative paths resolve from the project folder.

| Setting | Purpose |
|---|---|
| `model.size` | `small.en` (default) or `medium.en` for higher accuracy |
| `audio.max_record_seconds` | Safety cap on one recording (30 s) |
| `audio.pre_roll_seconds` / `post_roll_seconds` | Audio kept before Start and after Stop |
| `matching.enabled` | Only forward commands from the approved list |
| `matching.approved_commands_file` | The approved list (`eval/vln_commands.txt`); one command per line |
| `vln.enabled` | Send approved commands to the VLN server and wait for the result |
| `vln.url` | VLN server address (`http://127.0.0.1:8000`) |
| `vln.max_task_seconds` | Safety limit for one robot task (15 min) |
| `ui.host` | `127.0.0.1` keeps the console on this machine only |

To add an approved command, add a line to `eval/vln_commands.txt` and restart the console.

## Message format

Each command is logged to `outputs/commands.jsonl` as one JSON object. Key fields:

| Field | Meaning |
|---|---|
| `transcript` | What was recognised |
| `matched_command` | The approved wording sent to VLN |
| `accepted` | `false` means it was not sent |
| `reject_reason` | `no_speech`, `likely_noise`, `not_approved`, `ambiguous_command`, ... |
| `confidence` | 0 to 1, from the model's average token log-probability |
| `command_id` | Unique ID, shared with VLN for tracing |
| `words` | Each word with timing and confidence |

Run `python -m stt.schema` to print a complete example.

## Project structure

```
app.py                     operator console entry point
demo.py                    terminal demo
evaluate.py                multi-speaker accuracy evaluation
config.yaml                all settings
eval/commands.txt          evaluation commands
eval/vln_commands.txt      approved commands from the VLN team
integration/vln_server.py  VLN server template (FastAPI) for the VLN team
stt/
  config.py                loads config.yaml
  runtime.py               GPU/CPU selection, CUDA library setup
  audio.py                 push-to-talk recording with pre-roll and post-roll
  transcriber.py           faster-whisper, VAD, pre-Start filter, confidence, safety gate
  matcher.py               approved-command matching
  vln_client.py            sends commands to VLN and reads its status
  schema.py                JSON message format
  sinks.py                 console, JSON log, HTTP, ROS 2 outputs
  pipeline.py              connects the modules; loads the model once
  server.py                console web server, robot status, API
  web/index.html           console page
tests/                     unit tests: pre-Start filter, safety gate, command matching
scripts/                   environment check, mock receivers, ROS 2 listener
```

## Known limitations

- Not yet tested on the lab PC or with the real VLN `main()`; lab testing is planned before demo week.
- Accuracy evaluated with three speakers in a quiet room; lab-condition testing pending.
- A transcription error that turns one approved command into another (e.g. "bag" heard as "bottle") cannot be caught.
- Voice is not used to stop the robot mid-task; the tablet's damp mode is the emergency stop.
- The console has no login; it is bound to `127.0.0.1` so only the local machine can use it.
- English only, one command per recording.
- The ROS 2 output is kept for future use and has not been tested on hardware.