# K1 Speech-to-Text

Local, offline speech-to-text for voice-based Vision-Language Navigation (VLN) on the Booster K1 humanoid robot.
Part of ICT Capstone PG-S2-43, AIML Robotics Lab, University of Adelaide.

An operator presses **Start Listening**, speaks a navigation command, and presses **Stop Listening**.
The command is transcribed on the local machine and handed to the VLN module as a structured JSON message.

## Features

- **Fully local and offline.** faster-whisper runs on the machine itself; no audio leaves it and no API keys are needed.
- **Operator console.** A browser page with Start/Stop Listening, live mic level, transcript, confidence, per-word confidence and command history.
- **Push-to-talk with pre-roll and post-roll**, so first and last words are not clipped.
- **Voice activity detection** trims silence around each command.
- **Safety gate.** Empty audio and noise are rejected and never sent to VLN; everything else is forwarded with its confidence score.
- **Structured JSON output** to the console, a JSON log file, and ROS 2.
- **Cross-platform.** The same code runs on Windows and Ubuntu; GPU is used automatically when available, with CPU fallback.

## Evaluation (baseline, 28 Sep 2026)

20 scripted navigation commands, one speaker, quiet room, laptop microphone.
Word accuracy = 1 - WER over all trials, after lowercasing and removing punctuation. Target: at least 80%.

| Model | Word accuracy | Exact sentences | Mean processing time |
|---|---|---|---|
| small.en (default) | 99.2% | 19/20 | 0.17 s |
| medium.en | 100% | 20/20 | 0.47 s |

The first run scored 96.0%. Three of its four errors were the last word being cut off when Stop was pressed;
adding a 0.4 s post-roll fixed this, and the affected trials were re-recorded. Lab-condition results will follow.

`small.en` is the default because it leaves GPU memory for the VLN model on the shared lab PC and remains usable on CPU.

## Setup

### Windows

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Ubuntu (lab PC, not yet verified)

```bash
sudo apt install python3-venv libportaudio2
python3 -m venv .venv --system-site-packages   # makes ROS 2 (rclpy) visible inside the venv
source .venv/bin/activate
pip install -r requirements.txt
```

The venv must use the same Python version as the installed ROS 2 distro (3.10 for Humble, 3.12 for Jazzy).
On CPU-only machines, the two `nvidia-*` packages in `requirements.txt` can be skipped.

The first run downloads the Whisper model into `models/` (about 480 MB for small.en).

## Usage

| Command | What it does |
|---|---|
| `python app.py` | Operator console at http://127.0.0.1:8765 |
| `python demo.py` | Terminal demo: Enter to start, Enter to stop |
| `python demo.py file.wav` | Transcribe audio files |
| `python evaluate.py record` | Record the 20 evaluation commands |
| `python evaluate.py run` | Transcribe them and write a report to `outputs/eval/` |
| `python scripts/check_env.py` | Check Python, GPU and microphone |
| `python -m pytest` | Run unit tests |

## Configuration

All settings are in `config.yaml`. Relative paths resolve from the project folder.

| Setting | Purpose |
|---|---|
| `model.size` | `small.en` (default) or `medium.en` for higher accuracy |
| `model.initial_prompt` | Optional vocabulary hint, e.g. object and room names used by VLN |
| `audio.input_device` | `null` for the system default mic, or a device index |
| `gate.min_confidence` | 0 forwards everything; raise to reject low-confidence commands |
| `output.sinks` | Any of `console`, `jsonl`, `ros2` |
| `ros2.topic` | Topic VLN subscribes to |
| `ui.host` | `127.0.0.1` keeps the console on this machine only |

Environment overrides: `STT_DEVICE=cpu`, `STT_MODEL_SIZE=medium.en`, `STT_INPUT_DEVICE=3`.

## Output to VLN

Each command produces one JSON object. Key fields:

| Field | Meaning |
|---|---|
| `transcript` | The recognised command |
| `accepted` | `false` means do not act on it |
| `reject_reason` | `no_speech`, `likely_noise`, `low_confidence` or `empty_audio` |
| `confidence` | 0 to 1, from the model's average token log-probability |
| `timestamp` | ISO 8601, UTC |
| `words` | Each word with timing and confidence |
| `discarded_text` | Speech before Start that was removed |

Run `python -m stt.schema` to print a complete example.

### ROS 2

Set `output.sinks: [console, jsonl, ros2]` in `config.yaml`, source ROS 2, then start the app.
Accepted commands are published on `ros2.topic` as `std_msgs/String` containing the JSON.

```bash
source /opt/ros/<distro>/setup.bash
python app.py
ros2 topic echo /stt/command     # in a second terminal, to check
```

## Project structure

```
app.py               operator console entry point
demo.py              terminal demo
evaluate.py          20-trial accuracy evaluation
config.yaml          all settings
eval/commands.txt    evaluation commands
stt/
  config.py          loads config.yaml
  runtime.py         GPU/CPU selection, CUDA library setup
  audio.py           push-to-talk recording with pre-roll and post-roll
  transcriber.py     faster-whisper, VAD, confidence, safety gate
  schema.py          JSON message sent to VLN
  sinks.py           console, JSON log, ROS 2 outputs
  pipeline.py        connects the modules; loads the model once
  server.py          console web server and API
  web/index.html     console page
tests/               unit tests
scripts/             environment and microphone checks
```

## Known limitations

- Evaluated with one speaker in a quiet room; lab and multi-speaker testing pending.
- The ROS 2 output and Ubuntu setup have not yet been tested on the lab PC.
- The console has no login; it is bound to `127.0.0.1` so only the local machine can use it.
- English only, single command per recording (no continuous streaming), per the project scope.