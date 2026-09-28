# K1 Speech-to-Text

Local, offline speech-to-text for voice-based Vision-Language Navigation (VLN) on the Booster K1 humanoid robot.
Part of ICT Capstone PG-S2-43, AIML Robotics Lab, University of Adelaide.

An operator presses **Start Listening**, speaks a navigation command, and presses **Stop Listening**.
The command is transcribed on the local machine and handed to the VLN module as a structured JSON message.

## Features

- **Fully local and offline.** faster-whisper runs on the machine itself; no audio leaves it and no API keys are needed.
- **Operator console.** A browser page with Start/Stop Listening, live mic level, transcript, confidence, per-word confidence and command history.
- **Push-to-talk with pre-roll and post-roll**, so first and last words are not clipped.
- **Pre-Start filter.** Speech that ended before Start was pressed is removed from the command and logged.
- **Voice activity detection** trims silence around each command.
- **Safety gate.** Empty audio and noise are rejected and never sent to VLN; everything else is forwarded with its confidence score.
- **Structured JSON output** to the console, a JSON log file, and ROS 2.
- **Cross-platform.** The same code runs on Windows and Ubuntu; GPU is used automatically when available, with CPU fallback.

## Evaluation (28 Sep 2026)

20 scripted navigation commands per speaker, 3 speakers (A, B, C), quiet room, laptop microphone.
Word accuracy = 1 - WER, after lowercasing and removing punctuation. Target: at least 80% per speaker.
The evaluation applies the same pre-Start filter as the live console. Participants gave consent;
recordings stay local and are referred to only by letter.

| Speaker | small.en (default) | medium.en |
|---|---|---|
| A | 91.1% | 93.5% |
| B | 95.2% | 96.0% |
| C | 99.2% | 100% |
| **Combined, 60 trials** | **95.2%** | **96.5%** |
| Mean processing time | 0.13 s | 0.27 s |

Findings:
- A 0.4 s post-roll fixed last-word clipping (single-speaker baseline rose from 96.0% to 99.2%).
- Most remaining errors are similar-sounding word swaps (wait/write, desk/disk), which vary by speaker.
- The single-word command "Stop." is the least reliable (lowest confidence, misheard once with both
  models), so a non-voice stop is recommended.
- Numbers are transcribed as digits ("2 meters").

`small.en` is the default: medium.en is only 1.3 points better at twice the processing time and GPU memory,
which the VLN model also needs on the shared lab PC. Lab-condition results will follow.

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
| `python evaluate.py record --speaker a` | Record the 20 evaluation commands for speaker A (resumable) |
| `python evaluate.py record --speaker a --redo 5` | Re-record one trial |
| `python evaluate.py run --all` | Evaluate every speaker; report written to `outputs/eval/` |
| `python evaluate.py run --all --model medium.en` | Compare another model size |
| `python evaluate.py run --all --no-prestart-filter` | Evaluate raw audio without the pre-Start filter |
| `python scripts/check_env.py` | Check Python, GPU and microphone |
| `python scripts/ros2_listener.py` | Receive commands over ROS 2 and measure delay and packet loss |
| `python -m pytest` | Run unit tests |

## Configuration

All settings are in `config.yaml`. Relative paths resolve from the project folder.

| Setting | Purpose |
|---|---|
| `model.size` | `small.en` (default) or `medium.en` for higher accuracy |
| `model.initial_prompt` | Optional vocabulary hint, e.g. object and room names used by VLN |
| `audio.input_device` | `null` for the system default mic, or a device index |
| `audio.pre_roll_seconds` / `post_roll_seconds` | Audio kept before Start and after Stop |
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
| `timestamp`, `unix_time` | When the transcript was produced (UTC) |
| `command_id` | Unique ID for each command |
| `words` | Each word with timing and confidence |
| `discarded_text` | Speech before Start that was removed |

Run `python -m stt.schema` to print a complete example.

### ROS 2

Set `output.sinks: [console, jsonl, ros2]` in `config.yaml`, source ROS 2, then start the app.
Accepted commands are published on `ros2.topic` as `std_msgs/String` containing the JSON.

```bash
source /opt/ros/<distro>/setup.bash
python app.py
```

To check delivery, in a second terminal:

```bash
source /opt/ros/<distro>/setup.bash
ros2 topic echo /stt/command            # raw messages
python scripts/ros2_listener.py         # delay and packet loss; Ctrl+C for the summary
```

`scripts/ros2_listener.py` is also a minimal example of a subscriber: parse `msg.data` as JSON
and act only when `accepted` is true.

To check the network link between the lab PC and the robot: `ping -c 100 <robot-ip>`
(reports packet loss and average delay).

## Project structure

```
app.py                  operator console entry point
demo.py                 terminal demo
evaluate.py             multi-speaker accuracy evaluation
config.yaml             all settings
eval/commands.txt       evaluation commands
stt/
  config.py             loads config.yaml
  runtime.py            GPU/CPU selection, CUDA library setup
  audio.py              push-to-talk recording with pre-roll and post-roll
  transcriber.py        faster-whisper, VAD, pre-Start filter, confidence, safety gate
  schema.py             JSON message sent to VLN
  sinks.py              console, JSON log, ROS 2 outputs
  pipeline.py           connects the modules; loads the model once
  server.py             console web server and API
  web/index.html        console page
tests/                  unit tests
scripts/
  check_env.py          environment and microphone check
  quick_mic_test.py     first feasibility test
  ros2_listener.py      ROS 2 delay and packet-loss test
```

## Known limitations

- Evaluated with three speakers in a quiet room; lab-condition testing pending.
- The ROS 2 output and Ubuntu setup have not yet been tested on the lab PC.
- The console has no login; it is bound to `127.0.0.1` so only the local machine can use it.
- English only, single command per recording (no continuous streaming), per the project scope.
- "Stop" is short and the least reliable command; voice should not be the only way to stop the robot.
- Accuracy varies by speaker and accent (91 to 99% across three speakers).