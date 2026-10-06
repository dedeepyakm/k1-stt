"""Load config.yaml into typed settings.

- Relative paths resolve from the project root, so the same config works on
  Windows and Ubuntu.
- Unknown keys raise an error, so typos in config.yaml are caught early.
- A few environment variables override the file without editing it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Optional, Union

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class ModelConfig:
    size: str = "small.en"
    device: str = "auto"
    compute_type: str = "auto"
    download_dir: Path = Path("models")
    language: str = "en"
    beam_size: int = 5
    initial_prompt: str = ""
    word_timestamps: bool = True


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    input_device: Optional[Union[int, str]] = None
    max_record_seconds: float = 15.0
    pre_roll_seconds: float = 0.5
    post_roll_seconds: float = 0.4


@dataclass
class VadConfig:
    enabled: bool = True
    min_silence_ms: int = 500
    speech_pad_ms: int = 200


@dataclass
class GateConfig:
    reject_empty: bool = True
    max_no_speech_prob: float = 0.6
    noise_max_logprob: float = -1.0
    min_confidence: float = 0.0


@dataclass
class OutputConfig:
    sinks: list = field(default_factory=lambda: ["console", "jsonl"])
    jsonl_path: Path = Path("outputs/commands.jsonl")
    save_audio: bool = False
    audio_dir: Path = Path("recordings")


@dataclass
class HttpConfig:
    url: str = "http://127.0.0.1:5000/command"
    timeout_s: float = 1.0
    send_rejected: bool = False


@dataclass
class Ros2Config:
    topic: str = "/stt/command"
    node_name: str = "stt_publisher"
    publish_rejected: bool = False


@dataclass
class UiConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    history_size: int = 20


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    http: HttpConfig = field(default_factory=HttpConfig)
    ros2: Ros2Config = field(default_factory=Ros2Config)
    ui: UiConfig = field(default_factory=UiConfig)


# Environment variable -> (section, key). Lets you switch settings per machine
# without editing config.yaml, e.g.  $env:STT_DEVICE="cpu"
ENV_OVERRIDES = {
    "STT_MODEL_SIZE": ("model", "size"),
    "STT_DEVICE": ("model", "device"),
    "STT_INPUT_DEVICE": ("audio", "input_device"),
    "STT_HTTP_URL": ("http", "url"),
}


def _resolve_path(value: Any) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _apply(target: Any, data: dict, where: str) -> None:
    """Copy values from a dict onto a dataclass instance, recursively."""
    for key, value in (data or {}).items():
        if not hasattr(target, key):
            raise ValueError(f"Unknown config key '{where}{key}'")
        current = getattr(target, key)
        if is_dataclass(current):
            _apply(current, value, f"{where}{key}.")
        elif isinstance(current, Path):
            setattr(target, key, _resolve_path(value))
        else:
            setattr(target, key, value)


def load_config(path: Optional[Union[str, Path]] = None) -> Config:
    """Load settings: defaults, then config.yaml, then environment overrides."""
    config = Config()

    config_path = Path(path or os.environ.get("STT_CONFIG", DEFAULT_CONFIG_PATH))
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            _apply(config, yaml.safe_load(f) or {}, "")

    for env_name, (section, key) in ENV_OVERRIDES.items():
        if env_name in os.environ:
            value: Any = os.environ[env_name]
            if key == "input_device" and value.isdigit():
                value = int(value)
            setattr(getattr(config, section), key, value)

    # Make every path absolute, including defaults that were never overridden.
    config.model.download_dir = _resolve_path(config.model.download_dir)
    config.output.jsonl_path = _resolve_path(config.output.jsonl_path)
    config.output.audio_dir = _resolve_path(config.output.audio_dir)
    return config