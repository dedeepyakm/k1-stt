"""The message sent from STT to VLN: one JSON object per spoken command.

Fields are grouped: identity and time, the command itself, quality signals,
timing, and full detail (segments and words) for anyone who wants it.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

SCHEMA_VERSION = "1.0"


@dataclass
class Word:
    word: str
    start: float          # seconds from start of recording
    end: float
    probability: float    # model's confidence in this word, 0 to 1


@dataclass
class Segment:
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float


@dataclass
class STTResult:
    # Identity and time
    schema_version: str = SCHEMA_VERSION
    command_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = ""           # ISO 8601, UTC
    unix_time: float = 0.0

    # The command
    transcript: str = ""
    accepted: bool = False        # False = do not act on this command
    reject_reason: Optional[str] = None
    discarded_text: Optional[str] = None   # speech before Start, removed from transcript

    # Quality signals
    confidence: float = 0.0       # 0 to 1, from exp(avg_logprob)
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    language: str = "en"
    language_probability: Optional[float] = None

    # Timing
    audio_duration_s: float = 0.0
    speech_duration_s: float = 0.0
    processing_time_s: float = 0.0

    # Provenance and full detail
    source: str = "mic"           # mic | file
    model: dict = field(default_factory=dict)
    segments: list = field(default_factory=list)
    words: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.unix_time:
            self.unix_time = time.time()
        if not self.timestamp:
            self.timestamp = datetime.fromtimestamp(
                self.unix_time, timezone.utc).isoformat(timespec="milliseconds")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


if __name__ == "__main__":
    # Example message, useful to share with the VLN team.
    example = STTResult(
        transcript="Go to the kitchen and grab a coffee for me.",
        accepted=True,
        confidence=0.76,
        avg_logprob=-0.28,
        no_speech_prob=0.01,
        language_probability=1.0,
        audio_duration_s=5.0,
        speech_duration_s=2.8,
        processing_time_s=0.8,
        model={"size": "small.en", "device": "cuda", "compute_type": "float16"},
        segments=[Segment(0.1, 2.9, "Go to the kitchen and grab a coffee for me.", -0.28, 0.01)],
        words=[Word("Go", 0.1, 0.3, 0.98), Word("kitchen", 0.7, 1.1, 0.95)],
    )
    print(example.to_json(indent=2))