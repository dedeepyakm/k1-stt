"""The complete STT pipeline: load once, then handle many commands.

Maps directly to the operator UI:
    start_listening()  <- "Start Listening" button
    stop_listening()   <- "Stop Listening" button, returns the result
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np

from stt.audio import MicRecorder, load_audio_file, save_wav
from stt.config import Config, load_config
from stt.schema import STTResult
from stt.sinks import build_sinks
from stt.transcriber import Transcriber


class STTPipeline:
    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or load_config()
        c = self.cfg
        # Slow parts happen here, once.
        self.transcriber = Transcriber(c.model, c.vad, c.gate, c.audio.sample_rate)
        self.recorder = MicRecorder(c.audio.sample_rate, c.audio.input_device,
                                    c.audio.max_record_seconds, c.audio.pre_roll_seconds)
        self.sinks = build_sinks(c)

    # ---- lifecycle -------------------------------------------------------------
    def open(self) -> None:
        self.recorder.open()

    def close(self) -> None:
        self.recorder.close()
        for sink in self.sinks:
            sink.close()

    def __enter__(self) -> "STTPipeline":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- commands ---------------------------------------------------------------
    def start_listening(self) -> None:
        self.recorder.start()

    def stop_listening(self) -> STTResult:
        audio = self.recorder.stop()
        return self.process(audio, source="mic",
                            start_offset_s=self.recorder.last_pre_roll_seconds)

    def process_file(self, path: Union[str, Path]) -> STTResult:
        return self.process(load_audio_file(path, self.cfg.audio.sample_rate), source="file")

    def process(self, audio: np.ndarray, source: str = "mic",
                start_offset_s: float = 0.0) -> STTResult:
        result = self.transcriber.transcribe(audio, source=source,
                                             start_offset_s=start_offset_s)
        if self.cfg.output.save_audio and source == "mic":
            save_wav(audio, self.cfg.output.audio_dir / f"{result.command_id}.wav",
                     self.cfg.audio.sample_rate)
        for sink in self.sinks:
            try:
                sink.send(result)
            except Exception as e:   # one failing output must not stop the others
                warnings.warn(f"{type(sink).__name__} failed: {e}")
        return result