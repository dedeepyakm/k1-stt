"""Audio input: push-to-talk microphone recording and audio files.

MicRecorder matches the operator UI: start() on "Start Listening",
stop() on "Stop Listening". The mic stream stays open while the app runs and
keeps a short rolling pre-roll buffer, so the first word is never clipped.
Output is always 16 kHz mono float32, which is what Whisper expects.
"""
from __future__ import annotations

import threading
from collections import deque
from pathlib import Path
from typing import Optional, Union

import numpy as np


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Simple linear resampling. Only used if the mic cannot record at 16 kHz."""
    if src_rate == dst_rate or audio.size == 0:
        return audio
    n_out = int(round(audio.size * dst_rate / src_rate))
    x_old = np.linspace(0.0, 1.0, audio.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


class MicRecorder:
    """Push-to-talk recorder with pre-roll.

    Usage:
        with MicRecorder() as rec:   # opens the mic once
            rec.start()              # "Start Listening"
            ...
            audio = rec.stop()       # "Stop Listening"
    """

    def __init__(self, sample_rate: int = 16000,
                 device: Optional[Union[int, str]] = None,
                 max_seconds: float = 15.0,
                 pre_roll_seconds: float = 0.5) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self.max_seconds = max_seconds
        self.pre_roll_seconds = pre_roll_seconds
        self.limit_reached = threading.Event()   # set if max_seconds is hit
        self._stream = None
        self._capture_rate = sample_rate
        self._recording = False
        self._chunks: list = []
        self._n_samples = 0
        self._pre_roll: deque = deque()
        self._pre_roll_samples = 0
        self._lock = threading.Lock()

    # ---- state -------------------------------------------------------------
    @property
    def is_open(self) -> bool:
        return self._stream is not None

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def capture_rate(self) -> int:
        return self._capture_rate

    # ---- audio callback (runs on the audio thread) ---------------------------
    def _callback(self, indata, frames, time_info, status) -> None:
        chunk = indata[:, 0].copy()
        with self._lock:
            if self._recording:
                if self._n_samples >= int(self.max_seconds * self._capture_rate):
                    self.limit_reached.set()
                    return
                self._chunks.append(chunk)
                self._n_samples += chunk.size
            else:
                # Idle: keep only the most recent pre_roll_seconds of audio.
                self._pre_roll.append(chunk)
                self._pre_roll_samples += chunk.size
                max_pre = int(self.pre_roll_seconds * self._capture_rate)
                while self._pre_roll and self._pre_roll_samples - self._pre_roll[0].size >= max_pre:
                    self._pre_roll_samples -= self._pre_roll.popleft().size

    # ---- lifecycle ------------------------------------------------------------
    def _make_stream(self, rate: int):
        import sounddevice as sd
        return sd.InputStream(samplerate=rate, channels=1, dtype="float32",
                              device=self.device, callback=self._callback)

    def open(self) -> None:
        """Open the mic stream once. Recording only happens between start() and stop()."""
        if self.is_open:
            return
        import sounddevice as sd
        try:
            self._capture_rate = self.sample_rate
            self._stream = self._make_stream(self.sample_rate)
        except sd.PortAudioError:
            # Mic does not support 16 kHz: record at its native rate, convert on stop().
            info = sd.query_devices(self.device, "input")
            self._capture_rate = int(info["default_samplerate"])
            self._stream = self._make_stream(self._capture_rate)
        self._stream.start()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            self._recording = False
            self._pre_roll.clear()
            self._pre_roll_samples = 0

    def __enter__(self) -> "MicRecorder":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- push-to-talk ---------------------------------------------------------
    def start(self) -> None:
        if self._recording:
            raise RuntimeError("Already recording")
        self.open()   # no-op if already open
        with self._lock:
            self._chunks = list(self._pre_roll)        # seed with pre-roll
            self._n_samples = self._pre_roll_samples
            self._pre_roll.clear()
            self._pre_roll_samples = 0
            self.limit_reached.clear()
            self._recording = True

    def stop(self) -> np.ndarray:
        if not self._recording:
            raise RuntimeError("Not recording")
        with self._lock:
            self._recording = False
            audio = (np.concatenate(self._chunks) if self._chunks
                     else np.zeros(0, dtype=np.float32))
            self._chunks = []
        return _resample(audio, self._capture_rate, self.sample_rate)


def load_audio_file(path: Union[str, Path], sample_rate: int = 16000) -> np.ndarray:
    """Load any common audio format (wav, mp3, m4a, ...) as 16 kHz mono float32."""
    from faster_whisper import decode_audio
    return decode_audio(str(path), sampling_rate=sample_rate)


def save_wav(audio: np.ndarray, path: Union[str, Path], sample_rate: int = 16000) -> Path:
    import soundfile as sf
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate)
    return path


if __name__ == "__main__":
    # Manual test: push-to-talk recording from the configured mic.
    from stt.config import load_config

    cfg = load_config()
    a = cfg.audio
    with MicRecorder(a.sample_rate, a.input_device, a.max_record_seconds,
                     a.pre_roll_seconds) as rec:
        input("Mic ready. Press Enter to START, then speak immediately...")
        rec.start()
        input("Recording... press Enter to STOP.")
        audio = rec.stop()

    path = save_wav(audio, cfg.output.audio_dir / "audio_test.wav", a.sample_rate)
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    print(f"Duration   : {audio.size / a.sample_rate:.2f} s (includes {a.pre_roll_seconds} s pre-roll)")
    print(f"Peak level : {peak:.3f}  (0 = silent, 1 = maximum)")
    print(f"Captured at: {rec.capture_rate} Hz")
    print(f"Saved      : {path}")
    if peak < 0.01:
        print("Warning: almost silent. Check the mic is not muted or the wrong device.")