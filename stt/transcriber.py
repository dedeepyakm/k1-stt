"""Speech-to-text core: faster-whisper with VAD, confidence, and a safety gate."""
from __future__ import annotations

import math
import time
import warnings
from typing import List, Optional, Tuple

import numpy as np

from stt.config import GateConfig, ModelConfig, VadConfig
from stt.runtime import resolve_device
from stt.schema import Segment, STTResult, Word

PRE_START_GAP_S = 0.3   # pause that separates pre-Start chatter from the real command


class Transcriber:
    """Load the model once, then call transcribe() for each command."""

    def __init__(self, model_cfg: ModelConfig, vad_cfg: VadConfig,
                 gate_cfg: GateConfig, sample_rate: int = 16000,
                 warm_up: bool = True) -> None:
        self.model_cfg = model_cfg
        self.vad_cfg = vad_cfg
        self.gate_cfg = gate_cfg
        self.sample_rate = sample_rate
        self.model, self.device, self.compute_type = self._load_model()
        self.model_info = {"size": model_cfg.size, "device": self.device,
                           "compute_type": self.compute_type}
        if warm_up:
            self._warm_up()

    # ---- setup ---------------------------------------------------------------
    def _load_model(self):
        from faster_whisper import WhisperModel

        device, compute_type = resolve_device(self.model_cfg.device,
                                              self.model_cfg.compute_type)
        kwargs = dict(download_root=str(self.model_cfg.download_dir))
        try:
            model = WhisperModel(self.model_cfg.size, device=device,
                                 compute_type=compute_type, **kwargs)
        except Exception as e:
            if device == "cpu":
                raise
            warnings.warn(f"GPU load failed ({e}); falling back to CPU.")
            device, compute_type = "cpu", "int8"
            model = WhisperModel(self.model_cfg.size, device=device,
                                 compute_type=compute_type, **kwargs)
        return model, device, compute_type

    def _warm_up(self) -> None:
        """Run once on silence so the first real command is fast."""
        silence = np.zeros(self.sample_rate, dtype=np.float32)
        list(self.model.transcribe(silence, language=self.model_cfg.language,
                                   vad_filter=False)[0])

    # ---- main entry point -----------------------------------------------------
    def transcribe(self, audio: np.ndarray, source: str = "mic",
                   start_offset_s: float = 0.0) -> STTResult:
        """Transcribe one command.

        start_offset_s: where Start was pressed within the audio (= pre-roll length).
        Speech that ends before this point and is followed by a pause is removed.
        """
        audio_duration = audio.size / self.sample_rate
        if audio.size == 0:
            return STTResult(accepted=False, reject_reason="empty_audio",
                             source=source, model=self.model_info)

        t0 = time.perf_counter()
        m, v = self.model_cfg, self.vad_cfg
        segments, info = self.model.transcribe(
            audio,
            language=m.language,
            beam_size=m.beam_size,
            initial_prompt=m.initial_prompt or None,
            word_timestamps=m.word_timestamps,
            condition_on_previous_text=False,   # commands are independent
            vad_filter=v.enabled,
            vad_parameters=dict(min_silence_duration_ms=v.min_silence_ms,
                                speech_pad_ms=v.speech_pad_ms),
        )
        segments = list(segments)               # transcription is lazy
        processing_time = time.perf_counter() - t0

        words = [Word(w.word.strip(), round(w.start, 3), round(w.end, 3),
                      round(w.probability, 4))
                 for s in segments for w in (s.words or [])]
        transcript = " ".join(s.text.strip() for s in segments).strip()

        words, dropped = self._drop_pre_start_words(words, start_offset_s)
        if dropped:
            transcript = " ".join(w.word for w in words).strip()
        discarded_text = " ".join(w.word for w in dropped) or None

        avg_logprob, no_speech_prob = self._aggregate(segments)
        confidence = math.exp(avg_logprob) if avg_logprob is not None else 0.0
        accepted, reason = self._gate(transcript, confidence, no_speech_prob)

        return STTResult(
            transcript=transcript,
            accepted=accepted,
            reject_reason=reason,
            discarded_text=discarded_text,
            confidence=round(confidence, 4),
            avg_logprob=None if avg_logprob is None else round(avg_logprob, 4),
            no_speech_prob=None if no_speech_prob is None else round(no_speech_prob, 4),
            language=info.language,
            language_probability=round(info.language_probability, 4),
            audio_duration_s=round(audio_duration, 3),
            speech_duration_s=round(getattr(info, "duration_after_vad", audio_duration), 3),
            processing_time_s=round(processing_time, 3),
            source=source,
            model=self.model_info,
            segments=[Segment(round(s.start, 3), round(s.end, 3), s.text.strip(),
                              round(s.avg_logprob, 4), round(s.no_speech_prob, 4))
                      for s in segments],
            words=words,
        )

    # ---- helpers --------------------------------------------------------------
    @staticmethod
    def _drop_pre_start_words(words: List[Word], start_offset_s: float
                              ) -> Tuple[List[Word], List[Word]]:
        """Split off words spoken entirely before Start and followed by a pause.

        A word that starts early but continues past Start is kept, so a
        genuinely early first word ("Please...") is never lost.
        """
        if start_offset_s <= 0 or not words:
            return words, []
        cut = 0
        for i, w in enumerate(words):
            if w.end > start_offset_s:
                break
            next_start = words[i + 1].start if i + 1 < len(words) else None
            if next_start is None or next_start - w.end >= PRE_START_GAP_S:
                cut = i + 1
        return words[cut:], words[:cut]

    @staticmethod
    def _aggregate(segments) -> Tuple[Optional[float], Optional[float]]:
        """Duration-weighted average of per-segment scores (longer segments count more)."""
        if not segments:
            return None, None
        durations = [max(s.end - s.start, 1e-3) for s in segments]
        total = sum(durations)
        avg_lp = sum(s.avg_logprob * d for s, d in zip(segments, durations)) / total
        nsp = sum(s.no_speech_prob * d for s, d in zip(segments, durations)) / total
        return avg_lp, nsp

    def _gate(self, transcript: str, confidence: float,
              no_speech_prob: Optional[float]) -> Tuple[bool, Optional[str]]:
        """Minimal safety gate: reject empty or noise; forward everything else."""
        g = self.gate_cfg
        if g.reject_empty and not transcript:
            return False, "no_speech"
        if no_speech_prob is not None and no_speech_prob > g.max_no_speech_prob:
            return False, "likely_noise"
        if confidence < g.min_confidence:
            return False, "low_confidence"
        return True, None


if __name__ == "__main__":
    # Manual test:
    #   python -m stt.transcriber                 -> record from mic
    #   python -m stt.transcriber some_file.wav   -> transcribe a file
    import sys

    from stt.audio import MicRecorder, load_audio_file
    from stt.config import load_config

    cfg = load_config()
    print("Loading model and warming up ...")
    t = Transcriber(cfg.model, cfg.vad, cfg.gate, cfg.audio.sample_rate)
    print(f"Ready on {t.device}/{t.compute_type}")

    offset = 0.0
    if len(sys.argv) > 1:
        audio, source = load_audio_file(sys.argv[1], cfg.audio.sample_rate), "file"
    else:
        a = cfg.audio
        with MicRecorder(a.sample_rate, a.input_device, a.max_record_seconds,
                         a.pre_roll_seconds) as rec:
            input("Press Enter to START speaking...")
            rec.start()
            input("Recording... press Enter to STOP.")
            audio, source = rec.stop(), "mic"
            offset = rec.last_pre_roll_seconds

    print(t.transcribe(audio, source=source, start_offset_s=offset).to_json(indent=2))