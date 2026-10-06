"""Tests for the safety gate (which commands are forwarded to VLN)."""
from stt.config import GateConfig
from stt.transcriber import Transcriber

gate = Transcriber._gate
G = GateConfig()   # default settings


def test_short_clear_command_is_accepted():
    # Real case (6 Oct): "Turn left." was wrongly rejected as noise.
    # High no-speech estimate, but the words were recognised (log-prob about -0.92).
    assert gate(G, "Turn left.", 0.40, -0.92, 0.70) == (True, None)


def test_real_noise_is_rejected():
    # High no-speech estimate AND very uncertain words: treat as noise.
    assert gate(G, "Hmm.", 0.20, -1.60, 0.80) == (False, "likely_noise")


def test_empty_transcript_is_rejected():
    assert gate(G, "", 0.0, None, None) == (False, "no_speech")


def test_normal_command_is_accepted():
    assert gate(G, "Go to the kitchen.", 0.76, -0.27, 0.01) == (True, None)