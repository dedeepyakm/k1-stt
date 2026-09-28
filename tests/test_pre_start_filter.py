"""Tests for removing speech spoken before Start was pressed.

Word timings in the first test are taken from a real recording
(Sep 23), where pre-roll captured "the window" before the command.
"""
from stt.schema import Word
from stt.transcriber import Transcriber

drop = Transcriber._drop_pre_start_words


def texts(words):
    return " ".join(w.word for w in words)


def test_chatter_before_start_is_removed():
    # Real case: Start at 0.5 s, "the window." ended at 0.48 s, then a 1.2 s pause.
    words = [Word("the", 0.00, 0.14, 0.41), Word("window.", 0.14, 0.48, 0.92),
             Word("Find", 1.70, 1.98, 0.98), Word("the", 1.98, 2.16, 0.98),
             Word("red", 2.16, 2.36, 0.98), Word("chair", 2.36, 2.70, 0.90)]
    kept, dropped = drop(words, start_offset_s=0.5)
    assert texts(kept) == "Find the red chair"
    assert texts(dropped) == "the window."


def test_early_first_word_is_kept():
    # Operator starts "Please" slightly before Start; the word continues past it.
    words = [Word("Please", 0.30, 0.70, 0.79), Word("go", 0.70, 0.90, 0.95)]
    kept, dropped = drop(words, start_offset_s=0.5)
    assert texts(kept) == "Please go"
    assert dropped == []


def test_continuous_speech_before_start_is_kept():
    # Words end before Start but flow straight into the command (no pause): keep.
    words = [Word("Go", 0.20, 0.35, 0.9), Word("to", 0.35, 0.48, 0.9),
             Word("the", 0.52, 0.60, 0.9), Word("kitchen", 0.60, 0.95, 0.9)]
    kept, dropped = drop(words, start_offset_s=0.5)
    assert texts(kept) == "Go to the kitchen"
    assert dropped == []


def test_no_pre_roll_changes_nothing():
    # File input has no pre-roll (offset 0): never drop anything.
    words = [Word("the", 0.00, 0.14, 0.4), Word("window.", 0.14, 0.48, 0.9),
             Word("Find", 1.70, 1.98, 0.98)]
    kept, dropped = drop(words, start_offset_s=0.0)
    assert kept == words
    assert dropped == []