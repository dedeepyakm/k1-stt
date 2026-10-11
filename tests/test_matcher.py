"""Tests for approved-command matching, using the VLN team's real commands."""
from stt.matcher import CommandMatcher

CHAIR = "Move and stand beside the white chair. Ensure to keep safe distance from the chair."
CHAIR_WAVE = CHAIR + " Wave your hand after you reach the final destination"
BAG = "Move and stand beside the blue bag. Ensure to keep safe distance from the blue bag."
BOTTLE = "Move and stand beside the blue bottle. Ensure to keep safe distance from the blue bottle."
LADDER = "Move and stand beside the ladder. Ensure to keep safe distance from the ladder."
LADDER_STRAIGHT = "Move straight and stand beside the ladder. Ensure to keep safe distance from the ladder."
m = CommandMatcher([CHAIR, CHAIR_WAVE, BAG, BOTTLE, LADDER, LADDER_STRAIGHT])


def test_exact_command_is_matched():
    r = m.match(CHAIR)
    assert r.matched and r.command == CHAIR and r.distance == 0


def test_small_transcription_errors_are_corrected():
    # "besides" instead of "beside", and "the" dropped: still clearly the chair command.
    r = m.match("Move and stand besides the white chair, ensure to keep safe distance from chair.")
    assert r.matched and r.command == CHAIR


def test_object_equally_close_to_two_commands_is_blocked():
    # "blue bat" is as close to "blue bag" as to "blue bottle": never guess the object.
    r = m.match("Move and stand beside the blue bat. Ensure to keep safe distance from the blue bat.")
    assert not r.matched and r.reason == "ambiguous"


def test_unrelated_command_is_blocked():
    r = m.match("Go to the kitchen and grab me a coffee.")
    assert not r.matched and r.reason == "not_approved"


def test_wave_ending_selects_the_wave_command():
    r = m.match(CHAIR + " Wave your after you reach the final destination")
    assert r.matched and r.command == CHAIR_WAVE


def test_straight_variant_is_kept_apart():
    r = m.match(LADDER_STRAIGHT)
    assert r.matched and r.command == LADDER_STRAIGHT


def test_duplicates_are_removed():
    assert len(CommandMatcher([CHAIR, CHAIR, BAG]).commands) == 2