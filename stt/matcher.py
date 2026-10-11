"""Approved-command matching: only commands on the VLN team's list reach the robot.

Transcripts vary slightly ("beside" vs "besides", a dropped "the"), so exact matching
would block correct commands. Instead each transcript is compared word by word with
every approved command, and is accepted only when:

1. the closest approved command is within max_error_ratio of its length, and
2. it is closer than every other approved command by at least min_margin_words.

Rule 2 keeps commands that differ only by object apart ("blue bag" vs "blue bottle"):
a transcript equally close to two of them is blocked, never guessed.
The approved wording is what gets sent on, which also corrects small transcription errors.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union


def normalize(text: str) -> List[str]:
    """Lowercase, drop punctuation, split into words."""
    text = text.lower().replace("-", " ")
    return re.sub(r"[^\w\s']", " ", text).split()


def word_distance(a: List[str], b: List[str]) -> int:
    """Number of word insertions, deletions and substitutions to turn a into b."""
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i]
        for j, wb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (wa != wb)))
        prev = cur
    return prev[-1]


@dataclass
class MatchResult:
    matched: bool
    command: Optional[str]       # approved wording, when matched
    distance: Optional[int]      # word edits to the closest approved command
    margin: Optional[int]        # how many words closer than the runner-up
    reason: Optional[str] = None  # why not matched: empty | not_approved | ambiguous


class CommandMatcher:
    def __init__(self, commands: List[str], max_error_ratio: float = 0.25,
                 min_margin_words: int = 1) -> None:
        seen, unique = set(), []
        for c in commands:
            key = tuple(normalize(c))
            if key and key not in seen:
                seen.add(key)
                unique.append(c.strip())
        if not unique:
            raise ValueError("The approved command list is empty")
        self.commands = unique
        self._words = [normalize(c) for c in unique]
        self.max_error_ratio = max_error_ratio
        self.min_margin_words = min_margin_words

    @classmethod
    def from_file(cls, path: Union[str, Path], **kwargs) -> "CommandMatcher":
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        commands = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
        return cls(commands, **kwargs)

    def match(self, transcript: str) -> MatchResult:
        words = normalize(transcript)
        if not words:
            return MatchResult(False, None, None, None, "empty")
        scored = sorted((word_distance(words, w), i) for i, w in enumerate(self._words))
        best_d, best_i = scored[0]
        margin = scored[1][0] - best_d if len(scored) > 1 else None
        allowed = max(1, round(self.max_error_ratio * len(self._words[best_i])))
        if best_d > allowed:
            return MatchResult(False, None, best_d, margin, "not_approved")
        if margin is not None and margin < self.min_margin_words:
            return MatchResult(False, None, best_d, margin, "ambiguous")
        return MatchResult(True, self.commands[best_i], best_d, margin)