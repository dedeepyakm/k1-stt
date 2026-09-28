"""Multi-speaker 20-trial transcription accuracy evaluation (STT success criterion).

    python evaluate.py record --speaker a              record speaker A (resumable)
    python evaluate.py record --speaker a --redo 5     re-record one trial
    python evaluate.py run --speaker a                 report for one speaker
    python evaluate.py run --all                       every speaker, plus a combined total
    python evaluate.py run --all --model medium.en     compare another model size
    python evaluate.py run --all --no-prestart-filter  raw audio, without the live pre-Start filter

Recordings: recordings/eval/speaker_<name>/trial_XX.wav (kept local, never uploaded).
Each WAV has a small .json noting where Start was pressed, so the evaluation applies
the same pre-Start filter as the live console.

Accuracy definition: word accuracy = 1 - WER over all trials, after lowercasing and
removing punctuation. Target: at least 80% across 20 trials, for every speaker.
"""
import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import jiwer

from stt.config import PROJECT_ROOT, load_config

COMMANDS_FILE = PROJECT_ROOT / "eval" / "commands.txt"
TARGET_ACCURACY = 0.80
TARGET_TRIALS = 20


# ---- helpers -------------------------------------------------------------------
def load_commands() -> list:
    lines = COMMANDS_FILE.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def speaker_dir(cfg, speaker: str) -> Path:
    return cfg.output.audio_dir / "eval" / f"speaker_{speaker.lower()}"


def list_speakers(cfg) -> list:
    base = cfg.output.audio_dir / "eval"
    return sorted(p.name[len("speaker_"):] for p in base.glob("speaker_*") if p.is_dir())


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse spaces: so 'Kitchen.' == 'kitchen'."""
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^\w\s']", "", text)
    return " ".join(text.split())


def start_offset(cfg, wav: Path, use_filter: bool) -> float:
    """Where Start was pressed in the recording, so the pre-Start filter matches the live app."""
    if not use_filter:
        return 0.0
    meta = wav.with_suffix(".json")
    if meta.exists():
        return float(json.loads(meta.read_text(encoding="utf-8")).get("pre_roll_s", 0.0))
    return cfg.audio.pre_roll_seconds   # recordings made before metadata was saved


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _fmt(x, spec: str) -> str:
    return "n/a" if x is None else format(x, spec)


# ---- record ----------------------------------------------------------------------
def cmd_record(args, cfg) -> None:
    from stt.audio import recorder_from_config, save_wav

    commands = load_commands()
    if args.redo and not 1 <= args.redo <= len(commands):
        raise SystemExit(f"--redo must be between 1 and {len(commands)}")
    todo = [args.redo] if args.redo else range(1, len(commands) + 1)
    folder = speaker_dir(cfg, args.speaker)

    print(f"Speaker {args.speaker.upper()}: read each command naturally, as you would to the robot.")
    with recorder_from_config(cfg.audio) as rec:
        for i in todo:
            path = folder / f"trial_{i:02d}.wav"
            if path.exists() and not args.redo:
                print(f"Trial {i:02d} already recorded, skipping.")
                continue
            while True:
                print(f'\nTrial {i:02d}/{len(commands)}:  "{commands[i - 1]}"')
                input("  [Enter] Start > ")
                rec.start()
                input("  Speaking... [Enter] Stop > ")
                audio = rec.stop()
                if input("  [Enter] keep  |  r + Enter to redo > ").strip().lower() != "r":
                    save_wav(audio, path, cfg.audio.sample_rate)
                    path.with_suffix(".json").write_text(
                        json.dumps({"pre_roll_s": round(rec.last_pre_roll_seconds, 3)}),
                        encoding="utf-8")
                    print(f"  Saved {path.name}")
                    break
    print(f"\nDone for speaker {args.speaker.upper()}. Next: python evaluate.py run --all")


# ---- run -------------------------------------------------------------------------
def summarize(rows: list) -> dict:
    words = sum(r["ref_words"] for r in rows)
    errors = sum(r["wer"] * r["ref_words"] for r in rows)
    accuracy = max(0.0, 1 - errors / words) if words else 0.0
    exact = [r for r in rows if r["exact_match"]]
    wrong = [r for r in rows if not r["exact_match"]]
    times = [r["processing_time_s"] for r in rows]
    return {
        "trials": len(rows),
        "accuracy": accuracy,
        "exact": len(exact),
        "mean_time": _mean(times),
        "max_time": max(times) if times else None,
        "conf_exact": _mean([r["confidence"] for r in exact]),
        "conf_wrong": _mean([r["confidence"] for r in wrong]),
        "passed": accuracy >= TARGET_ACCURACY and len(rows) >= TARGET_TRIALS,
    }


def cmd_run(args, cfg) -> None:
    from stt.audio import load_audio_file
    from stt.transcriber import Transcriber

    if args.model:
        cfg.model.size = args.model
    use_filter = not args.no_prestart_filter
    commands = load_commands()
    speakers = list_speakers(cfg) if args.all else [args.speaker.lower()]
    if not speakers:
        raise SystemExit("No recordings found. Run: python evaluate.py record --speaker a")

    print(f"Loading {cfg.model.size} ...  (pre-Start filter: {'on' if use_filter else 'off'})")
    t = Transcriber(cfg.model, cfg.vad, cfg.gate, cfg.audio.sample_rate)

    rows = []
    for spk in speakers:
        folder = speaker_dir(cfg, spk)
        print(f"\nSpeaker {spk.upper()}")
        for i, ref in enumerate(commands, 1):
            path = folder / f"trial_{i:02d}.wav"
            if not path.exists():
                continue
            r = t.transcribe(load_audio_file(path, cfg.audio.sample_rate), source="file",
                             start_offset_s=start_offset(cfg, path, use_filter))
            ref_n, hyp_n = normalize(ref), normalize(r.transcript)
            wer = jiwer.wer(ref_n, hyp_n) if hyp_n else 1.0
            rows.append({
                "speaker": spk.upper(), "trial": i, "reference": ref,
                "transcript": r.transcript, "discarded_text": r.discarded_text or "",
                "ref_words": len(ref_n.split()), "wer": round(wer, 4),
                "exact_match": ref_n == hyp_n, "confidence": r.confidence,
                "accepted": r.accepted, "processing_time_s": r.processing_time_s,
            })
            mark = "OK  " if ref_n == hyp_n else "DIFF"
            note = f'   (ignored before Start: "{r.discarded_text}")' if r.discarded_text else ""
            print(f"  [{mark}] {i:02d}  wer={wer:.2f}  conf={r.confidence:.2f}  \"{r.transcript}\"{note}")
    if not rows:
        raise SystemExit("No recordings found for that speaker.")

    groups = {spk.upper(): [r for r in rows if r["speaker"] == spk.upper()] for spk in speakers}
    summaries = {name: summarize(rs) for name, rs in groups.items() if rs}
    if len(summaries) > 1:
        summaries["Combined"] = summarize(rows)
    overall_pass = all(s["passed"] for name, s in summaries.items() if name != "Combined")

    # ---- console summary ----
    print(f"\nModel: {cfg.model.size} on {t.device}/{t.compute_type}   "
          f"Pre-Start filter: {'on' if use_filter else 'off'}   "
          f"Target: {TARGET_ACCURACY:.0%} word accuracy over {TARGET_TRIALS} trials per speaker\n")
    print(f"{'Speaker':<10}{'Trials':>7}{'Accuracy':>10}{'Exact':>8}{'Mean time':>11}  Result")
    for name, s in summaries.items():
        print(f"{name:<10}{s['trials']:>7}{s['accuracy']:>10.1%}{s['exact']:>5}/{s['trials']:<2}"
              f"{_fmt(s['mean_time'], '.2f'):>9} s  {'PASS' if s['passed'] else 'NOT MET'}")
    print(f"\nOverall: {'PASS' if overall_pass else 'NOT MET'} (every speaker must meet the target)")

    # ---- write report ----
    out_dir = cfg.output.jsonl_path.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    scope = "all" if args.all else f"speaker_{speakers[0]}"
    suffix = "" if use_filter else "_nofilter"
    base = out_dir / f"{datetime.now():%Y%m%d_%H%M}_{cfg.model.size}_{scope}{suffix}"

    with open(f"{base}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md = [f"# STT evaluation report ({datetime.now():%d %b %Y, %H:%M})", "",
          f"**Model:** {cfg.model.size} on {t.device}/{t.compute_type}  ",
          f"**Pre-Start filter:** {'on (same as the live console)' if use_filter else 'off (raw audio)'}  ",
          f"**Target:** at least {TARGET_ACCURACY:.0%} word accuracy over {TARGET_TRIALS} trials, per speaker  ",
          "**Accuracy definition:** word accuracy = 1 - WER over all trials, after lowercasing "
          "and removing punctuation.", "",
          "| Speaker | Trials | Word accuracy | Exact sentences | Mean / max time | "
          "Confidence (exact vs not) | Result |",
          "|---|---|---|---|---|---|---|"]
    for name, s in summaries.items():
        md.append(f"| {name} | {s['trials']} | {s['accuracy']:.1%} | {s['exact']}/{s['trials']} | "
                  f"{_fmt(s['mean_time'], '.2f')} s / {_fmt(s['max_time'], '.2f')} s | "
                  f"{_fmt(s['conf_exact'], '.2f')} vs {_fmt(s['conf_wrong'], '.2f')} | "
                  f"{'PASS' if s['passed'] else 'NOT MET'} |")
    md += ["", f"**Overall: {'PASS' if overall_pass else 'NOT MET'}** (every speaker must meet the target)",
           "", "## Per-trial results", "",
           "| Speaker | # | Reference | Transcript | Ignored before Start | WER | Conf. | Time (s) |",
           "|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['speaker']} | {r['trial']} | {r['reference']} | {r['transcript']} | "
           f"{r['discarded_text']} | {r['wer']:.2f} | {r['confidence']:.2f} | "
           f"{r['processing_time_s']:.2f} |" for r in rows]
    wrong = [r for r in rows if not r["exact_match"]]
    if wrong:
        md += ["", "## Differences", ""]
        md += [f"- Speaker {r['speaker']}, trial {r['trial']}: expected \"{r['reference']}\", "
               f"heard \"{r['transcript']}\"" for r in wrong]
    Path(f"{base}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nReport: {base}.md\nCSV   : {base}.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-speaker STT accuracy evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="record one WAV per command for a speaker")
    rec.add_argument("--speaker", required=True, help="speaker label, e.g. a, b, c")
    rec.add_argument("--redo", type=int, help="re-record a single trial number")

    run = sub.add_parser("run", help="transcribe recordings and write the report")
    who = run.add_mutually_exclusive_group(required=True)
    who.add_argument("--speaker", help="evaluate one speaker")
    who.add_argument("--all", action="store_true", help="evaluate every recorded speaker")
    run.add_argument("--model", help="override model size, e.g. medium.en")
    run.add_argument("--no-prestart-filter", action="store_true",
                     help="evaluate raw audio without removing speech before Start")

    args = parser.parse_args()
    cfg = load_config()
    {"record": cmd_record, "run": cmd_run}[args.command](args, cfg)


if __name__ == "__main__":
    main()