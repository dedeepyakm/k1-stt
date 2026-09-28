"""20-trial transcription accuracy evaluation (STT success criterion).

    python evaluate.py record              record one WAV per command (resumable)
    python evaluate.py record --redo 5     re-record trial 5 only
    python evaluate.py run                 transcribe all recordings, write report
    python evaluate.py run --model base.en compare another model size

Accuracy definition (also written into the report):
    word accuracy = 1 - WER over all trials, after lowercasing and removing
    punctuation. Target: at least 80% across 20 trials.
"""
import argparse
import csv
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


def trial_path(cfg, i: int) -> Path:
    return cfg.output.audio_dir / "eval" / f"trial_{i:02d}.wav"


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse spaces: so 'Kitchen.' == 'kitchen'."""
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^\w\s']", "", text)
    return " ".join(text.split())


# ---- record ----------------------------------------------------------------------
def cmd_record(args, cfg) -> None:
    from stt.audio import recorder_from_config, save_wav

    commands = load_commands()
    if args.redo and not 1 <= args.redo <= len(commands):
        raise SystemExit(f"--redo must be between 1 and {len(commands)}")
    todo = [args.redo] if args.redo else range(1, len(commands) + 1)

    a = cfg.audio
    print("Read each command naturally, as you would to the robot.")
    with recorder_from_config(a) as rec:
        for i in todo:
            path = trial_path(cfg, i)
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
                    save_wav(audio, path, a.sample_rate)
                    print(f"  Saved {path.name}")
                    break
    print("\nRecording done. Next: python evaluate.py run")


# ---- run -------------------------------------------------------------------------
def cmd_run(args, cfg) -> None:
    from stt.audio import load_audio_file
    from stt.transcriber import Transcriber

    if args.model:
        cfg.model.size = args.model
    commands = load_commands()
    trials = [(i, ref, trial_path(cfg, i)) for i, ref in enumerate(commands, 1)
              if trial_path(cfg, i).exists()]
    if not trials:
        raise SystemExit("No recordings found. Run: python evaluate.py record")

    print(f"Loading {cfg.model.size} ...")
    t = Transcriber(cfg.model, cfg.vad, cfg.gate, cfg.audio.sample_rate)

    rows = []
    for i, ref, path in trials:
        r = t.transcribe(load_audio_file(path, cfg.audio.sample_rate), source="file")
        ref_n, hyp_n = normalize(ref), normalize(r.transcript)
        wer = jiwer.wer(ref_n, hyp_n) if hyp_n else 1.0
        rows.append({
            "trial": i, "reference": ref, "transcript": r.transcript,
            "ref_words": len(ref_n.split()), "wer": round(wer, 4),
            "exact_match": ref_n == hyp_n, "confidence": r.confidence,
            "accepted": r.accepted, "processing_time_s": r.processing_time_s,
        })
        mark = "OK  " if ref_n == hyp_n else "DIFF"
        print(f"  [{mark}] {i:02d}  wer={wer:.2f}  conf={r.confidence:.2f}  \"{r.transcript}\"")

    # Corpus-level WER: total word errors / total reference words.
    total_words = sum(r["ref_words"] for r in rows)
    total_errors = sum(r["wer"] * r["ref_words"] for r in rows)
    accuracy = max(0.0, 1 - total_errors / total_words)
    exact = [r for r in rows if r["exact_match"]]
    wrong = [r for r in rows if not r["exact_match"]]
    times = [r["processing_time_s"] for r in rows]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    passed = accuracy >= TARGET_ACCURACY and len(rows) >= TARGET_TRIALS

    summary = {
        "Model": f"{cfg.model.size} on {t.device}/{t.compute_type}",
        "Trials": f"{len(rows)} of {len(commands)} (target {TARGET_TRIALS})",
        "Word accuracy (1 - WER)": f"{accuracy:.1%} (target {TARGET_ACCURACY:.0%})",
        "Exact sentence match": f"{len(exact)}/{len(rows)} ({len(exact) / len(rows):.0%})",
        "Mean / max processing time": f"{mean(times):.2f} s / {max(times):.2f} s",
        "Mean confidence, exact vs not": f"{mean([r['confidence'] for r in exact]):.2f} vs "
                                         f"{mean([r['confidence'] for r in wrong]):.2f}",
        "Result": "PASS" if passed else "NOT MET",
    }

    # ---- write report ----
    out_dir = cfg.output.jsonl_path.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    base = out_dir / f"{stamp}_{cfg.model.size}"

    with open(f"{base}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md = [f"# STT evaluation report ({datetime.now():%d %b %Y, %H:%M})", "",
          "**Accuracy definition:** word accuracy = 1 - WER over all trials, after "
          "lowercasing and removing punctuation.", "",
          "| Metric | Value |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in summary.items()]
    md += ["", "## Per-trial results", "",
           "| # | Reference | Transcript | WER | Conf. | Time (s) |", "|---|---|---|---|---|---|"]
    md += [f"| {r['trial']} | {r['reference']} | {r['transcript']} | {r['wer']:.2f} | "
           f"{r['confidence']:.2f} | {r['processing_time_s']:.2f} |" for r in rows]
    if wrong:
        md += ["", "## Differences", ""]
        md += [f"- Trial {r['trial']}: expected \"{r['reference']}\", heard \"{r['transcript']}\""
               for r in wrong]
    Path(f"{base}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n" + "\n".join(f"{k:32}: {v}" for k, v in summary.items()))
    print(f"\nReport: {base}.md\nCSV   : {base}.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="STT 20-trial accuracy evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="record one WAV per command")
    rec.add_argument("--redo", type=int, help="re-record a single trial number")
    run = sub.add_parser("run", help="transcribe recordings and write the report")
    run.add_argument("--model", help="override model size, e.g. base.en")
    args = parser.parse_args()

    cfg = load_config()
    {"record": cmd_record, "run": cmd_run}[args.command](args, cfg)


if __name__ == "__main__":
    main()