"""STT demo app.

    python demo.py                   live: Enter to start, Enter to stop, q to quit
    python demo.py a.wav b.wav       transcribe audio files
"""
import argparse
import time

from stt.config import load_config
from stt.pipeline import STTPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Local speech-to-text for the Booster K1")
    parser.add_argument("files", nargs="*", help="audio files to transcribe instead of the mic")
    parser.add_argument("--config", help="path to a config.yaml (default: project config)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("Loading model (one time) ...")
    t0 = time.perf_counter()
    with STTPipeline(cfg) as pipe:
        t = pipe.transcriber
        print(f"Ready in {time.perf_counter() - t0:.1f}s on {t.device}/{t.compute_type} "
              f"| model: {cfg.model.size} | outputs: {', '.join(cfg.output.sinks)}")

        if args.files:
            for f in args.files:
                print(f"\n{f}")
                pipe.process_file(f)
            return

        print("\nPress Enter to START listening, Enter again to STOP. Type q then Enter to quit.")
        try:
            while True:
                if input("\n[Enter] Start listening > ").strip().lower() == "q":
                    break
                pipe.start_listening()
                input("  Listening... [Enter] Stop > ")
                pipe.stop_listening()
        except KeyboardInterrupt:
            pass
    print("Bye.")


if __name__ == "__main__":
    main()