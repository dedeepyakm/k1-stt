"""Quick check: record 5 s from the default mic, transcribe on GPU (CPU fallback)."""
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parent.parent       # project folder, works on any OS
MODELS_DIR = ROOT / "models"
RECORDINGS_DIR = ROOT / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 16000   # Whisper expects 16 kHz mono
SECONDS = 5
MODEL_SIZE = "small.en"


def add_nvidia_dll_dirs() -> None:
    if os.name != "nt":
        return
    import site
    for sp in site.getsitepackages():
        for bin_dir in (Path(sp) / "nvidia").glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]


def main() -> None:
    add_nvidia_dll_dirs()
    import ctranslate2
    import sounddevice as sd
    import soundfile as sf
    from faster_whisper import WhisperModel

    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Loading {MODEL_SIZE} on {device}/{compute_type} ...")
    model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type,
                         download_root=str(MODELS_DIR))

    input(f"\nPress Enter, then speak a command for {SECONDS} seconds...")
    print("Recording...")
    audio = sd.rec(int(SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    audio = audio.flatten()
    wav_path = RECORDINGS_DIR / "quick_test.wav"
    sf.write(wav_path, audio, SAMPLE_RATE)
    print(f"Saved: {wav_path}")

    t0 = time.perf_counter()
    segments, info = model.transcribe(audio, language="en", vad_filter=True)
    segments = list(segments)
    elapsed = time.perf_counter() - t0

    text = " ".join(s.text.strip() for s in segments)
    print(f"\nTranscript : {text or '(no speech detected)'}")
    print(f"Time       : {elapsed:.2f} s")
    for s in segments:
        print(f"  [{s.start:.1f}-{s.end:.1f}s] avg_logprob={s.avg_logprob:.2f} "
              f"no_speech_prob={s.no_speech_prob:.2f}")


if __name__ == "__main__":
    main()