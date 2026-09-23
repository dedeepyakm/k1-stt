"""Environment check: Python, GPU (CUDA), faster-whisper, microphone.
Runs on Windows and Ubuntu without changes."""
import os
import platform
import sys
import time
from pathlib import Path


def add_nvidia_dll_dirs() -> None:
    """On Windows, make pip-installed CUDA DLLs discoverable. No-op on Linux."""
    if os.name != "nt":
        return
    import site
    for sp in site.getsitepackages():
        for bin_dir in (Path(sp) / "nvidia").glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]


def main() -> None:
    print(f"Python   : {sys.version.split()[0]} ({platform.system()} {platform.release()})")
    add_nvidia_dll_dirs()

    import ctranslate2
    n_gpu = ctranslate2.get_cuda_device_count()
    print(f"CTranslate2: {ctranslate2.__version__}, CUDA devices: {n_gpu}")
    if n_gpu:
        print(f"GPU compute types: {sorted(ctranslate2.get_supported_compute_types('cuda'))}")

    import numpy as np
    from faster_whisper import WhisperModel
    audio = np.zeros(16000 * 2, dtype=np.float32)  # 2 s of silence, just to exercise the model

    for device, compute_type in [("cuda", "float16"), ("cpu", "int8")]:
        try:
            t0 = time.perf_counter()
            model = WhisperModel("tiny", device=device, compute_type=compute_type)
            segments, _ = model.transcribe(audio)
            list(segments)  # transcription is lazy; this forces it to run
            print(f"[OK]   {device}/{compute_type} in {time.perf_counter() - t0:.1f}s")
        except Exception as e:
            print(f"[FAIL] {device}/{compute_type}: {type(e).__name__}: {e}")

    import sounddevice as sd
    print("\nAudio devices:")
    print(sd.query_devices())
    print(f"Default input device index: {sd.default.device[0]}")


if __name__ == "__main__":
    main()