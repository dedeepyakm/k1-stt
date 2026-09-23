"""Runtime setup that differs between machines: CUDA libraries and device choice.

Everything here is automatic. The same code runs on the Windows laptop (GPU)
and the Ubuntu lab PC (GPU or CPU).
"""
from __future__ import annotations

import os
import site
from pathlib import Path
from typing import Tuple

_prepared = False


def _nvidia_dirs() -> list:
    """Folders of the pip-installed NVIDIA packages (nvidia-cublas, nvidia-cudnn)."""
    dirs = []
    for sp in site.getsitepackages():
        base = Path(sp) / "nvidia"
        if base.is_dir():
            dirs.append(base)
    return dirs


def prepare_runtime() -> None:
    """Make CUDA libraries discoverable. Safe to call many times."""
    global _prepared
    if _prepared:
        return
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    if os.name == "nt":
        # Windows: add the DLL folders to the search path.
        for base in _nvidia_dirs():
            for bin_dir in base.glob("*/bin"):
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]
    else:
        # Linux: preload the shared libraries so CTranslate2 can find them.
        # Best effort: if the lab PC has a system-wide CUDA install, this is not needed.
        import ctypes
        patterns = ("cublas/lib/libcublasLt.so*", "cublas/lib/libcublas.so*",
                    "cudnn/lib/libcudnn*.so*")
        for base in _nvidia_dirs():
            for pattern in patterns:
                for lib in sorted(base.glob(pattern)):
                    try:
                        ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass
    _prepared = True


def resolve_device(device: str = "auto", compute_type: str = "auto") -> Tuple[str, str]:
    """Turn 'auto' settings into a concrete (device, compute_type)."""
    prepare_runtime()
    import ctranslate2

    if device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type