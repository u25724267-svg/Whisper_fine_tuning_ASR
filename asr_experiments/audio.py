"""Shared audio decoding primitives for data preparation workflows."""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_mono_audio(path: Path, sampling_rate: int) -> np.ndarray:
    """Decode, downmix, and resample finite audio to a float32 mono waveform."""
    waveform, source_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if source_rate != sampling_rate:
        waveform = librosa.resample(
            waveform, orig_sr=source_rate, target_sr=sampling_rate
        )
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"Invalid audio: {path}")
    return waveform