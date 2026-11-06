import torchaudio


def _load_with_soundfile(audio_path: str):
    """Fallback when torchaudio's torchcodec backend is unavailable.

    Windows + torch 2.9 often has torchaudio but no loadable FFmpeg DLLs for
    torchcodec.  soundfile handles WAV/FLAC/OGG natively and the corpus clips
    are WAV, so this keeps training working without a system FFmpeg backend.
    """
    import numpy as np
    import soundfile as sf
    import torch

    data, sr = sf.read(audio_path, always_2d=True, dtype="float32")
    return torch.from_numpy(np.ascontiguousarray(data.T)), sr


def load_audio_stereo(audio_path: str, target_sample_rate: int, max_duration: float):
    """Load audio, resample, convert to stereo, and truncate."""
    try:
        audio, sr = torchaudio.load(audio_path)
    except Exception:  # noqa: BLE001 - torchcodec/FFmpeg unavailable
        audio, sr = _load_with_soundfile(audio_path)

    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)

    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2, :]

    max_samples = int(max_duration * target_sample_rate)
    if audio.shape[1] > max_samples:
        audio = audio[:, :max_samples]

    return audio, sr
