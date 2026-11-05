import numpy as np
import soundfile as sf
import torch
import torchaudio


def load_audio_stereo(audio_path: str, target_sample_rate: int, max_duration: float):
    """Load audio, resample, convert to stereo, and truncate."""
    # torchaudio 2.9 delegates load() to torchcodec, which is not bundled here.
    # soundfile handles wav/flac natively and avoids that extra dependency.
    try:
        data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        audio = torch.from_numpy(np.ascontiguousarray(data.T))
    except Exception:
        audio, sr = torchaudio.load(audio_path)
        if audio.dtype != torch.float32:
            audio = audio.float()

    if sr != target_sample_rate:
        audio = torchaudio.functional.resample(audio, sr, target_sample_rate)

    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2, :]

    max_samples = int(max_duration * target_sample_rate)
    if audio.shape[1] > max_samples:
        audio = audio[:, :max_samples]

    return audio, sr
