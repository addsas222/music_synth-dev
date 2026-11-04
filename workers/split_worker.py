"""Split a continuous playback recording into individual songs.

Runs in the compute venv (numpy + soundfile are guaranteed there).

Boundary strategy (all evidence is local, nothing is uploaded):
  * energy valleys: RMS envelope, frames quieter than a relative threshold
    for at least ``silence_s`` are candidate cuts;
  * media-title changes: if the caller supplies ``title_marks`` (from the
    system media session / window title timeline) those positions force a cut;
  * minimum song length: cuts that would produce a segment shorter than
    ``min_song_s`` are discarded (the shorter side is merged).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _read_wav(path: str):
    import soundfile as sf
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    return data, int(rate)


def _energy_envelope(mono: np.ndarray, rate: int, hop_s: float = 0.05):
    hop = max(1, int(rate * hop_s))
    frames = len(mono) // hop
    if frames <= 0:
        return np.zeros(1, dtype=np.float32), hop
    trimmed = mono[: frames * hop].reshape(frames, hop)
    rms = np.sqrt((trimmed ** 2).mean(axis=1) + 1e-12)
    return rms, hop


def _silence_cuts(rms: np.ndarray, hop: int, rate: int, silence_s: float,
                  min_gap_s: float) -> list[float]:
    if not len(rms):
        return []
    # Relative floor: a quiet passage is one at least 24 dB below the loud body.
    loud = float(np.percentile(rms, 90))
    floor = max(loud * 0.06, 1e-4)
    quiet = rms < floor
    need = max(1, int(silence_s / (hop / rate)))
    cuts, run_start = [], None
    for idx, is_quiet in enumerate(quiet):
        if is_quiet and run_start is None:
            run_start = idx
        elif not is_quiet and run_start is not None:
            if idx - run_start >= need:
                cuts.append((run_start + idx) / 2.0 * hop / rate)
            run_start = None
    if run_start is not None and len(quiet) - run_start >= need:
        cuts.append((run_start + len(quiet)) / 2.0 * hop / rate)
    # Collapse cuts that are closer than the minimum gap.
    merged: list[float] = []
    for cut in cuts:
        if not merged or cut - merged[-1] >= min_gap_s:
            merged.append(cut)
    return merged


def _apply_title_marks(cuts: list[float], marks: list[dict], min_gap_s: float,
                       duration_s: float) -> list[float]:
    forced = []
    for mark in marks or []:
        try:
            t = float(mark.get("t"))
        except (TypeError, ValueError, AttributeError):
            continue
        if 0.0 < t < duration_s:
            forced.append(t)
    all_cuts = sorted(set(round(c, 2) for c in cuts + forced))
    merged: list[float] = []
    for cut in all_cuts:
        if not merged or cut - merged[-1] >= min_gap_s:
            merged.append(cut)
    return merged


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out_path = Path(sys.argv[2])
    result = {"ok": False, "step": "split"}
    try:
        import soundfile as sf

        src = Path(job["input"]).resolve()
        out_dir = Path(job["out_dir"]).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        min_song_s = float(job.get("min_song_s", 25.0))
        silence_s = float(job.get("silence_s", 2.0))
        min_gap_s = float(job.get("min_gap_s", 20.0))

        data, rate = _read_wav(str(src))
        duration_s = len(data) / float(rate)
        mono = data.mean(axis=1)
        rms, hop = _energy_envelope(mono, rate)
        cuts = _silence_cuts(rms, hop, rate, silence_s, min_gap_s)
        cuts = _apply_title_marks(cuts, job.get("title_marks") or [], min_gap_s, duration_s)

        bounds = [0.0] + cuts + [duration_s]
        # Drop boundaries that would create a too-short song.
        filtered = [bounds[0]]
        for b in bounds[1:]:
            if b - filtered[-1] < min_song_s and b != duration_s:
                continue
            filtered.append(b)
        if len(filtered) >= 2 and duration_s - filtered[-1] < min_song_s and len(filtered) > 2:
            filtered.pop()
            filtered[-1] = duration_s

        tracks = []
        for idx in range(len(filtered) - 1):
            start, end = filtered[idx], filtered[idx + 1]
            if end - start < 3.0:
                continue
            a, b = int(start * rate), int(end * rate)
            name = f"{src.stem}_song{idx + 1:02d}.wav"
            target = out_dir / name
            sf.write(target, data[a:b], rate, subtype="PCM_16")
            tracks.append({
                "index": idx + 1,
                "path": str(target),
                "start_s": round(start, 2),
                "end_s": round(end, 2),
                "duration_s": round(end - start, 2),
            })
        result.update({
            "ok": bool(tracks),
            "source": str(src),
            "source_duration_s": round(duration_s, 2),
            "sample_rate": rate,
            "channels": int(data.shape[1]),
            "silence_cuts_s": [round(c, 2) for c in cuts],
            "tracks": tracks,
            "error": None if tracks else "没有切出任何歌曲（录音过短或全程静音）",
        })
    except Exception as exc:  # noqa: BLE001
        result.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
