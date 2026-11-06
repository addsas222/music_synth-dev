"""量一段人声的基频（用于「自动配调」）。

用法：python -I f0_worker.py <job.json> <result.json>

job:
  input   (str)   音频路径
  fmin    (float, default 65)
  fmax    (float, default 1000)
  max_s   (float, default 45) 只分析前 N 秒，够用且快

result:
  ok, median_hz, p10_hz, p90_hz, voiced_ratio
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out_path = Path(sys.argv[2])
    result: dict = {"ok": False, "step": "f0_probe"}
    try:
        import librosa
        import numpy as np

        src = Path(job["input"]).resolve()
        if not src.is_file():
            raise RuntimeError(f"文件不存在：{src}")
        fmin = float(job.get("fmin") or 65)
        fmax = float(job.get("fmax") or 1000)
        max_s = float(job.get("max_s") or 45)
        y, sr = librosa.load(str(src), sr=22050, mono=True, duration=max_s)
        if y.size == 0:
            raise RuntimeError("空音频")
        f0, voiced_flag, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048)
        f0 = f0[~np.isnan(f0)]
        if f0.size == 0:
            raise RuntimeError("没有检出稳定人声")
        result = {
            "ok": True,
            "median_hz": round(float(np.median(f0)), 1),
            "p10_hz": round(float(np.percentile(f0, 10)), 1),
            "p90_hz": round(float(np.percentile(f0, 90)), 1),
            "voiced_ratio": round(float(np.mean(voiced_flag)), 3),
            "frames": int(f0.size),
        }
    except Exception as exc:  # noqa: BLE001
        result = {"ok": False, "step": "f0_probe", "error": f"{type(exc).__name__}: {exc}"}

    out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
