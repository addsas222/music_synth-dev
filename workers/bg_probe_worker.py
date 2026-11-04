"""判断一段「清唱」录音里背景声（伴奏 / 音乐 / 环境噪声）是不是太大。

用法（和其它 worker 一样）：python -I bg_probe_worker.py <job.json> <result.json>

job:
  input         (str)   音频路径（wav/mp3/flac 都行，soundfile 读）
  threshold_db  (float, default 25)  「安静段」和「响段」相差多少 dB 以内算背景声大
  quiet_floor_db(float, default -36) 安静段高于这个值，说明底下一直有东西在响

result:
  ok, noisy(是否建议先做人声分离), gap_db, quiet_db, loud_db, silence_ratio, duration_s
  ok=False 时带 error（调用方按「不分离」继续，不阻断流程）

判定思路：真清唱在停顿处会明显掉下去（gap 很大，安静段很低）；
底下垫着音乐时，停顿处也被音乐托着，gap 变小。所以看 p10 与 p90 的差。
连续不断唱歌、几乎没有停顿的录音判断不了，保守地按「不分离」处理。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _read(path: Path):
    import soundfile as sf
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return data, int(rate)


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out_path = Path(sys.argv[2])
    result: dict = {"ok": False, "step": "bg_probe"}
    try:
        import numpy as np

        src = Path(job["input"]).resolve()
        data, rate = _read(src)
        mono = data.mean(axis=1) if getattr(data, "ndim", 1) > 1 else data
        if rate <= 0 or mono.size == 0:
            raise RuntimeError("空音频")

        hop = max(1, int(rate * 0.05))
        usable = (mono.size // hop) * hop
        frames = mono[:usable].reshape(-1, hop)
        rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1)) if frames.size else np.zeros(1)
        db = 20.0 * np.log10(np.maximum(rms, 1e-8))

        loud = float(np.percentile(db, 90))
        quiet = float(np.percentile(db, 10))
        gap = loud - quiet
        silence_ratio = float((db < (loud - 25.0)).mean())

        threshold = float(job.get("threshold_db") or 25.0)
        quiet_floor = float(job.get("quiet_floor_db") or -36.0)

        # 安静段本身足够低（比响段低 45dB 以上）→ 干净的干声，不分离
        # 安静段被托高了、且和响段接近 → 底下一直有音乐/噪声，先分离人声
        if quiet <= -45.0:
            noisy = False
            reason = "quiet-enough"
        elif quiet >= quiet_floor and gap <= threshold:
            noisy = True
            reason = "background-loud"
        else:
            noisy = False
            reason = "gaps-are-quiet"

        result = {
            "ok": True,
            "noisy": bool(noisy),
            "reason": reason,
            "gap_db": round(gap, 2),
            "quiet_db": round(quiet, 2),
            "loud_db": round(loud, 2),
            "silence_ratio": round(silence_ratio, 3),
            "threshold_db": threshold,
            "quiet_floor_db": quiet_floor,
            "duration_s": round(mono.size / float(rate), 2),
        }
    except Exception as exc:  # noqa: BLE001
        result = {"ok": False, "step": "bg_probe", "error": f"{type(exc).__name__}: {exc}"}

    out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
