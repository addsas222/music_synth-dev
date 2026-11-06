"""估算人声里的混响含量（0~0.6），供混音自动对齐原曲的混响度。

指标：CPP（倒谱峰突出度，Cepstral Peak Prominence）。
  - 干声的谐波峰在倒谱上非常突出，CPP 高；
  - 混响把能量抹平到时间轴上，CPP 随混响量单调下降。
本机校准值（同一段素材人工加湿，只统计有声帧的 CPP 中位数）：
  干声 ≈ 0.192，0.25 湿 ≈ 0.153，0.5 湿 ≈ 0.138，0.9 湿 ≈ 0.123。
未加能量门限时会被静音/分离底噪拉低，所以必须只取较响的帧。

输出：
  {"ok": true, "wet": 0.32, "cpp": 0.104, "frames": 1234, "note": "..."}
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

# CPP → 混响量的线性映射：0.190 以上视为干声，0.120 以下视为很湿。
_CPP_DRY = 0.190
_CPP_WET = 0.120
_WET_MAX = 0.50


def _cpp(samples, rate):
    import librosa
    import numpy as np

    n_fft = 1024
    hop = 256
    spec = np.abs(librosa.stft(samples, n_fft=n_fft, hop_length=hop))
    if spec.shape[1] < 8:
        return None, 0
    log_spec = np.log(spec + 1e-8)
    ceps = np.fft.irfft(log_spec, axis=0)
    lo = max(2, int(rate / 500))     # 500 Hz 以下的倒谱频率
    hi = min(ceps.shape[0] - 1, int(rate / 60))  # 60 Hz 以上
    if hi <= lo + 4:
        return None, 0
    # 只取能量较高的帧：静音和分离底噪会把 CPP 拉低，误判成"很湿"
    energy = np.sqrt((spec ** 2).sum(axis=0) + 1e-12)
    keep = energy > np.percentile(energy, 50)
    values = []
    for j in np.where(keep)[0]:
        column = ceps[lo:hi, j]
        if column.size < 5:
            continue
        peak = float(np.max(column))
        baseline = float(np.median(column))
        values.append(peak - baseline)
    if not values:
        return None, 0
    return float(np.median(values)), len(values)


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out_path = Path(sys.argv[2])
    result = {"ok": False, "step": "reverb_estimate"}
    try:
        import soundfile as sf
        src = Path(str(job["input"]))
        if not src.is_file():
            raise FileNotFoundError(f"找不到人声：{src}")
        data, rate = sf.read(str(src), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        cpp, frames = _cpp(mono, int(rate))
        if cpp is None:
            result.update({"ok": True, "wet": None, "cpp": None, "frames": frames,
                           "note": "音频太短或没有足够谐波"})
        else:
            ratio = (_CPP_DRY - cpp) / (_CPP_DRY - _CPP_WET)
            wet = round(max(0.0, min(1.0, ratio)) * _WET_MAX, 3)
            result.update({"ok": True, "wet": wet, "cpp": round(cpp, 4),
                           "frames": frames, "secs": round(len(mono) / float(rate), 2),
                           "note": "CPP 估计：值越低说明原曲人声混响越多"})
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["trace"] = traceback.format_exc()[-1500:]
    out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
