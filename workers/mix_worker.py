"""Mix the synthesized vocal with the original accompaniment."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def _limit(x, sr, ceiling=0.95, passes=2, window_ms=10.0):
    """Peak limiter: no sample may reach ``ceiling``.

    The old code hard-clipped the vocal (audible as crackle) and then scaled the
    whole mix by 0.99/peak whenever one sample went over, which cost the track a
    few dB of level.  A limiter shapes the gain around the peaks instead, so the
    vocal keeps its level and nothing clips.
    """
    import numpy as np

    y = x.astype(np.float32)
    hop = max(1, int(sr * 0.001))
    win = max(1, int(window_ms))
    kernel = np.hanning(max(3, 2 * win + 1)).astype(np.float32)
    kernel /= kernel.sum()
    for _ in range(max(1, passes)):
        mono = np.abs(y) if y.ndim == 1 else np.max(np.abs(y), axis=1)
        count = len(mono) // hop
        blocks = mono[:count * hop].reshape(count, hop).max(axis=1)
        padded = np.pad(blocks, (win, win), mode="edge")
        env = np.lib.stride_tricks.sliding_window_view(padded, 2 * win + 1).max(axis=1)
        gain = np.minimum(1.0, ceiling / np.maximum(env, 1e-9))
        gain = np.convolve(gain, kernel, mode="same")
        curve = np.interp(np.arange(len(y)) / hop, np.arange(len(gain)), gain).astype(np.float32)
        y = y * (curve if y.ndim == 1 else curve[:, None])
    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak > ceiling:
        y *= ceiling / peak
    return y


def _fftconv(x, h):
    import numpy as np
    n = len(x) + len(h) - 1
    nfft = 1 << max(1, (n - 1).bit_length())
    y = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(h, nfft), nfft)
    return y[: len(x)]


def _reverb(x, sr, seed=1, rt60=1.5, predelay_ms=12.0, damp=0.35):
    """轻量混响：衰减噪声当脉冲响应，左右用不同随机种子做出立体声宽度。"""
    import numpy as np

    rng = np.random.RandomState(seed)
    n = max(1, int(sr * rt60))
    tline = np.arange(n) / float(sr)
    ir = rng.standard_normal(n).astype(np.float32) * np.exp(-6.9 * tline / rt60)
    # 高频衰减（一阶低通），让尾巴更自然
    for i in range(1, n):
        ir[i] = damp * ir[i] + (1.0 - damp) * ir[i - 1]
    pre = int(sr * predelay_ms / 1000.0)
    if pre > 0:
        ir = np.concatenate([np.zeros(pre, dtype=np.float32), ir])
    ir /= float(np.abs(ir).sum() + 1e-9)
    wet = _fftconv(x, ir).astype(np.float32)
    # 按 RMS 归一到和干声同级，否则混响几乎听不见、左右也就没有宽度
    rms_x = float(np.sqrt((x.astype(np.float64) ** 2).mean() + 1e-12))
    rms_w = float(np.sqrt((wet.astype(np.float64) ** 2).mean() + 1e-12))
    if rms_w > 0:
        wet *= rms_x / rms_w
    return wet


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out_path = Path(sys.argv[2])
    result = {"ok": False, "step": "mix"}
    try:
        import librosa
        import numpy as np
        import soundfile as sf

        vocal_path = Path(job["vocal"]).resolve()
        inst_path = Path(job["instrumental"]).resolve()
        out_dir = Path(job["out_dir"]).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        vocal, vr = sf.read(str(vocal_path), dtype="float32", always_2d=True)
        inst, ir = sf.read(str(inst_path), dtype="float32", always_2d=True)
        vocal_mono = vocal.mean(axis=1)
        if vr != ir:
            vocal_mono = librosa.resample(vocal_mono, orig_sr=vr, target_sr=ir)
        # ---- 人声清晰度：高通 + 存在感/明亮度补偿（只作用在人声上） ----
        # 变声后的干声能量容易挤在 300-1k，混进伴奏会显得发闷；这里按固定曲线
        # 做轻微低中频收敛和 1-6k 提亮。clarity=0 关闭，1.0 为默认，1.5 最亮。
        try:
            clarity = float(job.get("clarity", 1.0))
        except (TypeError, ValueError):
            clarity = 1.0
        clarity = max(0.0, min(1.5, clarity))
        if clarity > 0:
            try:
                from scipy.signal import butter, sosfiltfilt
                sos = butter(2, 80 / (ir / 2), btype="highpass", output="sos")
                vocal_mono = sosfiltfilt(sos, vocal_mono).astype(np.float32)
            except Exception:  # noqa: BLE001 - scipy missing: skip the high-pass only
                pass
            pts_f = np.array([20, 80, 120, 250, 400, 700, 1200, 2000, 3500,
                              5000, 7000, 10000, 16000], dtype=np.float64)
            pts_g = np.array([0, 0, -1.0, -2.5, -1.5, 0, 1.0, 4.0, 5.0,
                              4.0, 2.5, 2.0, 1.0], dtype=np.float64) * clarity
            spec = librosa.stft(vocal_mono, n_fft=2048, hop_length=512)
            freqs = librosa.fft_frequencies(sr=ir, n_fft=2048)
            g_db = np.interp(np.log2(np.maximum(freqs, 20)), np.log2(pts_f), pts_g)
            shaped = librosa.istft(spec * (10 ** (g_db / 20.0))[:, None],
                                   hop_length=512, length=len(vocal_mono))
            vocal_mono = shaped.astype(np.float32)
        # ---- 咬字增强：把原始分离人声的辅音/齿音补回变声结果 ----
        # 去噪会把 3-8k 的辅音噪声削掉，变声后咬字会发糊；这里把原始人声
        # 高通后的高频按能量门限叠回一小部分，恢复字头。consonant=0 关闭。
        try:
            consonant = float(job.get("consonant", 0.35))
        except (TypeError, ValueError):
            consonant = 0.35
        consonant = max(0.0, min(0.8, consonant))
        ref_path = str(job.get("reference_vocal") or "")
        if consonant > 0 and ref_path and Path(ref_path).is_file():
            try:
                ref_audio, ref_sr = sf.read(ref_path, dtype="float32", always_2d=True)
                ref_mono = ref_audio.mean(axis=1)
                if int(ref_sr) != int(ir):
                    ref_mono = librosa.resample(ref_mono, orig_sr=int(ref_sr), target_sr=int(ir))
                n_ref = min(len(ref_mono), len(vocal_mono))
                if n_ref > 0:
                    from scipy.signal import butter as _butter
                    from scipy.signal import sosfiltfilt as _sos
                    sos = _butter(4, 3800 / (ir / 2), btype="highpass", output="sos")
                    ref_hf = _sos(sos, ref_mono[:n_ref]).astype(np.float32)
                    env = librosa.feature.rms(y=ref_hf, frame_length=1024, hop_length=256)[0]
                    scale = float(np.percentile(env, 90)) + 1e-9
                    gate = np.interp(np.arange(len(ref_hf)),
                                     np.arange(len(env)) * 256,
                                     np.minimum(1.0, env / scale)).astype(np.float32)
                    add = ref_hf * gate * consonant
                    vocal_mono = np.pad(vocal_mono, (0, max(0, n_ref - len(vocal_mono))))
                    vocal_mono = vocal_mono.copy()
                    vocal_mono[:n_ref] += add
            except Exception:  # noqa: BLE001 - restoration is best effort
                pass
        if inst.shape[1] == 1:
            inst = np.repeat(inst, 2, axis=1)
        # 伴奏移调：为全局音域适配预留。人声整体 +N 半音时，伴奏应 -N 半音
        # 以保持成品回到原调。默认 0 / 缺失不移调，幂等。
        try:
            shift = int(job.get("accompaniment_shift") or 0)
        except (TypeError, ValueError):
            shift = 0
        if shift != 0:
            shifted = [librosa.effects.pitch_shift(inst[:, ch], sr=ir, n_steps=-shift)
                       for ch in range(inst.shape[1])]
            inst = np.stack(shifted, axis=1)
        length = max(len(inst), len(vocal_mono))
        inst2 = np.zeros((length, inst.shape[1]), dtype=np.float32)
        inst2[: len(inst)] = inst
        voc2 = np.zeros(length, dtype=np.float32)
        voc2[: len(vocal_mono)] = vocal_mono

        # Keep the accompaniment at its original level; the generated vocal is
        # normalized to a comparable RMS so the mix is not dominated by either.
        inst_rms = float(np.sqrt((inst2 ** 2).mean() + 1e-12))
        voc_rms = float(np.sqrt((voc2 ** 2).mean() + 1e-12))
        gain = min(4.0, max(0.25, inst_rms / max(voc_rms, 1e-6) * float(job.get("vocal_gain", 1.0))))
        try:
            vocal_level_db = float(job.get("vocal_level_db", -2.0))
        except (TypeError, ValueError):
            vocal_level_db = -2.0
        vocal_level_db = max(-12.0, min(6.0, vocal_level_db))
        gain *= 10 ** (vocal_level_db / 20.0)
        try:
            bgm_gain_db = float(job.get("bgm_gain_db", 0.0))
        except (TypeError, ValueError):
            bgm_gain_db = 0.0
        bgm_gain_db = max(-12.0, min(12.0, bgm_gain_db))
        inst2 *= 10 ** (bgm_gain_db / 20.0)
        voc_gained = _limit(voc2 * gain, ir, ceiling=0.95)

        # 人声混响 + 立体声：干声居中，混响左右用不同 IR 拉出宽度
        try:
            wet_amount = float(job.get("vocal_reverb", 0.12))
        except (TypeError, ValueError):
            wet_amount = 0.12
        wet_amount = max(0.0, min(0.6, wet_amount))
        if wet_amount > 0:
            wet_l = _reverb(voc_gained, ir, seed=11)
            wet_r = _reverb(voc_gained, ir, seed=29)
        else:
            wet_l = wet_r = np.zeros_like(voc_gained)
        dry_gain = 1.0 - 0.5 * wet_amount
        vocal_stereo = np.stack([
            voc_gained * dry_gain + wet_l * wet_amount,
            voc_gained * dry_gain + wet_r * wet_amount,
        ], axis=1).astype(np.float32)

        if inst2.shape[1] == 1:
            inst2 = np.repeat(inst2, 2, axis=1)
        mix = inst2 + vocal_stereo

        # 立体声加宽（M/S）：单声道内容保持不变，只把左右差异放大。
        # 1.0 = 保持伴奏原样（相位最干净）；>1 会让伴奏更宽，但中间声部可能变薄。
        try:
            width = float(job.get("stereo_width", 1.0))
        except (TypeError, ValueError):
            width = 1.0
        width = max(1.0, min(2.0, width))
        if width > 1.0 and mix.shape[1] == 2:
            mid = (mix[:, 0] + mix[:, 1]) * 0.5
            side = (mix[:, 0] - mix[:, 1]) * 0.5 * width
            mix = np.stack([mid + side, mid - side], axis=1).astype(np.float32)
        # Same rule for the mix: limit the peaks instead of scaling the whole
        # song down by whatever the loudest single sample was.
        mix = _limit(mix, ir, ceiling=0.97)
        final_path = out_dir / "final.wav"
        sf.write(str(final_path), mix, ir, subtype="PCM_24")
        vocal_out = out_dir / "vocal_only.wav"
        sf.write(str(vocal_out), vocal_stereo, ir, subtype="PCM_24")
        result.update({
            "ok": True,
            "final": str(final_path),
            "vocal_only": str(vocal_out),
            "sample_rate": ir,
            "duration_s": round(length / ir, 2),
            "vocal_gain": round(gain, 3),
            "vocal_peak": round(float(np.max(np.abs(voc_gained))), 4),
            "mix_peak": round(float(np.max(np.abs(mix))), 4),
            "vocal_rms_after": round(float(np.sqrt((voc_gained.astype(np.float64) ** 2).mean())), 5),
            "mix_rms": round(float(np.sqrt((mix.astype(np.float64) ** 2).mean())), 5),
            "accompaniment_rms": round(inst_rms, 5),
            "vocal_rms_before": round(voc_rms, 5),
            "accompaniment_shift": shift,
            "channels": int(mix.shape[1]),
            "vocal_reverb": round(wet_amount, 3),
            "stereo_width": round(width, 2),
            "clarity": round(clarity, 2),
            "vocal_level_db": round(vocal_level_db, 2),
            "bgm_gain_db": round(bgm_gain_db, 2),
            "consonant": round(consonant, 2),
        })
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["trace"] = traceback.format_exc()[-2000:]
    out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
