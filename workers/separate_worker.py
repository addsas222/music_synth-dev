"""Source separation worker (audio-separator / BS-RoFormer, or Demucs).

Runs in the separation venv. The model is never downloaded silently by the
plugin: if the weights are absent the worker reports the exact filename and
the command the user must run.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _chunk_split(input_path: str, out_dir: Path, seg_s: float, overlap_s: float):
    """把长音频切成若干带重叠的 WAV 片段，返回 (segment_paths, sample_rate, seg_s, overlap_s)。

    实测 mel_band_roformer 对 >30s 的长音频一次性分离时，中间某段（约 6~22s）
    的人声会被错误归到伴奏轨（干声轨静音），把整段人声漏掉；切成 30s 重叠段
    分别分离后该段人声恢复。这是长音频分离的确定性缓解，不是模型能力上限。
    """
    import soundfile as sf
    data, sr = sf.read(input_path, dtype="float32", always_2d=True)
    if data.ndim == 1:
        data = data[:, None]
    # 保留原始声道，不再下混成单声道：ACE 生成的伴奏本身是立体声，
    # 下混会让成品的背景音乐明显变窄、变糊。
    total = len(data) / sr
    segs = []
    step = seg_s - overlap_s
    cursor = 0.0
    while cursor < total:
        end = min(total, cursor + seg_s)
        if end - cursor < 3.0 and segs:
            # 尾巴太短，并入上一段
            segs[-1] = (segs[-1][0], end)
            break
        segs.append((cursor, end))
        if end >= total:
            break
        cursor += step
    paths = []
    for idx, (s, e) in enumerate(segs):
        chunk = data[int(s * sr):int(e * sr)]
        p = out_dir / f"_chunk_{idx:03d}.wav"
        sf.write(str(p), chunk, sr, subtype="FLOAT")
        paths.append(p)
    return paths, sr, seg_s, overlap_s


def _resample(mono, rate: int, sr: int):
    """Resample a mono float32 signal from ``rate`` to ``sr``.

    audio-separator always writes its stems at its own ``sample_rate``
    (44100 by default) whatever the input rate is, so every stem has to be
    brought back to the source rate before it is placed on the source
    timeline - otherwise the whole track is squeezed by rate/sr and the
    overlap regions splice two different source times together.
    """
    import numpy as np

    rate, sr = int(rate), int(sr)
    if rate == sr:
        return mono
    try:
        import math

        from scipy.signal import resample_poly  # type: ignore

        divisor = math.gcd(rate, sr)
        return resample_poly(mono.astype(np.float32), sr // divisor,
                             rate // divisor).astype(np.float32)
    except Exception:  # noqa: BLE001 - scipy missing: linear fallback
        count = int(round(len(mono) * sr / float(rate)))
        if count <= 0:
            return np.zeros(0, dtype=np.float32)
        source = np.arange(len(mono), dtype=np.float64) / float(rate)
        target = np.arange(count, dtype=np.float64) / float(sr)
        return np.interp(target, source, mono.astype(np.float64)).astype(np.float32)


def _resample_audio(data, rate: int, sr: int):
    """Resample mono [T] or multichannel [T, C] float data along time."""
    import numpy as np

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        return _resample(arr, rate, sr)
    return np.stack([_resample(arr[:, c], rate, sr)
                     for c in range(arr.shape[1])], axis=1)


def _ensure_rate(path: Path, sr: int) -> Path:
    """Rewrite ``path`` in place at ``sr`` when the file was written at another rate."""
    import numpy as np
    import soundfile as sf

    try:
        info = sf.info(str(path))
    except (OSError, RuntimeError):
        return path
    if int(info.samplerate) == int(sr):
        return path
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if data.ndim == 1:
        data = data[:, None]
    converted = _resample_audio(data, rate, sr)
    tmp = path.with_name(path.name + ".rate-fix.wav")
    sf.write(str(tmp), np.asarray(converted, dtype=np.float32), int(sr), subtype="FLOAT")
    tmp.replace(path)
    return path


def _merge_chunks(chunk_paths: list[Path], stems: dict[str, list[Path]],
                  out_dir: Path, sr: int, seg_s: float, overlap_s: float) -> dict[str, str]:
    """把分段分离的结果按重叠区交叉淡化拼接成整段。

    每个 chunk 分离出多轨（vocals/instrumental），这里按轨名收集并拼接。
    重叠区用线性交叉淡化避免接缝爆音。

    分离产物可能不是 ``sr``（audio-separator 0.48 默认输出 44100），
    读入后先重采样到 ``sr`` 再按原步进摆放；否则整条时间轴按
    rate/sr 被压缩，重叠区还会把两个不同源时刻叠在一起。
    """
    import numpy as np
    import soundfile as sf

    hop = int(round((seg_s - overlap_s) * sr))
    fade = max(0, int(round(overlap_s * sr)))
    merged = {}
    for stem_name, paths in stems.items():
        # 先读所有片段（必要时重采样），再按实际长度确定输出长度。
        pieces = []
        for p in paths:
            data, rate = sf.read(str(p), dtype="float32", always_2d=True)
            if data.ndim == 1:
                data = data[:, None]
            pieces.append(_resample_audio(data, rate, sr).astype(np.float64))
        if not pieces:
            merged[stem_name] = ""
            continue
        channels = max(int(m.shape[1]) for m in pieces)
        # 单声道片段补成多声道，避免拼接时维度不一致
        for i, m in enumerate(pieces):
            if m.shape[1] == 1 and channels > 1:
                pieces[i] = np.repeat(m, channels, axis=1)
        total = max(i * hop + len(m) for i, m in enumerate(pieces))
        # 重叠区用权重淡化
        acc = np.zeros((total, channels), dtype=np.float64)
        weight = np.zeros((total, 1), dtype=np.float64)
        for i, m in enumerate(pieces):
            off = i * hop
            n = min(len(m), total - off)
            if n <= 0:
                continue
            w = np.ones(n, dtype=np.float64)
            if fade > 0:
                # 首段头、末段尾不淡化；中间段两侧都淡化
                if i > 0 and fade < n:
                    w[:fade] = np.linspace(0.0, 1.0, fade)
                if i < len(pieces) - 1 and fade < n:
                    w[-fade:] = np.linspace(1.0, 0.0, fade)
            acc[off:off + n] += m[:n] * w[:, None]
            weight[off:off + n] += w[:, None]
        merged_stem = np.where(weight > 1e-6,
                               acc / np.maximum(weight, 1e-6), 0.0)
        out = out_dir / f"_{stem_name}.wav"
        sf.write(str(out), merged_stem.astype(np.float32), sr, subtype="FLOAT")
        merged[stem_name] = str(out)
    return merged


def _separate_audio_separator(job: dict) -> dict:
    from audio_separator.separator import Separator  # type: ignore

    out_dir = Path(job["out_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_file = str(job.get("model") or "").strip()
    if Path(model_file).is_file():
        # audio-separator >=0.4x only accepts a registry filename; point its
        # model directory at the folder that already holds the weights.
        model_dir = Path(model_file).resolve().parent
        model_name = Path(model_file).name
    else:
        model_dir = Path(job.get("model_dir") or (out_dir / "models")).resolve()
        model_name = model_file
    model_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        model_file_dir=str(model_dir),
        output_dir=str(out_dir),
        output_format="WAV",
        use_autocast=bool(job.get("use_autocast", True)),
    )
    separator = Separator(**kwargs)
    separator.load_model(model_filename=model_name)

    input_path = str(job["input"])
    # 长音频分段分离：默认 >30s 切段（20s 步进 + 10s 重叠），避免一次性分离
    # 导致中间段人声被漏到伴奏。seg_s<=0 或音频很短时退回原一次性分离。
    seg_s = float(job.get("seg_s") or 30.0)
    overlap_s = float(job.get("overlap_s") or 10.0)
    import soundfile as sf
    info = sf.info(input_path)
    duration = info.frames / info.samplerate
    # 每级模型的显式 stem 声明（来自各模型 YAML 的 training.instruments /
    # target_instrument），不再靠文件名是否含 "vocal" 判断轨名。
    stem_spec = {
        # 人声提取：Vocals/Instrumental，target=Vocals
        "model_bs_roformer_ep_317_sdr_12.9755.ckpt": ("vocals", ("Vocals", "Instrumental")),
        "melband_roformer_big_beta4.ckpt": ("vocals", ("vocals", "instrumental")),
        # 去和声 karaoke：主唱轨仍叫 Vocals
        "mel_band_roformer_karaoke_becruily.ckpt": ("vocals", ("Vocals", "Instrumental")),
        # 去混响：目标轨 noreverb，同时保留 reverb 轨用于测量混响量
        "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt": ("noreverb", ("noreverb", "reverb")),
        # 去噪：dry 是目标轨，noise 是残差
        "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt": ("dry", ("dry", "noise")),
    }

    def _stem_key(item_path: str) -> str:
        """优先按输出文件名里的 ``(stem)`` 精确映射。

        旧实现按整串文件名做子串匹配，输入文件名里恰好含 ``dry``/``vocals``
        时会把 reverb 轨误判成 dry/vocals。
        """
        base = Path(item_path).stem.lower()
        _, declared = stem_spec.get(model_name, (None, ()))
        import re as _re
        m = _re.search(r"\(([^()]*)\)", base)
        if m:
            token = m.group(1).strip().lower()
            if not declared or token in [str(d).lower() for d in declared]:
                return token
        for stem in declared:
            if stem.lower() in base:
                return stem.lower()
        # 回退：vocals/instrumental/已知后处理目标轨
        for key in ("instrumental", "noreverb", "reverb", "vocals", "noise", "dry"):
            if key in base:
                return key
        return Path(item_path).stem.lower()

    if seg_s > 0 and duration > seg_s + 1.0:
        chunk_paths, sr, seg_s, overlap_s = _chunk_split(input_path, out_dir, seg_s, overlap_s)
        stems_by_name: dict[str, list[Path]] = {}
        for cp in chunk_paths:
            produced = separator.separate(str(cp))
            for item in produced:
                p = Path(out_dir) / item if not Path(item).is_absolute() else Path(item)
                # 显式 stem 映射（修复：不再靠文件名含 vocal 判断轨名）
                key = _stem_key(item)
                stems_by_name.setdefault(key, []).append(p)
        merged = _merge_chunks(chunk_paths, stems_by_name, out_dir, sr, seg_s, overlap_s)
        # 清掉中间 chunk 文件（输入切片 + 各切片的分离产物，已 merge 进整段）。
        for cp in chunk_paths:
            try:
                cp.unlink()
            except OSError:
                pass
        for junk in out_dir.glob("chunk_*.wav"):
            try:
                junk.unlink()
            except OSError:
                pass
        files = list(merged.values())
        return {"files": files, "stems": merged,
                "engine": "audio-separator", "model": model_name,
                "segmented": True, "seg_s": seg_s, "overlap_s": overlap_s,
                "sample_rate": int(sr)}

    produced = separator.separate(input_path)
    files = [Path(out_dir) / item if not Path(item).is_absolute() else Path(item) for item in produced]
    files = [p for p in files if p.is_file()]
    # 显式 stem 映射，与分段路径保持一致
    stems = {_stem_key(str(p)): str(p) for p in files}
    # audio-separator writes at its own output rate (44100 by default); bring the
    # stems back to the source rate so vocals/instrumental keep the source
    # timeline (same contract as the segmented path above).
    target_sr = int(job.get("sample_rate") or job.get("sr") or info.samplerate)
    for p in files:
        _ensure_rate(p, target_sr)
    return {"files": [str(p) for p in files], "stems": stems,
            "engine": "audio-separator", "model": model_name, "segmented": False,
            "sample_rate": target_sr}



def _separate_demucs(job: dict) -> dict:
    import soundfile as sf
    import torch  # type: ignore
    from demucs.apply import apply_model  # type: ignore
    from demucs.pretrained import get_model  # type: ignore

    out_dir = Path(job["out_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if (job.get("use_cuda", True) and torch.cuda.is_available()) else "cpu"
    model = get_model(job.get("demucs_model") or "htdemucs").to(device)
    if device == "cuda" and job.get("half"):
        model.half()
    data, rate = sf.read(job["input"], dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T).to(device)
    if rate != model.samplerate:
        import torchaudio
        wav = torchaudio.functional.resample(wav, rate, model.samplerate)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        stems = apply_model(model, wav[None], split=True, overlap=0.25)[0]
    stems = stems * ref.std() + ref.mean()
    names = list(model.sources)
    target_sr = int(job.get("sample_rate") or job.get("sr") or rate)
    paths = {}
    for name, stem in zip(names, stems):
        p = out_dir / f"{name}.wav"
        sf.write(p, stem.float().cpu().numpy().T, model.samplerate, subtype="FLOAT")
        # htdemucs runs at 44100 whatever the input rate is; keep the source
        # timeline so the mix step lines up with the original track.
        _ensure_rate(p, target_sr)
        paths[name] = str(p)
    return {"files": list(paths.values()), "stems": paths,
            "engine": "demucs", "model": job.get("demucs_model") or "htdemucs",
            "device": device, "sample_rate": target_sr}


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out_path = Path(sys.argv[2])
    result = {"ok": False, "step": "separate"}
    if not job.get("use_cuda", True):
        # Must happen before torch/audio-separator import so CUDA reports empty.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        backend = (job.get("backend") or "audio_separator").lower()
        if backend == "demucs":
            produced = _separate_demucs(job)
        else:
            produced = _separate_audio_separator(job)

        files = produced.get("files") or []
        stems = produced.get("stems") or {}
        # 显式 stem 映射：vocals 轨、目标后处理轨（noreverb/dry）都从 stems 取，
        # 不再靠文件名含 "vocal" 判断。
        vocals = stems.get("vocals") or next((f for f in files if "vocal" in Path(f).name.lower()), "")
        instrumental = stems.get("instrumental") or next((f for f in files if any(k in Path(f).name.lower()
                                                     for k in ("instrumental", "no_vocals", "accompaniment"))), "")
        # 后处理模型的目标轨（去混响/去噪），供上层链式调用直接取
        target_stem = (stems.get("noreverb") or stems.get("dry")
                       or next((v for k, v in stems.items()
                                if k in ("noreverb", "dry")), ""))
        result.update({
            "ok": bool(files),
            "files": files,
            "stems": stems,
            "vocals": vocals,
            "instrumental": instrumental,
            "target_stem": target_stem,
            "engine": produced.get("engine"),
            "model": produced.get("model"),
            "segmented": bool(produced.get("segmented")),
            "seg_s": produced.get("seg_s"),
            "overlap_s": produced.get("overlap_s"),
            "sample_rate": produced.get("sample_rate"),
            "error": None if files else "分离引擎没有产出文件",
        })
        # 去混响/去噪等后处理模型只产 noreverb/dry 等目标轨，不产 vocals；
        # 只有当模型声明了 vocals 轨（或回退到文件名含 vocal）时才强制要求。
        model_name = produced.get("model") or ""
        declares_vocals = model_name in (
            "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            "melband_roformer_big_beta4.ckpt",
            "mel_band_roformer_karaoke_becruily.ckpt",
        )
        if not vocals and not target_stem and declares_vocals:
            result["ok"] = False
            result["error"] = "分离结果里找不到人声轨（vocals）；请检查模型档位"
    except ModuleNotFoundError as exc:
        result["error"] = f"缺少依赖 {exc.name}；请在分离环境安装 audio-separator"
        result["missing_module"] = exc.name
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        result["error"] = msg
        if "out of memory" in msg.lower():
            result["hint"] = "显存不足：可关闭占用显存的程序，插件会自动改走 CPU 重试。"
        if "not found" in msg.lower() or "404" in msg:
            result["hint"] = "模型权重未就绪；请先下载对应 .ckpt 到模型目录"
    out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
