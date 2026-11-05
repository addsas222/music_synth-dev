"""ACE-Step 音色 LoRA 训练 worker（在 DDSP 环境 + acestep/pylibs 下运行）。

流程：
  1. 扫描素材目录（wav/flac/mp3...），生成 dataset.json（caption + lyrics）
  2. preprocess_audio_files：VAE + 文本编码 + DiT 特征 → .pt
  3. training_v2 fixed：LoRA 训练 → <out>/final（adapter_config + adapter_model）
  4. 把 final 复制到 <project_root>/loras/<name>/，供原创生成时加载

job：plugin_dir, project_root, pylibs, audio_dir, name, caption, lyrics,
     epochs, rank, alpha, batch, gradient_accumulation, max_duration
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import traceback
from pathlib import Path

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".wma"}

# 插件里的模型目录名 -> ACE-Step training_v2 的 variant 别名
MODEL_ALIASES = {
    "acestep-v15-turbo": "turbo",
    "acestep-v15-base": "base",
    "acestep-v15-sft": "sft",
    "acestep-v15-xl-turbo": "xl_turbo",
    "acestep-v15-xl-base": "xl_base",
    "acestep-v15-xl-sft": "xl_sft",
}


def _safe_name(text: str) -> str:
    # 60 chars: the plugin appends a variant suffix (e.g. _xl_turbo) when
    # training several base models; truncating at 40 could collide.
    name = re.sub(r'[\\/:*?"<>|\s]+', "_", str(text or "")).strip(" .")[:60]
    return name or "ace_voice"


def _audio_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def _auto_caption(files: list[Path]) -> tuple[str, str]:
    """Listen to a few clips and describe the voice in English (for ACE) + Chinese (UI)."""
    import numpy as np
    import librosa

    f0s, cents = [], []
    for p in files[:6]:
        try:
            y, sr = librosa.load(str(p), sr=22050, mono=True, duration=20)
            if y.size < 4096:
                continue
            f0 = librosa.yin(y, fmin=60, fmax=800, sr=sr, frame_length=2048)
            f0 = f0[np.isfinite(f0)]
            if f0.size:
                f0s.append(float(np.median(f0)))
            S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
            freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
            total = float(S.sum())
            if total > 0:
                cents.append(float((freqs * S.sum(axis=1)).sum() / total))
        except Exception:
            continue
    f0 = float(np.median(f0s)) if f0s else 180.0
    cent = float(np.median(cents)) if cents else 2000.0
    if f0 >= 165:
        gender, zh_gender = "female", "女声"
    elif f0 <= 130:
        gender, zh_gender = "male", "男声"
    else:
        gender, zh_gender = "androgynous", "中性声"
    if f0 >= 260:
        register, zh_reg = "high", "高音区"
    elif f0 < 140:
        register, zh_reg = "low", "低音区"
    else:
        register, zh_reg = "mid", "中音区"
    if cent >= 2200:
        timbre, zh_timbre = "bright", "偏亮"
    elif cent < 1500:
        timbre, zh_timbre = "warm", "偏暖"
    else:
        timbre, zh_timbre = "balanced", "均衡"
    caption = (f"{gender} pop vocal, {register} register, {timbre} tone, "
               "clear, expressive, single singer")
    caption_zh = f"{zh_gender} · {zh_reg} · {zh_timbre}"
    return caption, caption_zh


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out_path = Path(sys.argv[2])
    result = {"ok": False, "step": "ace_train"}
    try:
        plugin_dir = Path(job["plugin_dir"]).resolve()
        project_root = Path(job["project_root"]).resolve()
        pylibs = Path(job["pylibs"]).resolve()
        sys.path.insert(0, str(plugin_dir))
        sys.path.insert(0, str(pylibs))

        import torch
        torch.set_default_dtype(torch.bfloat16)

        from acestep.training.path_safety import set_safe_root
        set_safe_root(str(project_root))

        model_name = str(job.get("model_name") or "acestep-v15-turbo").strip()
        model_alias = str(job.get("model_alias") or MODEL_ALIASES.get(model_name) or "turbo")
        print(f"[ace-train] base model: {model_name} (variant={model_alias})", flush=True)

        # 固定通道：插件直接传「准备素材」里已处理的切片；兼容旧的 audio_dir 参数
        paths = [Path(str(x)) for x in (job.get("audio_paths") or [])
                 if str(x) and Path(str(x)).is_file()]
        if paths:
            files = paths
            audio_dir = files[0].parent
        else:
            audio_dir = Path(job["audio_dir"]).resolve()
            files = _audio_files(audio_dir)
        if not files:
            raise RuntimeError(f"素材目录里没有音频：{audio_dir}")
        if len(files) < 5:
            raise RuntimeError(f"素材太少（{len(files)} 个），至少放 5 段再训练")

        name = _safe_name(job.get("name") or "ace_voice")
        # 不再让用户填英文风格；从素材音高/频谱自动判断
        caption, caption_zh = _auto_caption(files)
        lyrics = "la la la la"
        print(f"[ace-train] auto caption: {caption} ({caption_zh})", flush=True)
        work = project_root / "train" / name
        tensors = work / "tensors"
        train_out = work / "out"
        for d in (work, tensors, train_out):
            d.mkdir(parents=True, exist_ok=True)

        # 1) dataset.json
        import soundfile as sf
        samples = []
        for p in files:
            try:
                dur = float(sf.info(str(p)).duration)
            except Exception:
                dur = 0.0
            samples.append({
                "audio_path": str(p),
                "caption": caption,
                "lyrics": lyrics,
                "duration": round(dur, 2),
            })
        dataset_json = work / "dataset.json"
        dataset_json.write_text(json.dumps(samples, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        print(f"[ace-train] {len(samples)} clips -> {audio_dir}", flush=True)

        # 2) preprocess
        from acestep.training_v2.preprocess import preprocess_audio_files
        pre = preprocess_audio_files(
            audio_dir=str(audio_dir),
            output_dir=str(tensors),
            checkpoint_dir=str(project_root / "checkpoints"),
            variant=model_alias,
            max_duration=float(job.get("max_duration") or 60.0),
            dataset_json=str(dataset_json),
            device=str(job.get("device") or "cuda"),
            precision=str(job.get("precision") or "bf16"),
            progress_callback=lambda i, n, msg: print(f"[preprocess] {msg} ({i}/{n})", flush=True),
        )
        if not pre.get("processed"):
            raise RuntimeError("预处理没有产出任何样本：" + str(pre.get("failed")))
        print(f"[ace-train] preprocessed {pre.get('processed')} samples", flush=True)

        # 3) LoRA training (fixed subcommand)
        from acestep.training_v2.cli.args import build_root_parser
        from acestep.training_v2.cli.train_fixed import run_fixed
        epochs = max(1, min(100, int(job.get("epochs") or 10)))
        args = build_root_parser().parse_args([
            "--plain", "--yes", "fixed",
            "--checkpoint-dir", str(project_root / "checkpoints"),
            "--model-variant", model_alias,
            "--base-model", model_alias,
            "--device", str(job.get("device") or "cuda"),
            "--precision", str(job.get("precision") or "bf16"),
            "--dataset-dir", str(tensors),
            "--output-dir", str(train_out),
            "--epochs", str(epochs),
            "--batch-size", str(max(1, int(job.get("batch") or 1))),
            "--gradient-accumulation", str(max(1, int(job.get("gradient_accumulation") or 4))),
            "--num-workers", "0",
            "--rank", str(max(4, min(128, int(job.get("rank") or 32)))),
            "--alpha", str(max(4, min(256, int(job.get("alpha") or 64)))),
            "--save-every", str(epochs),
        ])
        rc = run_fixed(args)
        if rc not in (0, None):
            raise RuntimeError(f"训练失败，退出码 {rc}")

        final_dir = train_out / "final"
        if not final_dir.is_dir():
            raise RuntimeError("训练结束但没有产出 final 目录")
        saved = project_root / "loras" / name
        if saved.exists():
            shutil.rmtree(saved, ignore_errors=True)
        shutil.copytree(final_dir, saved)
        result.update({
            "ok": True,
            "name": name,
            "model": model_name,
            "model_alias": model_alias,
            "lora_path": str(saved),
            "samples": len(samples),
            "epochs": epochs,
            "caption": caption,
            "caption_zh": caption_zh,
        })
        print(f"[ace-train] saved LoRA to {saved}", flush=True)
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["trace"] = traceback.format_exc()[-6000:]
        print("[ace-train] FAILED: " + result["error"], flush=True)
    finally:
        try:
            out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
