"""SoulX-Singer SVC worker: prompt timbre + target vocal -> converted vocal.

Runs inside the dedicated SoulX venv (`<compute_root>/envs/soulx`).  It uses
SoulX's official SVC checkpoint (`model-svc.pt`) and the RMVPE model from the
preprocess package to build the F0 contours; the output is resampled to the
plugin's 48 kHz working rate so the existing mix worker can use it directly.

Job:
    repo, model_path, config_path, rmvpe_path,
    prompt_wav, target_wav, out_dir, device,
    n_steps, cfg, fp16, auto_shift
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np


def _load_mono(path: str, sample_rate: int):
    import librosa
    import soundfile as sf
    import torch

    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    return torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out_path = Path(sys.argv[2])
    result = {"ok": False, "step": "soulx_svc"}
    try:
        repo = Path(str(job["repo"])).resolve()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        if str(repo / "preprocess") not in sys.path:
            sys.path.insert(0, str(repo))

        import soundfile as sf
        import torch
        from omegaconf import OmegaConf

        from cli.inference_svc import build_model
        from preprocess.tools.f0_extraction import F0Extractor

        device = str(job.get("device") or "cuda")
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        use_fp16 = bool(job.get("fp16", True)) and device.startswith("cuda")
        config = OmegaConf.load(str(job["config_path"]))
        sr = int(config.audio.sample_rate)
        hop = int(config.audio.hop_size)
        out_dir = Path(str(job["out_dir"]))
        out_dir.mkdir(parents=True, exist_ok=True)

        model = build_model(
            model_path=str(job["model_path"]),
            config=config,
            device=device,
            use_fp16=use_fp16,
        )

        f0ext = F0Extractor(
            model_path=str(job["rmvpe_path"]),
            device=device,
            target_sr=sr,
            hop_size=hop,
            verbose=False,
        )
        prompt_f0 = out_dir / "prompt_f0.npy"
        target_f0 = out_dir / "target_f0.npy"
        f0ext.process(str(job["prompt_wav"]), f0_path=str(prompt_f0), verbose=False)
        f0ext.process(str(job["target_wav"]), f0_path=str(target_f0), verbose=False)

        pt_wav = _load_mono(str(job["prompt_wav"]), sr).to(device)
        gt_wav = _load_mono(str(job["target_wav"]), sr).to(device)
        pt_f0 = torch.from_numpy(np.load(prompt_f0)).unsqueeze(0).to(device)
        gt_f0 = torch.from_numpy(np.load(target_f0)).unsqueeze(0).to(device)

        with torch.no_grad():
            audio, shift = model.infer(
                pt_wav=pt_wav,
                gt_wav=gt_wav,
                pt_f0=pt_f0,
                gt_f0=gt_f0,
                auto_shift=bool(job.get("auto_shift", False)),
                pitch_shift=0,
                n_steps=int(job.get("n_steps") or 32),
                cfg=float(job.get("cfg") or 3.0),
                use_fp16=use_fp16,
            )
        audio = audio.squeeze().float().cpu().numpy()
        if sr != 48000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
        # SVC 的输出电平基本跟着 prompt 走：提示样本偏轻时整条干声会低 5–8 dB。
        # 进 DDSP / 混音前先把输出对齐到原干声的 RMS，峰值再软限到 0.95。
        level_matched = bool(job.get("match_input_level", False))
        if level_matched:
            ref = gt_wav.detach().float().cpu().numpy().reshape(-1)
            ref_rms = float(np.sqrt(np.mean(ref ** 2)) + 1e-9)
            out_rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-9)
            if ref_rms > 1e-5 and out_rms > 1e-5:
                audio = audio * (ref_rms / out_rms)
            max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
            if max_abs > 0.95:
                audio = audio * (0.95 / max_abs)
        peak = float(np.max(np.abs(audio)) + 1e-9)
        rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-9)
        target = out_dir / "converted.wav"
        sf.write(str(target), audio, 48000, subtype="PCM_24")
        result.update({
            "ok": True,
            "wav": str(target),
            "duration_s": round(len(audio) / 48000.0, 3),
            "peak": round(peak, 5),
            "rms": round(rms, 5),
            "pitch_shift": int(shift) if isinstance(shift, (int, float)) else 0,
            "engine": "soulx-svc",
            "level_matched": level_matched,
        })
        print(f"[soulx-svc] saved {target}", flush=True)
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["trace"] = traceback.format_exc()[-6000:]
        print("[soulx-svc] FAILED: " + result["error"], flush=True)
    finally:
        try:
            out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
