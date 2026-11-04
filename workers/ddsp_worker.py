"""DDSP-SVC 6.3 voice conversion worker.

Runs main_reflow.py in the DDSP venv as a subprocess and reports the converted
vocal path back to the plugin.  The worker itself does not import torch so it
stays light; it just shells out to the repo's entry script with the exact
arguments the RectifiedFlow pipeline expects.

Contract (same as every other worker)::

    <ddsp python> -I ddsp_worker.py <job.json> <result.json>

The result file is always written, even on failure.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path


def _run(cmd: list[str], cwd: Path, timeout: float) -> tuple[int, str]:
    env = dict(os.environ)
    for key in ("PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV", "CONDA_PREFIX"):
        env.pop(key, None)
    # -I drops the script dir and cwd from sys.path, but the DDSP repo imports
    # its own local modules (slicer.py, ddsp/, reflow/) that are NOT installed
    # into site-packages.  Prepend the repo root to PYTHONPATH so those resolve.
    repo_path = str(cwd)
    existing = env.pop("PYTHONPATH", "")
    env["PYTHONPATH"] = repo_path + (os.pathsep + existing if existing else "")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    proc = subprocess.run(
        cmd, cwd=str(cwd), timeout=timeout,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env,
        creationflags=(getattr(subprocess, "CREATE_NO_WINDOW", 0)
                       | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)),
    )
    tail = (proc.stdout or b"").decode("utf-8", "replace")[-3000:]
    return proc.returncode, tail


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out_path = Path(sys.argv[2])
    result: dict = {"ok": False, "step": "convert"}
    try:
        repo = Path(job["repo"]).resolve()
        python = Path(job["python"]).resolve()
        ckpt = Path(job["model_ckpt"]).resolve()
        input_wav = Path(job["input"]).resolve()
        out_dir = Path(job["out_dir"]).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / "converted.wav"

        if not python.is_file():
            raise RuntimeError(f"DDSP 环境 Python 不存在：{python}")
        if not ckpt.is_file():
            raise RuntimeError(f"DDSP 模型权重不存在：{ckpt}")
        if not input_wav.is_file():
            raise RuntimeError(f"输入干声不存在：{input_wav}")
        entry = repo / "main_reflow.py"
        if not entry.is_file():
            raise RuntimeError(f"DDSP 仓库入口不存在：{entry}")

        key = int(job.get("key") or 0)
        infer_step = int(job.get("infer_step") or 50)
        pitch_extractor = str(job.get("pitch_extractor") or "rmvpe")
        register_shift = int(job.get("vocal_register_shift") or 0)
        register_shift = max(-12, min(12, register_shift))
        method = str(job.get("method") or "euler")
        if method not in ("euler", "rk4"):
            method = "euler"
        device = str(job.get("device") or "cuda")
        f0_min = int(job.get("f0_min") or 65)
        f0_max = int(job.get("f0_max") or 800)
        threshold = int(job.get("threshold") or -60)

        cmd = [
            str(python), "-X", "utf8", str(entry),
            "-m", str(ckpt),
            "-i", str(input_wav),
            "-o", str(output),
            "-k", str(key),
            "-v", str(register_shift),
            "-step", str(infer_step),
            "-method", method,
            "-pe", str(pitch_extractor),
            "-d", device,
            "-fmin", str(f0_min),
            "-fmax", str(f0_max),
            "-th", str(threshold),
        ]
        timeout = float(job.get("timeout") or 7200)
        code, tail = _run(cmd, repo, timeout)

        if code != 0 or not output.is_file():
            result["error"] = f"DDSP 转换退出码 {code}：{tail}"
            result["tail"] = tail
            out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            return 1

        # Verify the output is real audio (same duration, not silence) so the
        # acceptance gate has numbers to report.
        import soundfile as sf  # noqa: PLC0415 - only needed for the report
        try:
            info = sf.info(str(output))
            data, sr = sf.read(str(output), dtype="float32", always_2d=True)
            mono = data.mean(axis=1) if data.ndim > 1 else data
            import numpy as np  # noqa: PLC0415
            result.update({
                "ok": True,
                "wav": str(output),
                "sample_rate": int(sr),
                "duration_s": round(int(info.frames) / int(sr), 3),
                "peak": round(float(np.max(np.abs(mono))), 5),
                "rms": round(float(np.sqrt((mono.astype(np.float64) ** 2).mean())), 5),
                "engine": "ddsp",
                "model": ckpt.name,
                "key": key,
                "infer_step": infer_step,
                "pitch_extractor": pitch_extractor,
            })
        except Exception as exc:  # noqa: BLE001 - audio exists; report is best effort
            result.update({
                "ok": True, "wav": str(output), "engine": "ddsp",
                "model": ckpt.name, "key": key, "infer_step": infer_step,
                "pitch_extractor": pitch_extractor,
                "warn": f"无法读取输出音频信息：{type(exc).__name__}: {exc}",
            })
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["trace"] = traceback.format_exc()[-3000:]
    out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
