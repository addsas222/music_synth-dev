"""DDSP-SVC 6.3 RectifiedFlow training worker.

Runs ``train_reflow.py`` inside the DDSP venv as a subprocess with ``cwd`` set
to the repo root (the entry script and the vocoder both load assets by relative
paths).  The worker itself stays light: it imports nothing heavy, only builds a
modified copy of ``configs/reflow.yaml`` with the caller's overrides and then
shells out.

Contract (same as every other worker)::

    <ddsp python> -I train_worker.py <job.json> <result.json>

The result file is always written, even on failure.  The caller (``ms_train``)
passes a job dict whose keys are filtered against ``main()``'s signature before
they are forwarded here, so unknown keys never leak into the subprocess.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

# Keys we are willing to accept from the caller.  Anything else is dropped so a
# stray/unknown key can never reach the DDSP subprocess or the config file.
ALLOWED_KEYS = {
    "repo", "python", "train_dir", "epochs", "batch_size", "device",
    "progress_out",
}


def _filter_job(job: dict) -> dict:
    """Keep only the keys the worker declares; forward nothing unknown."""
    return {k: v for k, v in job.items() if k in ALLOWED_KEYS}


def _run(cmd: list[str], cwd: Path, timeout: float, progress_out: Path | None,
         tail_size: int = 3000) -> tuple[int, str]:
    env = dict(os.environ)
    for key in ("PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV", "CONDA_PREFIX"):
        env.pop(key, None)
    # The DDSP repo imports its own local modules (slicer.py, ddsp/, reflow/,
    # optimizer/, ...) that are NOT installed into site-packages.  ``-I`` drops
    # the script dir and cwd from sys.path, so prepend the repo root to
    # PYTHONPATH.  (PYTHONPATH is the one path source ``-I`` does NOT ignore.)
    repo_path = str(cwd)
    existing = env.pop("PYTHONPATH", "")
    env["PYTHONPATH"] = repo_path + (os.pathsep + existing if existing else "")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    proc = subprocess.Popen(
        cmd, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert proc.stdout is not None
    tail: list[str] = []
    try:
        for raw in proc.stdout:
            line = raw.decode("utf-8", "replace").rstrip()
            if not line:
                continue
            tail.append(line)
            tail = tail[-40:]
            if progress_out is not None:
                try:
                    progress_out.write_text(
                        json.dumps({"last_line": line[-200:]}, ensure_ascii=False),
                        encoding="utf-8")
                except OSError:
                    pass
    finally:
        code = proc.wait()
    return code, "\n".join(tail)[-tail_size:]


def _load_yaml(path: Path) -> dict:
    import yaml  # noqa: PLC0415 - only the ddsp venv has it, imported lazily

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _find_latest_ckpt(expdir: Path) -> str:
    if not expdir.is_dir():
        return ""
    candidates = [p for p in expdir.glob("model_*.pt") if p.is_file()]
    if not candidates:
        return ""
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out_path = Path(sys.argv[2])
    result: dict = {"ok": False, "step": "train"}
    try:
        job = _filter_job(job)
        repo = Path(job["repo"]).resolve()
        python = Path(job["python"]).resolve()
        train_dir = Path(job["train_dir"]).resolve()
        epochs = int(job.get("epochs") or 3000)
        batch_size = int(job.get("batch_size") or 8)
        device = str(job.get("device") or "auto")
        progress_out = Path(job["progress_out"]) if job.get("progress_out") else None

        if not python.is_file():
            raise RuntimeError(f"DDSP 环境 Python 不存在：{python}")
        entry = repo / "train_reflow.py"
        if not entry.is_file():
            raise RuntimeError(f"DDSP 训练入口不存在：{entry}")
        if not (repo / "configs" / "reflow.yaml").is_file():
            raise RuntimeError(f"DDSP 配置文件缺失：{repo / 'configs' / 'reflow.yaml'}")
        if not train_dir.is_dir():
            raise RuntimeError(f"训练目录不存在：{train_dir}（需含 audio/ 子目录）")

        # device: 'auto' -> force cuda/cpu explicitly so epochs/batch_size
        # overrides always land in the config we hand to train_reflow.py.
        if device in {"cpu", "cuda"}:
            cfg = _load_yaml(repo / "configs" / "reflow.yaml")
            cfg["device"] = device
        else:
            cfg = _load_yaml(repo / "configs" / "reflow.yaml")
        cfg.setdefault("train", {})["epochs"] = epochs
        cfg["train"]["batch_size"] = batch_size
        import yaml  # noqa: PLC0415
        run_cfg = repo / "cache" / f"reflow_run_{os.getpid()}.yaml"
        run_cfg.parent.mkdir(parents=True, exist_ok=True)
        run_cfg.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                           encoding="utf-8")
        config_path = run_cfg

        cmd = [str(python), "-X", "utf8", str(entry), "-c", str(config_path)]
        timeout = float(job.get("timeout") or (3600 * 24))
        code, tail = _run(cmd, repo, timeout, progress_out)

        expdir = repo / "exp" / "reflow-test"
        ckpt = _find_latest_ckpt(expdir)
        if code != 0:
            result["error"] = f"DDSP 训练退出码 {code}：{tail}"
            result["tail"] = tail
            out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            return 1
        result.update({
            "ok": True,
            "ckpt": ckpt,
            "expdir": str(expdir),
            "epochs": epochs,
            "batch_size": batch_size,
            "device": device,
            "tail": tail,
        })
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["trace"] = traceback.format_exc()[-3000:]
    out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
