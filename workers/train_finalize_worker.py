"""DDSP-SVC 6.3 训练保留 worker（finalize）。

契约（train_integration_plan_20261103.md 第 2 节）：:

    <ddsp python> train_finalize_worker.py <job.json> <result.json>

job = {repo, run_dir, step, name}
result = {ok, ckpt, config, trash_manifest}

职责（计划第 1 节第 6 步「保留」）：
  1. 将所选里程碑 ckpt（model_<step>.pt）+ config.yaml 落到 exp\<name>\
     （重名自动加后缀 name_2、name_3 …）。
  2. 其余里程碑 ckpt 移入 E:\\NekoMusicSynth\\_trash_<日期>\\train_<run_id>\\
     并写清单（路径/大小/sha256），**不永久删除**。
  3. 不切换 data/config.json 的 ddsp_ckpt（由用户另行点「设为当前」）。

不改动 repo/data、不切 ddsp_ckpt。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write(path: Path, obj: dict) -> None:
    tmp = path.with_name(".finalize.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _unique_name(exp_root: Path, base: str) -> str:
    if not (exp_root / base).exists():
        return base
    i = 2
    while (exp_root / f"{base}_{i}").exists():
        i += 1
    return f"{base}_{i}"


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out_path = Path(sys.argv[2])
    result: dict = {"ok": False, "step": "train_finalize"}

    try:
        repo = Path(job["repo"]).resolve()
        run_dir = Path(job["run_dir"]).resolve()
        step = int(job["step"])
        name = str(job.get("name") or "训练音色").strip() or "训练音色"

        expdir = repo / "exp" / f"_run_{run_dir.name}"
        if not expdir.is_dir():
            raise RuntimeError(f"训练 expdir 不存在：{expdir}")

        ckpt = expdir / f"model_{step}.pt"
        if not ckpt.is_file():
            raise RuntimeError(f"所选 ckpt 不存在：{ckpt}")

        config = expdir / "config.yaml"
        if not config.is_file():
            raise RuntimeError(f"config.yaml 不存在：{config}")

        # 1. 落到 exp\<name>\（重名加后缀）
        exp_root = repo / "exp"
        final_name = _unique_name(exp_root, name)
        final_dir = exp_root / final_name
        final_dir.mkdir(parents=True, exist_ok=True)
        final_ckpt = final_dir / f"model_{step}.pt"
        shutil.copy2(ckpt, final_ckpt)
        # config.yaml 里的 expdir 改为新目录（相对路径）
        try:
            import yaml  # noqa: PLC0415
            cfg = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            cfg = {}
        cfg["env"] = cfg.get("env") or {}
        cfg["env"]["expdir"] = f"exp/{final_name}"
        (final_dir / "config.yaml").write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

        # 2. 其余里程碑 ckpt → _trash（含清单，不永久删）
        trash_root = Path(os.environ.get("NEKO_TRASH_ROOT",
                          "E:/NekoMusicSynth")) / f"_trash_{datetime.now().strftime('%Y%m%d')}"
        trash_dir = trash_root / f"train_{run_dir.name}"
        trash_dir.mkdir(parents=True, exist_ok=True)

        manifest = {"run_id": run_dir.name, "name": name, "kept_step": step,
                    "kept_to": str(final_dir), "moved": []}
        for p in sorted(expdir.glob("model_*.pt")):
            if p.name == f"model_{step}.pt":
                continue
            size = p.stat().st_size
            dst = trash_dir / p.name
            shutil.move(str(p), str(dst))
            manifest["moved"].append({
                "path": str(dst), "size": size, "sha256": _sha256(dst),
            })
        (trash_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        # 保留的 ckpt 在原 expdir 里保留（不删），其余已移走。
        # 同时把保留 ckpt 也复制一份到 expdir 原位置（保留现场可续训）。
        result.update({
            "ok": True,
            "ckpt": str(final_ckpt),
            "config": str(final_dir / "config.yaml"),
            "final_name": final_name,
            "name": final_name,
            "trash_manifest": str(trash_dir / "manifest.json"),
        })
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
