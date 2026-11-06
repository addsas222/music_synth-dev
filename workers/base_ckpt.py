"""Inspect / normalize singing-voice base checkpoints for DDSP-SVC 6.x.

Supported containers:
  * DDSP-SVC 6.x:  ``{'global_step': int, 'model': {'ddsp_model.*', 'reflow_model.*'}}``
  * SVCFusion .sf_dlc: a torch zip whose ``files['model_0.pt']`` holds the real dict
  * plain state_dict / ``{'state_dict': ...}`` / ``{'weight': ...}`` / ``{'model': ...}``
  * DataParallel ``module.`` prefixes are stripped

Other frameworks (RVC / so-vits-svc) are detected and rejected with a clear
message: their tensor layouts are not compatible with DDSP-SVC, and silently
loading them would produce a broken model.

CLI contract (same as the other workers):
    <python> base_ckpt.py <job.json> <result.json>
job: {"path": "<checkpoint>", "convert": "<optional output .pt>"}
result: {ok, framework, arch, path, converted, error}
"""
from __future__ import annotations

import json
import re
import sys
import traceback
from pathlib import Path

_PREFIXES = ("model.", "module.", "net.", "state_dict.")


def _torch():
    import torch  # noqa: PLC0415 - only available in the DDSP venv
    return torch


def load_raw(path: str | Path):
    """Load a checkpoint, unwrapping the .sf_dlc container if needed."""
    torch = _torch()
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and isinstance(ckpt.get("files"), dict):
        inner = ckpt["files"].get("model_0.pt")
        if inner is not None:
            ckpt = inner
    return ckpt


def unwrap(ckpt) -> tuple[dict | None, int]:
    """Return (state_dict, global_step) from the common wrapper shapes."""
    gs = 0
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "weight", "net", "model_state_dict"):
            value = ckpt.get(key)
            if isinstance(value, dict) and value and \
                    all(hasattr(v, "shape") for v in list(value.values())[:8]):
                try:
                    gs = int(ckpt.get("global_step") or ckpt.get("step") or 0)
                except (TypeError, ValueError):
                    gs = 0
                return value, gs
        values = list(ckpt.values())
        if values and all(hasattr(v, "shape") for v in values[:8]):
            return ckpt, 0
    return None, 0


def strip_prefix(sd: dict) -> dict:
    if any(k.startswith("module.") for k in sd):
        return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    return sd


def detect_framework(sd: dict) -> str:
    keys = list(sd)
    if any(k.startswith("ddsp_model.") for k in keys) and \
            any(k.startswith("reflow_model.") for k in keys):
        return "ddsp-svc-reflow"
    if any(k.startswith("enc_p.") for k in keys) or \
            any(k.startswith("dec.") for k in keys) or "emb_g.weight" in keys:
        return "rvc"
    if any(k.startswith("enc_p.") for k in keys) or \
            any(k.startswith("flow.") for k in keys) or \
            any(k.startswith("enc_q.") for k in keys):
        return "so-vits-svc"
    return "unknown"


def _scalar(sd: dict, key: str, default):
    value = sd.get(key)
    try:
        return float(value.reshape(-1)[0])
    except Exception:  # noqa: BLE001
        return default


def _count_layers(sd: dict, pattern: str) -> int:
    found = set()
    for key in sd:
        m = re.search(pattern, key)
        if m:
            try:
                found.add(int(m.group(1)))
            except (TypeError, ValueError):
                pass
    return (max(found) + 1) if found else 0


def infer_arch(sd: dict) -> dict:
    """Infer the DDSP-SVC model/data architecture from tensor shapes."""
    arch = {
        "framework": "ddsp-svc-reflow",
        "sampling_rate": int(_scalar(sd, "ddsp_model.sampling_rate", 44100)),
        "block_size": int(_scalar(sd, "ddsp_model.block_size", 512)),
        "win_length": int(_scalar(sd, "ddsp_model.win_length", 2048)),
        "n_layers": _count_layers(sd, r"velocity_fn\.residual_layers\.(\d+)\."),
        "n_aux_layers": _count_layers(sd, r"decoder\.encoder_layers\.(\d+)\."),
        "use_pitch_aug": any("aug_shift_embed" in k for k in sd),
        "n_spk": 1,
    }
    proj = sd.get("reflow_model.velocity_fn.input_projection.weight")
    if proj is not None and getattr(proj, "ndim", 0) == 2:
        arch["n_chans"] = int(proj.shape[0])
        arch["mel_bins"] = int(proj.shape[1])
    cond = sd.get("ddsp_model.unit2ctrl.stack2.0.weight_v")
    if cond is not None and getattr(cond, "ndim", 0) == 3:
        arch["n_aux_chans"] = int(cond.shape[0])
    enc_in = sd.get("ddsp_model.unit2ctrl.stack.0.weight_v")
    enc_out = int(enc_in.shape[1]) if enc_in is not None and enc_in.ndim == 3 else 0
    if enc_out == 256:
        arch.update(encoder="hubertsoft", encoder_hop_size=320,
                    encoder_out_channels=256)
    elif enc_out == 768:
        arch.update(encoder="contentvec768l12tta2x", encoder_hop_size=160,
                    encoder_out_channels=768)
    else:
        arch.update(encoder="", encoder_hop_size=0, encoder_out_channels=enc_out)
    return arch


def inspect_checkpoint(path: str | Path) -> dict:
    raw = load_raw(path)
    sd, global_step = unwrap(raw)
    if sd is None:
        raise ValueError("文件里读不到模型权重（不是 PyTorch checkpoint）")
    sd = strip_prefix(sd)
    framework = detect_framework(sd)
    info = {
        "ok": True,
        "path": str(path),
        "framework": framework,
        "global_step": global_step,
        "tensors": len(sd),
        "arch": {},
        "supported": framework == "ddsp-svc-reflow",
    }
    if framework == "ddsp-svc-reflow":
        info["arch"] = infer_arch(sd)
    else:
        info["error"] = (
            f"检测到 {framework} 底模；DDSP-SVC 训练只接受 DDSP-SVC 6.x "
            f"RectifiedFlow 底模，无法直接转换")
    return info


def normalize_checkpoint(src: str | Path, dst: str | Path) -> dict:
    """Validate + write a plain ``{'global_step', 'model'}`` checkpoint."""
    info = inspect_checkpoint(src)
    if not info.get("supported"):
        raise ValueError(info.get("error") or "不支持的底模格式")
    raw = load_raw(src)
    sd, global_step = unwrap(raw)
    sd = strip_prefix(sd or {})
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch = _torch()
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    torch.save({"global_step": int(global_step or 0), "model": sd}, str(tmp))
    tmp.replace(dst)
    info["converted"] = str(dst)
    info["converted_size"] = dst.stat().st_size
    return info


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out_path = Path(sys.argv[2])
    result = {"ok": False, "step": "base_ckpt"}
    try:
        info = inspect_checkpoint(str(job["path"]))
        convert = str(job.get("convert") or "").strip()
        if convert and info.get("supported"):
            info = normalize_checkpoint(str(job["path"]), convert)
        result.update(info)
        result["ok"] = bool(info.get("supported"))
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["trace"] = traceback.format_exc()[-3000:]
    finally:
        try:
            out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
