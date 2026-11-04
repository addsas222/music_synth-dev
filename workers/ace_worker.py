"""ACE-Step text2music worker（在 envs/ace 的独立 Python 里运行）。

输入 job（由插件 ms_ace_generate 写入）：
    plugin_dir, project_root, output_dir, prompt, lyrics, vocal_language,
    duration, bpm, keyscale, timesignature, instrumental, inference_steps,
    guidance_scale, seed, name

输出 result：
    {ok, wav, audios, prompt, lyrics, duration, error}

这个 worker 只负责“生成一首完整的歌”，不做分离/变声；后续由插件现有的
分离 → DDSP 变声 → 混音链路处理。
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out_path = Path(sys.argv[2])
    result = {"ok": False, "step": "ace_generate"}
    try:
        plugin_dir = Path(job["plugin_dir"]).resolve()
        sys.path.insert(0, str(plugin_dir))
        pylibs = Path(str(job.get("pylibs") or ""))
        if str(pylibs) and pylibs.is_dir():
            sys.path.insert(0, str(pylibs))
        import ace_runtime  # noqa: E402

        def maybe_float(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def maybe_int(value):
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        duration = maybe_float(job.get("duration"))
        bpm = maybe_int(job.get("bpm"))
        request = ace_runtime.AceStepGenerateRequest(
            prompt=str(job.get("prompt") or ""),
            lyrics=str(job.get("lyrics") or ""),
            task_type="text2music",
            instrumental=bool(job.get("instrumental")),
            vocal_language=str(job.get("vocal_language") or "unknown"),
            bpm=bpm,
            keyscale=str(job.get("keyscale") or ""),
            timesignature=str(job.get("timesignature") or ""),
            duration=duration if duration and duration > 0 else -1.0,
            inference_steps=max(1, min(60, int(job.get("inference_steps") or 12))),
            guidance_scale=float(job.get("guidance_scale") or 7.0),
            seed=maybe_int(job.get("seed")) if maybe_int(job.get("seed")) is not None else -1,
            use_random_seed=bool(job.get("use_random_seed", True)),
            audio_format="wav",
        )
        runtime_config = ace_runtime.AceStepRuntimeConfig(
            project_root=str(job["project_root"]),
            model_name=str(job.get("model_name") or "acestep-v15-turbo"),
            device=str(job.get("device") or "auto"),
            offload_to_cpu=bool(job.get("offload_to_cpu", True)),
            offload_dit_to_cpu=bool(job.get("offload_dit_to_cpu", True)),
            output_dir=str(job["output_dir"]),
        )
        generated = ace_runtime.generate_with_acestep_local(
            request, runtime_config, save_dir=str(job["output_dir"]))
        if not generated.success:
            raise RuntimeError(generated.error or generated.status_message
                               or "ACE-Step 生成失败")
        if not generated.audio_paths:
            raise RuntimeError("ACE-Step 没有返回音频")
        result.update({
            "ok": True,
            "wav": str(generated.audio_paths[0]),
            "audios": [str(p) for p in generated.audio_paths],
            "prompt": generated.prompt,
            "lyrics": generated.lyrics,
            "duration": (generated.metas or {}).get("duration"),
        })
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["trace"] = traceback.format_exc()[-6000:]
    finally:
        try:
            out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
