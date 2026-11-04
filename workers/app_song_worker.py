"""Per-application song recorder for 为她而唱 (standalone capture).

Launched by ``song_listener.py`` with the *synth* venv Python.  It spawns the
copy of ``application_loopback_worker.py`` using the isolated capture runtime
(managed CPython + proc-tap), reads 48 kHz stereo frames, and cuts a full song
whenever Windows' media session (SMTC) changes track or playback stops.

It writes one WAV per finished song and appends JSON lines to an events file so
the plugin can create a track and honour a "record, then sing" anchor.

No microphone, no system-wide loopback: only the selected process tree.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path

_HEADER = struct.Struct("!cI")
_MAX_PACKET = 2_000_000
_SOFT_CLIP_KNEE = 0.95


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _read_exact(pipe, size: int) -> bytes:
    out = bytearray()
    while len(out) < size:
        part = pipe.read(size - len(out))
        if not part:
            raise EOFError(f"capture pipe closed ({len(out)}/{size})")
        out.extend(part)
    return bytes(out)


def _read_packet(pipe):
    tag, length = _HEADER.unpack(_read_exact(pipe, _HEADER.size))
    if tag not in {b"R", b"D", b"E", b"H"} or length > _MAX_PACKET:
        raise RuntimeError(f"invalid capture frame {tag!r}/{length}")
    return tag, _read_exact(pipe, length)


def _apply_gain(audio, *, target_peak: float = 0.707, max_gain: float = 20.0):
    import numpy as np
    arr = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak <= 1e-6:
        return arr
    gain = min(max_gain, target_peak / peak)
    out = arr if abs(gain - 1.0) < 1e-4 else arr * gain
    over = np.abs(out) > _SOFT_CLIP_KNEE
    if over.any():
        clipped = out[over]
        sign = np.sign(clipped)
        mag = np.abs(clipped)
        soft = _SOFT_CLIP_KNEE + (1.0 - _SOFT_CLIP_KNEE) * np.tanh(
            (mag - _SOFT_CLIP_KNEE) / max(1e-6, 1.0 - _SOFT_CLIP_KNEE))
        out[over] = sign * soft
    return out.astype(np.float32)


def _split_title(text: str) -> tuple[str, str]:
    raw = str(text or "").strip()
    for sep in (" - ", " – ", " — "):
        if sep in raw:
            left, _, right = raw.partition(sep)
            if left.strip() and right.strip():
                return right.strip()[:120], left.strip()[:120]
    return raw[:120], ""


class Capture:
    def __init__(self, job: dict):
        self.job = job
        self.proc = None

    def start(self) -> None:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        argv = [
            str(self.job["capture_python"]), "-I", "-S",
            str(self.job["app_worker"]),
            str(self.job["capture_vendor"]), str(self.job["capture_addon"]),
            str(int(self.job["pid"])), str(self.job["token"]), "stereo",
        ]
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, creationflags=flags, bufsize=0)
        return None

    def packet(self):
        if self.proc is None or self.proc.stdout is None:
            raise RuntimeError("capture process is not running")
        return _read_packet(self.proc.stdout)

    def alive(self) -> bool:
        return bool(self.proc is not None and self.proc.poll() is None)

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=3)
            except Exception:  # noqa: BLE001
                try:
                    self.proc.kill()
                except OSError:
                    pass


def _smtc(job: dict) -> dict:
    """Read the Windows media session once (best effort, bounded)."""
    python = job.get("smtc_python")
    vendor = job.get("smtc_vendor")
    worker = job.get("smtc_worker")
    if not (python and vendor and worker):
        return {}
    if not (Path(str(python)).is_file() and Path(str(worker)).is_file()):
        return {}
    try:
        proc = subprocess.run(
            [str(python), "-I", "-S", str(worker), str(vendor)],
            capture_output=True, timeout=5.0, stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    try:
        data = json.loads((proc.stdout or b"").decode("utf-8", "replace"))
    except (UnicodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _music_session(job: dict, smtc: dict) -> dict | None:
    """Pick the playing music session; prefer the captured executable's app id."""
    wants = str(job.get("exe_name") or "").strip().casefold()
    stem = wants[:-4] if wants.endswith(".exe") else wants
    rows = [row for row in (smtc.get("sessions") or []) if isinstance(row, dict)]
    rows = [row for row in rows if row.get("playing") is True
            and str(row.get("playback_type") or "").lower() == "music"]
    if not rows:
        return None
    if stem:
        for row in rows:
            app = str(row.get("app") or "").casefold()
            if stem and stem in app:
                return row
    return rows[0]


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out_path = Path(sys.argv[2])
    out_dir = Path(job["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    events_path = Path(job["events_path"]); events_path.parent.mkdir(parents=True, exist_ok=True)
    status_path = Path(job["status_path"])
    stop_path = Path(str(job["stop_path"])) if job.get("stop_path") else None
    capture = Capture(job)
    result = {"ok": False, "step": "song_capture"}

    try:
        import numpy as np
        import soundfile as sf
    except Exception as exc:  # noqa: BLE001
        out_path.write_text(json.dumps({"ok": False, "error": f"synth venv 缺少 numpy/soundfile: {exc}"},
                                       ensure_ascii=False), encoding="utf-8")
        return 1

    state = {"state": "waiting", "title": "", "artist": "", "started_at": "",
             "seconds": 0.0, "frames": 0, "songs": 0, "error": "",
             "pid": int(job["pid"]), "mode": "app_channel"}
    try:
        min_s = max(5.0, float(job.get("min_song_s") or 20.0))
        max_s = max(30.0, float(job.get("max_song_s") or 900.0))
        preroll_frames = max(0, min(5, int(job.get("preroll_frames") or 3)))
        silence_end = max(1, int(job.get("silence_end_frames") or 4))
        smtc_every = max(1, int(job.get("smtc_every_frames") or 2))
        sample_rate = 48000
        rolling = []
        current_sf = None
        current_tmp: Path | None = None
        current_meta = {"title": "", "artist": ""}
        current_frames = 0
        current_started = 0.0
        silence_count = 0
        frame_index = 0
        last_status = 0.0
        song_seq = 0
        stopped = False

        def write_status(force: bool = False) -> None:
            nonlocal last_status
            now = time.monotonic()
            if not force and now - last_status < 2.0:
                return
            last_status = now
            if current_sf is not None:
                state.update({"state": "recording",
                              "title": current_meta["title"], "artist": current_meta["artist"],
                              "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_started)),
                              "seconds": round(current_frames / sample_rate, 1)})
            tmp = status_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, status_path)

        def open_song(meta: dict, with_preroll: bool) -> None:
            nonlocal current_sf, current_tmp, current_meta, current_frames, current_started, song_seq
            song_seq += 1
            current_meta = dict(meta)
            current_tmp = out_dir / f"._recording_{os.getpid()}_{song_seq}.wav"
            current_sf = sf.SoundFile(str(current_tmp), mode="w", samplerate=sample_rate,
                                      channels=2, subtype="FLOAT")
            current_frames = 0
            current_started = time.time()
            if with_preroll:
                for frame in rolling:
                    current_sf.write(frame)
                    current_frames += len(frame)

        def finish_song(reason: str = "") -> None:
            nonlocal current_sf, current_tmp, current_frames, song_seq
            if current_sf is None:
                return
            try:
                current_sf.close()
            except Exception:  # noqa: BLE001
                pass
            tmp = current_tmp
            current_sf = None
            current_tmp = None
            if tmp is None or not tmp.is_file():
                current_frames = 0
                return
            seconds = current_frames / float(sample_rate)
            if seconds < min_s:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                current_frames = 0
                return
            # Trim trailing silence (we keep writing while SMTC still says
            # playing, so a paused player may leave a quiet tail).
            try:
                data, rate = sf.read(str(tmp), dtype="float32", always_2d=True)
            except Exception as exc:  # noqa: BLE001
                data, rate = None, sample_rate
                state["error"] = f"读取录音失败：{exc}"
            final_path = out_dir / f"song_{time.strftime('%Y%m%d_%H%M%S')}_{song_seq}.wav"
            duration = 0.0
            if data is not None and len(data):
                env = np.max(np.abs(data), axis=1)
                threshold = max(0.0015, float(np.percentile(env, 95)) * 0.02)
                keep = np.where(env > threshold)[0]
                if keep.size:
                    end = int(keep[-1]) + int(rate * 0.4)
                    data = data[:min(len(data), end)]
                data = _apply_gain(data.mean(axis=1, keepdims=True).repeat(2, axis=1)
                                   if data.shape[1] == 1 else data)
                sf.write(str(final_path), data, rate, subtype="PCM_24")
                duration = round(len(data) / float(rate), 2)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            event = {"ok": duration > 0, "path": str(final_path),
                     "title": current_meta.get("title") or "",
                     "artist": current_meta.get("artist") or "",
                     "duration_s": duration, "ended_at": _now(), "reason": reason,
                     "sample_rate": sample_rate}
            if not event["ok"]:
                event["error"] = state.get("error") or "录音为空"
            with events_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            state["songs"] = int(state.get("songs") or 0) + 1
            current_frames = 0
            write_status(force=True)

        capture.start()
        tag, payload = capture.packet()
        if tag == b"E":
            raise RuntimeError("采集组件报错：" + payload.decode("utf-8", "replace")[:300])
        if tag != b"R":
            raise RuntimeError("采集组件没有发送 READY")
        state.update({"state": "ready", "device": "应用声道（逐进程）"})
        write_status(force=True)

        while True:
            if stop_path is not None and stop_path.exists():
                stopped = True
                break
            if not capture.alive():
                raise RuntimeError("采集进程已退出（目标应用可能关闭）")
            tag, payload = capture.packet()
            if tag == b"E":
                raise RuntimeError("采集组件报错：" + payload.decode("utf-8", "replace")[:300])
            if tag != b"H" or not payload or len(payload) % 8:
                continue
            frame = np.frombuffer(payload, dtype="<f4").reshape(-1, 2).copy()
            frame_index += 1
            rolling.append(frame)
            if len(rolling) > preroll_frames:
                rolling.pop(0)
            if current_sf is not None:
                current_sf.write(frame)
                current_frames += len(frame)
            level = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))

            if frame_index % smtc_every == 0:
                smtc = _smtc(job)
                music = _music_session(job, smtc)
                if (music is None and bool(job.get("assume_music"))
                        and not (smtc.get("sessions") or [])):
                    # SMTC 整体不可用（但目标应用已被判定为音乐播放器）：
                    # 退化为「按静音/最长时长切歌」，标题留空由面板补。
                    music = {"playing": True, "playback_type": "music",
                             "title": "", "artist": "", "app": ""}
                if music:
                    title, artist = _split_title(str(music.get("title") or ""))
                    artist = artist or str(music.get("artist") or "").strip()[:120]
                    meta = {"title": title, "artist": artist}
                    if current_sf is None:
                        open_song(meta, with_preroll=True)
                    elif meta["title"] and meta["title"] != current_meta["title"]:
                        finish_song("title_change")
                        open_song(meta, with_preroll=False)
                    elif meta["artist"] and meta["artist"] != current_meta["artist"]:
                        current_meta.update(meta)
                    silence_count = 0
                else:
                    if current_sf is not None:
                        silence_count += 1
                        if silence_count >= silence_end:
                            finish_song("playback_stopped")
                            silence_count = 0

            if current_sf is not None:
                if level < 0.0015:
                    silence_count += 1
                    if silence_count >= max(silence_end * 2, 8):
                        finish_song("silence")
                        silence_count = 0
                else:
                    silence_count = 0
                if current_frames / float(sample_rate) >= max_s:
                    finish_song("max_length")
            write_status()

        finish_song("stopped")
        state.update({"state": "stopped", "title": "", "artist": "", "seconds": 0.0})
        write_status(force=True)
        result = {"ok": True, "songs": state.get("songs", 0), "stopped": stopped}
    except Exception as exc:  # noqa: BLE001
        try:
            finish_song("capture_lost")  # type: ignore[name-defined]
        except Exception:  # noqa: BLE001
            pass
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["state"] = "failed"
        try:
            write_status(force=True)  # type: ignore[name-defined]
        except Exception:  # noqa: BLE001
            pass
        result = {"ok": False, "error": state["error"], "trace": traceback.format_exc()[-2000:]}
    finally:
        try:
            capture.stop()
        except Exception:  # noqa: BLE001
            pass
        out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
