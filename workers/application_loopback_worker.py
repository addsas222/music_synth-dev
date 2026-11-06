"""Per-process Windows WASAPI capture, from an isolated managed CPython.

Uses the pinned proc-tap 1.1.1 Windows wheel (native ActivateAudioInterfaceAsync
process-tree capture). Unlike the old SoundCard listener this never captures a
mixed default speaker. Does not enumerate microphone or record a WAV file.

Protocol matches isolated_loopback.py: READY R; optional 48kHz stereo H;
2-second float32 16kHz mono D; bounded UTF-8 error E. Audio stays in RAM.
"""
from __future__ import annotations

import json
import struct
import sys
import time
import traceback
from pathlib import Path

_HEADER = struct.Struct('!cI')
_OUT = sys.stdout.buffer
WINDOW_FRAMES = 96_000


def _send(tag, data):
    if len(data) > 1_000_000:
        raise RuntimeError('PCM packet size exceeded')
    _OUT.write(_HEADER.pack(tag, len(data)))
    _OUT.write(data)
    _OUT.flush()


def _message(tag, info):
    _send(tag, json.dumps(info, ensure_ascii=False).encode('utf-8'))


def _stream(tap, source, token, *, stereo: bool):
    import app_channels
    import numpy as np
    from scipy.signal import resample_poly

    pending = bytearray()
    framesize = 2 * 4   # 48kHz stereo interleaved float32 (ProcTap API)
    needed = WINDOW_FRAMES * framesize
    last_emitted = time.monotonic()
    while True:
        # Check process identity before each frame; PID reuse is not allowed.
        if not app_channels.verify_selected_process(source, token):
            raise RuntimeError('selected_process_exited_or_reused_pid')
        # ProcTap's context manager calls start(); read() is only valid
        # afterwards. A crashed native session must not appear as silence.
        if not tap.is_running:
            raise RuntimeError('selected_process_native_capture_stopped')
        chunk = tap.read(timeout=0.3)
        if chunk:
            if not isinstance(chunk, (bytes, bytearray, memoryview)):
                raise RuntimeError('process_tap_returned_invalid_pcm')
            if len(chunk) % framesize or len(chunk) > 2_000_000:
                raise RuntimeError('process_tap_returned_invalid_frame_size')
            pending.extend(chunk)
        # Always return two-second frames to existing audio collector. In a
        # silent selected application, emit silence but never replace it by
        # whole-system sound. Bound packet and buffer sizes strictly.
        if len(pending) < needed:
            if time.monotonic() - last_emitted < 2.2:
                continue
            pending.extend(b'\x00' * (needed - len(pending)))
        frame = bytes(pending[:needed])
        del pending[:needed]
        if len(pending) > 2 * needed:
            # Prioritize real-time over building up hours of audio buffering.
            del pending[:len(pending)-needed]
        samples = np.frombuffer(frame, dtype='<f4').reshape(WINDOW_FRAMES, 2)
        if not np.all(np.isfinite(samples)):
            raise RuntimeError('non_finite_process_audio')
        if stereo:
            _send(b'H', frame)
        mono = resample_poly(samples.mean(axis=1), 1, 3).astype('<f4', copy=False)
        if mono.shape != (32000,):
            raise RuntimeError('bad_downsample_size')
        _send(b'D', mono.tobytes())
        last_emitted = time.monotonic()


def main():
    stage = 'initializing_process_capture'
    try:
        try:
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError, OSError):
            pass
        if sys.platform != 'win32':
            raise RuntimeError('Windows_10_20348_or_later_required')
        if len(sys.argv) not in (5, 6):
            raise RuntimeError('missing_isolated_capture_parameters')
        main_vendor, addon, raw_pid, token = sys.argv[1:5]
        stereo = len(sys.argv) == 6 and sys.argv[5] == 'stereo'
        if not Path(main_vendor).is_dir() or not Path(addon).is_dir():
            raise RuntimeError('isolated_capture_environment_missing')
        sys.path[:0] = [str(Path(addon).resolve()),str(Path(main_vendor).resolve()),
                        str(Path(__file__).resolve().parent)]
        import app_channels
        pid = int(raw_pid)
        if not app_channels.verify_selected_process(pid, token):
            raise RuntimeError('application_identity_changed_before_capture')
        build = int(sys.getwindowsversion().build)
        if build < 20348:
            raise RuntimeError('Windows_build_does_not_support_process_loopback')
        stage = 'importing_native_process_tap'
        from proctap import ProcessAudioCapture
        stage = 'opening_selected_process_audio'
        tap = ProcessAudioCapture(pid=pid)
        with tap:
            # ProcTap.__enter__ starts the WASAPI process loopback.
            # Report READY only after the real native capture starts. This is
            # not just successful process enumeration or name classification.
            _message(b'R', {'device':f'单独应用 PID {pid}',
                            'sample_rate':16000,'capture_rate':48000,
                            'mode':'wasapi_process_tree','pid':pid})
            stage = 'streaming_selected_process_pcm'
            _stream(tap, pid, token, stereo=stereo)
    except BaseException as exc:
        print(f'[{stage}] {type(exc).__name__}: {exc}', file=sys.stderr)
        traceback.print_exc(limit=7, file=sys.stderr)
        try:
            _message(b'E',{'stage':stage,'error':str(exc)[:340]})
        except (OSError, BrokenPipeError):
            pass
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(main())
