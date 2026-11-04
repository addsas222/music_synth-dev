"""Read Windows system media metadata in the plugin's isolated CPython.

No audio is captured here. Bytes-only JSON avoids Windows GBK stdout failures.
"""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
import sys


def _to_seconds(value):
    if value is None:
        return None
    if hasattr(value,'total_seconds'):
        return max(0., float(value.total_seconds()))
    if hasattr(value,'duration'):
        # winrt TimeSpan duration is stored in 100ns units.
        return max(0., float(value.duration)/10_000_000.)
    return None

async def _session_info(session):
    props = await session.try_get_media_properties_async()
    playback = session.get_playback_info()
    state = getattr(playback, 'playback_status', None)
    # WinRT Python bindings sometimes expose enum numeric values instead of names.
    state_text = str(getattr(state, 'name', state)).strip().lower()
    playing = state_text in {'4', 'playing'} or state_text.endswith('.playing')
    # PlaybackType is supplied by PlaybackInfo. Some bindings expose an
    # extra property on metadata too; use the authoritative live playback
    # value first so old metadata cannot misclassify videos as songs.
    media_type = getattr(playback, 'playback_type', None) if playback is not None else None
    if media_type is None and props is not None:
        media_type = getattr(props, 'playback_type', None)
    # C++/WinRT enum variants, or Python WinRT enum values.
    media_name = str(getattr(media_type, 'name', media_type or '')).lower()
    if media_name not in {'music','video'}:
        # MediaPlaybackType: Unknown=0, Music=1, Video=2, Image=3.
        media_name = {'1': 'music', '2': 'video'}.get(media_name, 'unknown')
    timeline = session.get_timeline_properties()
    # A source may expose multiple independently playing sessions in one app
    # (e.g. several browser tabs). Do not collapse by executable name.
    return {'available': props is not None,
            'playing': playing,
            'source': 'windows_smtc',
            'app': str(getattr(session, 'source_app_user_model_id', '') or '')[:160],
            'player': str(getattr(session, 'source_app_user_model_id', '') or '')[:160],
            'title': str(getattr(props, 'title', '') or '')[:180],
            'artist': str(getattr(props, 'artist', '') or '')[:180],
            'album': str(getattr(props, 'album_title', '') or '')[:180],
            'playback_type': media_name,
            'position_s': _to_seconds(getattr(timeline, 'position', None)) if timeline else None,
            'duration_s': _to_seconds(getattr(timeline, 'end_time', None)) if timeline else None}


async def get_media_metadata():
    from winrt.windows.media.control import GlobalSystemMediaTransportControlsSessionManager
    manager = await GlobalSystemMediaTransportControlsSessionManager.request_async()
    if manager is None:
        return {'available': False, 'reason': 'no_system_media_session', 'sessions': []}
    current = manager.get_current_session()
    # get_sessions() is optional in mocked/older runtimes.
    try:
        all_sessions = list(manager.get_sessions())[:12]
    except (AttributeError, OSError, RuntimeError):
        all_sessions = [current] if current is not None else []
    playing_sessions = []
    current_info = None
    async def inspect(session):
        if session is None:
            return None
        try:
            return await asyncio.wait_for(_session_info(session), timeout=.85)
        except (RuntimeError,OSError,ValueError,AttributeError,asyncio.TimeoutError):
            return None
    # Inspect concurrently: 12 sequential WinRT queries could exceed the
    # 2.5s worker timeout, causing a false "single source" fallback.
    results = await asyncio.gather(*(inspect(item) for item in all_sessions))
    errors = sum(item is not None and data is None for item,data in zip(all_sessions,results))
    for item,data in zip(all_sessions,results):
        if data is None:
            continue
        if item is current:
            current_info = data
        if data.get('playing'):
            playing_sessions.append(data)
    if current_info is None and current is not None:
        try:
            current_info = await asyncio.wait_for(_session_info(current), timeout=.85)
            if current_info['playing'] and current_info not in playing_sessions:
                playing_sessions.append(current_info)
        except (RuntimeError, OSError, ValueError, AttributeError, asyncio.TimeoutError):
            current_info = None
    result = (current_info if current_info and current_info.get('playing')
              else next((x for x in playing_sessions if x.get('available')), None)
              or current_info or {'available': False, 'reason': 'no_system_media_session'})
    # Playback sessions describe player intent, not proof of active sound.
    return {**result, 'sessions': playing_sessions,
            'source_evidence_note': 'SMTC_PLAYBACK_NOT_PER_PROCESS_PCM',
            'incomplete_source_probe':bool(errors)}


def main():
    if len(sys.argv)!=2:
        return 2
    vendor=Path(sys.argv[1]).resolve()
    if not vendor.is_dir():
        return 2
    sys.path.insert(0,str(vendor))
    try:
        output=asyncio.run(asyncio.wait_for(get_media_metadata(),timeout=2.5))
    except (ImportError,RuntimeError,OSError,ValueError,asyncio.TimeoutError) as exc:
        output={'available':False,'reason':type(exc).__name__}
    sys.stdout.buffer.write(json.dumps(output,ensure_ascii=False).encode('utf-8')+b'\n')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
