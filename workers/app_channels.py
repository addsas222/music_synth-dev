"""Conservative Windows app-process catalogue for selective *real* PCM capture.

SMTC titles are NOT audio routing.  Selection is by verified pid+creation FILETIME;
Windows process-tree WASAPI loopback captures only the chosen process family.
No native calls at import. No host SDK or external package dependencies.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Optional

CAPTURE_MODES = frozenset({'monitor', 'music', 'video', 'auto', 'ai'})


@dataclass(frozen=True)
class AppIdentity:
    pid: int
    created: str
    executable: str
    parent_pid: int = 0
    host_ai: bool = False
    path: str = ''

    def as_dict(self):
        return {'pid': self.pid, 'created': self.created, 'name': self.executable,
                'parent_pid':self.parent_pid, 'host_ai': self.host_ai,
                'path':self.path}


def _winapi():
    if sys.platform != 'win32':
        raise OSError('逐进程应用音频回环仅支持 Windows 10 Build 20348+ / Windows 11')
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    return ctypes, wintypes, kernel


def process_start_token(pid: int) -> str:
    """Native kernel process creation time; prevents accidental PID reuse.

    The capture helper validates the token again after its own launch. An
    inaccessible PID is not selectable (never use an unverified fallback).
    """
    ct, w, k = _winapi()
    if not isinstance(pid, int) or pid < 5 or pid > 0xffffffff:
        return ''
    k.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    k.OpenProcess.restype = w.HANDLE
    k.GetProcessTimes.argtypes = [w.HANDLE, ct.POINTER(w.FILETIME), ct.POINTER(w.FILETIME),
                                  ct.POINTER(w.FILETIME), ct.POINTER(w.FILETIME)]
    k.GetProcessTimes.restype = w.BOOL
    k.CloseHandle.argtypes = [w.HANDLE]
    k.CloseHandle.restype = w.BOOL
    handle = k.OpenProcess(0x1000, False, pid)
    if not handle:
        return ''
    try:
        started, left, kernel_time, user_time = w.FILETIME(), w.FILETIME(), w.FILETIME(), w.FILETIME()
        if not k.GetProcessTimes(handle, ct.byref(started), ct.byref(left),
                                  ct.byref(kernel_time), ct.byref(user_time)):
            return ''
        return str((int(started.dwHighDateTime) << 32) | int(started.dwLowDateTime))
    finally:
        k.CloseHandle(handle)


# Exact program executable identifiers, not a substring match that could exclude
# another company's app. If these do not match the installed N.E.K.O build,
# the UI must ask the user to pin the true host before promising exclusion.
NEKO_HOST_EXECUTABLES = frozenset({'n.e.k.o.exe'})
MUSIC_APPLICATIONS = frozenset({'spotify.exe','cloudmusic.exe','qqmusic.exe',
    'kgmusic.exe','kwmusic.exe','foobar2000.exe','musicbee.exe','aimp.exe',
    'winamp.exe','applemusic.exe'})
VIDEO_APPLICATIONS = frozenset({'potplayer.exe','potplayermini64.exe',
    'mpc-hc64.exe','mpc-be64.exe'})
# Browsers are deliberately NOT in either list: a browser plays music, video and
# silent pages with the same executable, so its window/media title is the only
# honest evidence and that evidence is always a hint, never a confirmation.
BROWSER_APPLICATIONS = frozenset({'chrome.exe','msedge.exe','firefox.exe','brave.exe',
    'opera.exe','vivaldi.exe','360se.exe','360chrome.exe','qqbrowser.exe',
    'sogouexplorer.exe','arc.exe'})
_BROWSER_SESSION_TOKENS = (('chrome','chrome.exe'),('msedge','msedge.exe'),
    ('microsoftedge','msedge.exe'),('firefox','firefox.exe'),('brave','brave.exe'),
    ('opera','opera.exe'),('vivaldi','vivaldi.exe'),('360se','360se.exe'),
    ('sogou','sogouexplorer.exe'),('qqbrowser','qqbrowser.exe'),('arc','arc.exe'))


def browser_executable(app_id):
    """Map an SMTC app identifier to a browser executable, else ''.

    Windows media sessions identify a browser by application id rather than by
    pid, so only the executable family can be matched. Ambiguous ids stay ''.
    """
    low = str(app_id or '').strip().casefold()
    if not low:
        return ''
    # Match whole id components only: a substring test would map unrelated
    # application ids (for example "...Search...") onto Arc or Chrome.
    parts = [part for part in re.split(r'[^a-z0-9]+', low) if part]
    for token, exe in _BROWSER_SESSION_TOKENS:
        if token in parts or (len(token) >= 5 and
                              any(part.startswith(token) for part in parts)):
            return exe
    return ''


def identify_host_tree(processes, *, known_hosts=NEKO_HOST_EXECUTABLES,
                       host_exe_path: str = ''):
    """Fixed AI app and its descendants, using process parent relationships.

    No fuzzy matching (for example, all python.exe or all browsers).
    """
    by_pid={int(p['pid']):p for p in processes if p.get('pid')}
    pinned=normalized_exe_path(host_exe_path)
    roots={pid for pid,p in by_pid.items()
           if str(p.get('name','')).casefold() in known_hosts or
           (pinned and normalized_exe_path(p.get('path',''))==pinned)}
    ai=set(roots)
    for _ in range(12):
        old=len(ai)
        for pid,process in by_pid.items():
            parent=int(process.get('parent_pid') or 0)
            if parent in ai and parent!=pid:
                # The parent must exist and have an older creation timestamp.
                try:
                    if int(by_pid[parent]['created']) < int(process['created']):
                        ai.add(pid)
                except (ValueError,KeyError,TypeError):
                    continue
        if len(ai) == old:
            break
    return ai,roots


VIDEO_TITLE_HINTS = ('视频', '播放器', '直播', '弹幕', '在线观看', 'bilibili', '哔哩',
                     'youtube', 'youku', 'iqiyi', '爱奇艺', '优酷', '腾讯视频', 'mgtv',
                     '芒果tv', 'douyin', '抖音', '西瓜视频', 'twitch', 'netflix',
                     'potplayer', 'vlc', 'mpv')
MUSIC_TITLE_HINTS = ('音乐', '歌词', '播放列表', '网易云', 'qq音乐', '酷狗', '酷我',
                     'spotify', 'youtube music', 'music.youtube')
_BV_PATTERN = re.compile(r'bv[0-9a-z]{10}', re.IGNORECASE)
_AV_PATTERN = re.compile(r'(?<![a-z0-9])av\d{4,}', re.IGNORECASE)


def _classify_title(title):
    """Window/media-title hints; returns 'music', 'video' or None."""
    low = str(title or '').casefold()
    if not low:
        return None
    # A music-only site wins over the generic keywords it also contains.
    if 'youtube music' in low or 'music.youtube' in low:
        return 'music'
    if _BV_PATTERN.search(low) or _AV_PATTERN.search(low):
        return 'video'
    if any(k in low for k in VIDEO_TITLE_HINTS):
        return 'video'
    if any(k in low for k in MUSIC_TITLE_HINTS):
        return 'music'
    return None


def is_video_title(title):
    """True when a window/media title looks like a video page (hint only)."""
    return _classify_title(title) == 'video'


def choose_title_evidence(titles, fallback_titles=()):
    """Pick (tag, title) from candidate titles; a video hint wins over music.

    A browser reports every open window, and the audible tab is not always the
    front one, so all candidates are considered. Returns ('', first_title) when
    no candidate is decisive, so the caller still has a title for context.
    """
    music = ('', '')
    plain = ('', '')
    for source in (titles or (), fallback_titles or ()):
        for raw in _as_text_tuple(source):
            text = str(raw or '').strip()
            if not text:
                continue
            if not plain[1]:
                plain = ('', text[:180])
            tag = _classify_title(text)
            if tag == 'video':
                return 'video', text[:180]
            if tag == 'music' and not music[1]:
                music = ('music', text[:180])
    if music[1]:
        return music
    return plain


def _as_text_tuple(value):
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return ()


def tag_application(app, *, media_role=None, window_title=None, classifier=None):
    """Lightweight *candidate* tags; no PCM is used.

    Order of evidence: fixed AI host -> user/media role -> window title ->
    process name -> auxiliary classifier (optional callable taking the
    metadata dict and returning 'music', 'video' or None). Unknown is a real
    third state, not an error. The classifier is only consulted when the
    earlier steps are inconclusive; any exception falls back to unknown.
    """
    if app.get('host_ai'):
        return {'tag':'ai','confidence':'fixed_host_process_tree',
                'confirmed':True,'label':'AI 声音（自动隔离）'}
    role=str(media_role or '').lower()
    if role in {'music','video'}:
        return {'tag':role,'confidence':'media_session_or_user_label',
                'confirmed':False,'label':'音乐（待音频验证）' if role=='music' else '视频（待音频验证）'}
    evidence_title=str(window_title if window_title is not None
                       else (app.get('title') or ''))
    title_tag=_classify_title(evidence_title)
    if title_tag:
        return {'tag':title_tag,'confidence':'window_title_hint','confirmed':False,
                'title_evidence':evidence_title.strip()[:180],
                'label':'可能是音乐' if title_tag=='music' else '可能是视频'}
    name=str(app.get('name','')).casefold()
    if name in MUSIC_APPLICATIONS:
        return {'tag':'music','confidence':'known_player_app_hint',
                'confirmed':False,'label':'可能是音乐'}
    if name in VIDEO_APPLICATIONS:
        return {'tag':'video','confidence':'known_video_app_hint',
                'confirmed':False,'label':'可能是视频'}
    if classifier is not None:
        try:
            verdict=str(classifier(dict(app)) or '').casefold()
        except Exception:  # noqa: BLE001 - auxiliary model must never break scanning
            verdict=''
        if verdict in {'music','video'}:
            return {'tag':verdict,'confidence':'auxiliary_model_hint','confirmed':False,
                    'label':'可能是音乐' if verdict=='music' else '可能是视频'}
    return {'tag':'unknown','confidence':'needs_selected_pcm_or_player_metadata',
            'confirmed':False,'label':'有声 · 待判断'}


def process_image_path(pid: int) -> str:
    """Query an absolute Windows executable path; never trust a display name.

    Some protected processes deny read access. An unavailable path cannot
    be manually pinned as the AI host; no name-based fallback for custom names.
    """
    try:
        ct,w,k=_winapi()
        k.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD]
        k.OpenProcess.restype=w.HANDLE
        k.QueryFullProcessImageNameW.argtypes=[w.HANDLE,w.DWORD,w.LPWSTR,ct.POINTER(w.DWORD)]
        k.QueryFullProcessImageNameW.restype=w.BOOL
        k.CloseHandle.argtypes=[w.HANDLE]
        k.CloseHandle.restype=w.BOOL
        h=k.OpenProcess(0x1000,False,int(pid))
        if not h:
            return ''
        try:
            name=ct.create_unicode_buffer(32768)
            size=w.DWORD(len(name))
            return name.value if k.QueryFullProcessImageNameW(h,0,name,ct.byref(size)) else ''
        finally:
            k.CloseHandle(h)
    except (OSError,ValueError,TypeError,AttributeError):
        return ''


def normalized_exe_path(value: str) -> str:
    return str(value or '').strip().replace('/','\\').casefold()


def list_app_processes(*, limit: int = 200, pid_allowlist: Optional[set[int]]=None,
                       host_exe_path: str = '', exe_name_filter: str = '') -> list[dict]:
    """Enumerate verified PIDs; optionally return *only audible* session owners.

    A native peak meter scan provides pid_allowlist. Without one, this is a
    process-inventory helper for existing SDK compatibility, NOT audio proof.
    """
    ct, w, k = _winapi()

    class PROCESSENTRY32W(ct.Structure):
        _fields_ = [('dwSize', w.DWORD), ('cntUsage', w.DWORD),
                    ('th32ProcessID', w.DWORD), ('th32DefaultHeapID', ct.c_size_t),
                    ('th32ModuleID', w.DWORD), ('cntThreads', w.DWORD),
                    ('th32ParentProcessID', w.DWORD), ('pcPriClassBase', w.LONG),
                    ('dwFlags', w.DWORD), ('szExeFile', w.WCHAR * 260)]

    k.CreateToolhelp32Snapshot.argtypes = [w.DWORD, w.DWORD]
    k.CreateToolhelp32Snapshot.restype = w.HANDLE
    k.Process32FirstW.argtypes = [w.HANDLE, ct.POINTER(PROCESSENTRY32W)]
    k.Process32NextW.argtypes = [w.HANDLE, ct.POINTER(PROCESSENTRY32W)]
    k.Process32FirstW.restype = k.Process32NextW.restype = w.BOOL
    k.CloseHandle.argtypes = [w.HANDLE]
    k.CloseHandle.restype = w.BOOL
    handle = k.CreateToolhelp32Snapshot(0x2, 0)
    if not handle or ct.cast(handle, ct.c_void_p).value == ct.c_void_p(-1).value:
        raise OSError('Windows 进程列表读取失败')
    candidates = []
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ct.sizeof(entry)
        ok = k.Process32FirstW(handle, ct.byref(entry))
        while ok:
            pid = int(entry.th32ProcessID)
            name = str(entry.szExeFile).strip()
            if pid > 4 and name and len(candidates) < 1800:
                candidates.append((pid, name, int(entry.th32ParentProcessID)))
            ok = k.Process32NextW(handle, ct.byref(entry))
    finally:
        k.CloseHandle(handle)
    # A passive recovery probe only needs the saved app and its ancestors.
    allow=None if pid_allowlist is None else set(pid_allowlist)
    exact_name=str(exe_name_filter or '').strip().casefold()
    relevant=set()
    if exact_name and allow is None:
        parents={pid:parent for pid,_,parent in candidates}
        for pid,name,_ in candidates:
            if name.casefold() != exact_name:
                continue
            cursor=pid
            for _ in range(12):
                if cursor <= 4 or cursor in relevant:
                    break
                relevant.add(cursor)
                cursor=parents.get(cursor,0)
    all_infos=[]
    for pid,name,parent in candidates:
        maybe_host = (name.casefold() in NEKO_HOST_EXECUTABLES or
                      (bool(host_exe_path) and
                       name.casefold()==host_exe_path.replace('/','\\').split('\\')[-1].casefold()))
        needed = ((allow is None and (not exact_name or pid in relevant)) or
                  (allow is not None and pid in allow) or maybe_host)
        created = process_start_token(pid) if needed else ''
        path = process_image_path(pid) if (
            (allow is None and (not exact_name or pid in relevant)) or
            (allow is not None and pid in allow) or maybe_host) else ''
        all_infos.append({'pid':pid,'name':name,'parent_pid':parent,
                          'created':created,'path':path})
    # Ancestry for audio-active children whose parent isn't itself audible.
    # Get creation tokens only for parents of included PIDs, not 1500 processes.
    if allow is not None:
        by_pid={x['pid']:x for x in all_infos}
        for pid in allow:
            node=by_pid.get(pid)
            for _ in range(12):
                if not node:
                    break
                parent=by_pid.get(node['parent_pid'])
                if not parent or parent['pid'] == pid:
                    break
                if not parent['created']:
                    parent['created'] = process_start_token(parent['pid'])
                node=parent
    ai,roots=identify_host_tree(all_infos,host_exe_path=host_exe_path)
    result=[]
    for item in sorted(all_infos,key=lambda x:(x['name'].casefold(),x['pid'])):
        pid=item['pid']
        if allow is not None and pid not in allow:
            continue
        if exact_name and item['name'].casefold() != exact_name:
            continue
        if not item['created']:
            continue
        result.append(AppIdentity(pid,item['created'],item['name'],
                                  item['parent_pid'],pid in ai,item.get('path','')).as_dict())
        if len(result) >= max(1, min(500, int(limit))):
            break
    return result


def verify_selected_process(pid, created) -> bool:
    """Fail closed on closed, recycled, inaccessible or non-Windows processes."""
    try:
        valid_pid = int(pid)
        expected = str(created)
        return (valid_pid > 4 and len(expected) >= 12 and expected.isdecimal()
                and process_start_token(valid_pid) == expected)
    except (TypeError, ValueError, OSError):
        return False


def selected_scene(decision, selected: dict, *, ai_voice_active: bool = False) -> dict:
    """Scene policy for **verified** one-app PCM, not mixed system loopback.

    Recognizes only categories supported by signal and explicit user routing.
    No automatic transcript for a source marked AI or monitor-only.
    """
    mode = str(selected.get('mode') or 'monitor').lower()
    pid = int(selected.get('pid') or 0)
    source_key = f"app:{pid}:{selected.get('created', '')}"
    music = bool(getattr(decision, 'music', False))
    speech = float(getattr(decision, 'speech_score', 0) or 0)
    music_score = float(getattr(decision, 'music_score', 0) or 0)
    kind, reason = 'unknown', 'selected_app_monitor_only'
    if mode == 'ai':
        kind, reason = 'ai_output', 'selected_application_is_ai_audio'
    elif mode == 'music':
        # A silent channel is not a contradiction either: keep the reviewed
        # music class visible while the channel is momentarily quiet, exactly
        # like the video branch does, so a brief quiet intro cannot downgrade
        # an already-reviewed source back to 'candidate'.
        decision_reason = str(getattr(decision, 'reason', '') or '')
        if decision_reason in {'input_too_quiet', 'insufficient_samples', 'invalid_pcm'}:
            kind, reason = 'unknown', 'selected_app_channel_silent'
        else:
            kind, reason = ('music', 'selected_app_music_confirmed') if music else ('unknown', 'selected_app_not_music')
    elif mode == 'video':
        # A video normally mixes dialogue with background music, so music
        # presence must not veto the class. Only *local speech evidence*
        # decides whether the video-speech channel is verified. A music video
        # (MV) has a video frame but music audio: route the audio to the music
        # recognizer (dual channel) instead of discarding it as speechless video.
        decision_reason = str(getattr(decision, 'reason', '') or '')
        if decision_reason in {'input_too_quiet', 'insufficient_samples', 'invalid_pcm'}:
            kind, reason = 'unknown', 'selected_app_channel_silent'
        elif speech >= .10:
            kind = 'video'
            reason = ('reviewed_video_speech_with_background_music'
                      if (music or music_score > .25) else
                      'user_selected_video_source_with_speech_evidence')
        elif music or music_score > .25:
            kind, reason = 'music', 'music_video_music_audio_channel'
        else:
            kind, reason = 'unknown', 'selected_video_no_speech_evidence'
    elif mode == 'auto':
        kind, reason = ('music', 'selected_app_auto_music') if music else ('unknown', 'selected_app_auto_unclassified')
    # An AI application must not be analyzed even if the OS attributes its
    # render stream to the same process; process-tree capture has this limit.
    if ai_voice_active and mode not in {'ai','monitor'} and selected.get('might_contain_ai_tts') is True:
        kind, reason = 'ai_output', 'ai_tts_may_share_selected_process'
    # A video class is confirmed by the window reviewer, but the speech
    # channel additionally needs this window's own speech evidence.
    verified = kind in {'music', 'video'} and not (kind == 'video' and speech < .10)
    return {'kind':kind, 'verified':verified, 'reason':reason,
            'speech_evidence':speech >= .10, 'music_evidence':bool(music),
            'source_count':1, 'source_key':source_key, 'source_pid':pid,
            'capture_provenance':'wasapi_process_tree',
            'classification_evidence':'user_mode_and_local_yamnet',
            'music_score':music_score, 'speech_score':speech}
