# 为她而唱（music_synth）

全自动唱歌模块：**采集 AI 正在听的歌 → 按开头/结尾切歌 → 识歌与歌词校准 → 人声/伴奏分离 →
调教出 MIDI+歌词 → DiffSinger 音色库合成 → 混回伴奏 → 面板出曲、AI 调用演唱**。

所有计算都在本机完成，音频与模型不上传。

## 一次触发

面板「录下正在听的歌并出曲」，或 AI 调用 `ms_run_auto`。入口会**立即返回 `job_id`**（宿主对单次
入口调用有 30s 上限），采集与出曲在插件后台线程继续，面板每 3 秒轮询 `ms_state` 看进度。

1. `music_perception:record_playback` 用 Windows 扬声器回环录 48kHz 立体声 WAV，同时记录
   Windows 媒体会话标题时间线（不占麦克风、不混入 AI 自己的声音）。
2. `workers/split_worker.py` 以「静音能量谷 + 标题变化」切歌；短于 `min_song_s`（默认 25s）的
   片段自动合并，宁可少切也不把一首歌切碎。
3. `music_perception` 的媒体标题与官方/识别歌词用于歌名、歌词校准。
4. `workers/separate_worker.py` 走 audio-separator（默认 BS-RoFormer / Mel-Band RoFormer Karaoke）。
5. `workers/tune_worker.py` 走 SoulX-Singer 预处理链：RMVPE 基频 → 人声检测 → Paraformer 中文
   歌词 → ROSVOT 音符 + RWBD 词边界 → `metadata.json`。
6. `workers/diffsinger_worker.py` 把音符/歌词转成 OpenUtau DiffSinger 音素序列（中文 pinyin 走
   `dsdict-zh` + g2pM，英文走 `dsdict-en` + g2p_en），依次跑时长模型（dsdur）→ 方差模型
   （气息/发声）→ 声学模型（ONNX，CPU 推理）→ NSF-HiFiGAN 声码器（GPU），输出 44.1kHz 人声。
7. `workers/mix_worker.py` 把人声按 RMS 对齐后混回原伴奏，输出 `final.wav`。

每一步都是独立子进程，产物落盘；某一步失败时保留此前所有产物并在曲目里写清失败步骤。

## 音色

- 合成音色完全由所选 **DiffSinger 音色库**决定（默认绮萱 Qixuan v2.7.0，中/英/日音素表，
  Apache 兼容条款）。面板「合成音色」下拉框列出 `models\diffsinger\*` 下所有音色库，
  选择后写入 `diffsinger_bank`，下一首生效。
- 面板上传样本（`ms_set_timbre`）或填本地路径，存档在插件 `data/timbre/`。**当前版本没有自动
  音色训练**，样本只留档、不参与合成；曲目 `timbre_source` 如实写成 `voicebank:<音色库名>`。
- 音域默认取音色库 `dsconfig.yaml` 的 `pitch_range`，没有则 48–84（MIDI）；可用 `vocal_range`
  覆盖。低于音域下限的音符自动升 12 个半音（提高八度），与旧合成器行为一致。

## 首次使用（空环境）

第一次打开面板会看到「首次使用」卡片，也可以直接调用入口：

1. `ms_setup_check` 逐项体检：Python / FFmpeg / NVIDIA 驱动 / torch / onnxruntime /
   audio-separator / SoulX 预处理仓库 / DiffSinger 仓库 / DiffSinger 音色库 / 声码器 /
   预处理权重 / 分离权重，每项给出
   `ready` / `missing` / `mismatch`（必装）或 `optional`，以及缺失项的总下载大小。
2. 在面板里选安装目录（默认建议剩余空间最大的非系统盘，可改任意盘），
   点「一键安装缺失项」→ `ms_setup_install`，立即返回 `job_id`，进度用 `ms_setup_status` 轮询。
3. 安装是幂等的、可断点重跑：已存在的 venv/权重会跳过；pip 复用缓存；模型下载走
   HTTP Range，中断后从 `.part` 续传；HuggingFace 权重用 `snapshot_download(resume_download=True)`。
4. 模型与依赖分开管理：`models_dir` / `sep_model_dir` 独立可配，换目录后只补缺失文件，
   不会把已经下好的几个 GB 重新下一遍。

没有系统 Python 时，安装器会自动下载可携带的 python-build-standalone（约 25MB）再建 venv，
不需要用户预装任何东西。国内网络可在向导里把下载源切到 `mirror`（hf-mirror.com）。

## 路径与目录

配置在插件 `data/config.json`（宿主 `[music_synth]` 段也可预置）：

| 键 | 留空时的默认 |
| --- | --- |
| `compute_root` | 自动选剩余空间最大的非系统盘（≥8GB），旧安装 `E:\NekoMusicSynth` 会被识别复用 |
| `envs_dir` / `models_dir` / `work_dir` | 从 `compute_root` 派生 |
| `sep_model_dir` | `work_dir\models` |
| `diffsinger_bank` | 自动选 `models\diffsinger` 下第一个带 `dsvariance` 的音色库 |
| `vocal_range` | 音色库 `pitch_range`，再缺省 `48,84` |
| `base_python` | 自动找 `py -3.10/3.11/3.12` 或 PATH/AppData 里的 Python |

`ms_set_config` 可以单独改任意一项，改动立即生效（下一次运行重新解析路径），不需要重启插件。

## 运行期自适应（每次运行都重新探测）

`ms_process_track` / `ms_run_auto` 的**每一次**运行开始时都会重新读取显卡型号、显存总量与
**当前空闲显存**、内存、CUDA 计算能力，并把原始探测结果写进 `pipeline.log` 的 `PROBE`/`PLAN`
行与曲目的 `plan` 字段（面板可见）。`ms_probe` 可随时手动重测。

参数按**空闲显存**而不是型号决定（同一张卡，后台占着显存时会自动降档）：

| 空闲显存 | 精度 | 合成步数 | 并发 | 分离 step | 单段时长 |
| --- | --- | --- | --- | --- | --- |
| ≥16GB | fp16 | 32 | 2 | 2 | 30s |
| ≥12GB | fp16 | 28 | 2 | 2 | 30s |
| ≥9GB | fp16 | 24 | 1 | 4 | 20s |
| ≥7GB | fp16 | 20 | 1 | 4 | 20s |
| 5.8–7GB | fp16 | 16 | 1 | 8 | 15s |
| 2–5.8GB | fp16 | 16 | 1 | 8 | 15s，**合成分离走 CPU** |
| <2GB / 无 N 卡 / sm<6 | fp32 | 12 | 1 | 8 | 15s，全流程 CPU，并给出明确警告 |

- 40 系（sm_89）走 fp16；没有 50 系专属路径（NVFP4/FP8），不会拿没验证的量化冒充。
- 显存不足时按「降并发 → 降步数/缩短分段 → 分离或合成改 CPU」逐级降级；真正的 OOM 还会
  自动重试（降参数 → CPU），并在曲目里写明建议（关掉占显存的程序 / 接受 CPU 速度）。
- 用户仍可在面板把档位改成 low/mid/high，但执行时仍以当前空闲显存为准做安全钳制。

## AI 调用

`@llm_tool(name="music_synth_sing")` → `ms_sing`：按歌名或 `track_id` 找到**已完成合成**的成品，
在本机播放并返回路径；没有成品时如实返回 `ok=false` 与待处理曲目，不会假装唱过。

播放走 ffplay（有进程可停）→ winsound（Windows 自带，仅 WAV）→ 系统默认程序三级回退。
`os.startfile` 单独不可靠：这台机器把 .wav 关联到商店版「媒体播放器」，从插件进程打开会静默退出
（2026-10-31 实测），所以不作为首选。`ms_stop_playback` 可停止当前播放。

## 关于“6 轨”

Karaoke 版 Mel-Band/BS-RoFormer 的产出是**人声 + 伴奏 2 轨**，不是 6 轨。
本模块按真实产出标注 `stems: 2`；需要鼓/贝斯/吉他等分轨时请把 `sep_backend` 切到
`demucs` 并用 `htdemucs_6s`，但那就不是本档位默认的 Karaoke 模型了。

## 版权

- 默认 BS-RoFormer / Mel-Band RoFormer Karaoke 权重为 **CC BY-NC-SA**，仅限非商业使用。
- 需要商用或 MIT 许可时把 `sep_backend` 切到 `demucs`（HTDemucs，MIT 权重）。这一步需要先在
  `envs/sep` 里 `pip install demucs`——本机没有装，`demucs` 分支尚未实跑验证，只验证了默认
  RoFormer 路径。

## 已知边界

- 采集与合成两套环境：`envs/soulx`（torch 2.10.0+cu128，CUDA 可用）与 `envs/sep`
  （audio-separator 0.48 + onnxruntime + CUDA torch）。
- torchaudio 2.10 已移除内置解码器，`torchaudio.load` 转调 torchcodec（需要 FFmpeg 共享库）。
  `diffsinger_worker.py` 全部改用 soundfile 读写，不依赖 torchaudio 解码。
- DiffSinger 声学/方差/时长模型走 **onnxruntime CPU**（实测 30s 音频 24 steps 约 4s），
  声码器走 GPU（torch）；不需要 onnxruntime-gpu，也不会因为显存不足而卡住整条链。
- 50 系显卡目前使用 fp16（CUDA 12.8 版 torch），未做 FP4 量化，避免拿“已开启 NVFP4”冒充。
- 音素覆盖：中文（pinyin）与英文（`dsdict-en` 词表）实跑验证；日语需要音色库自带
  `dsdict-ja.yaml` 与分词器，绮萱 v2.7.0 未附带，所以日文歌词会退化为 `SP` 并在
  `unknown_lyrics` 里如实列出。英文的 g2p_en 需要 NLTK 数据包，缺失时回退到 `dsdict-en`
  词表查找，查不到按字母拆音素。
- Vocal2Midi 的模型目录（GAME / tifa / Qwen3-ASR / RMVPE）需要用户自行下载；缺失时
  `tune_backend=vocal2midi` 会明确报出缺哪个目录，默认仍走 SoulX 预处理链。
- 歌名/歌词校准用的是 `music_perception` 的媒体会话标题与官方歌词入口；仓库里没有可跨插件
  调用的窗口截图 OCR 入口，所以截图这一路没有接入，切歌主要靠静音能量谷 + 标题变化。
