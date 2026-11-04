// 为她而唱 · 训练向导的模拟数据（2026-11-03）
//
// 用途：后端 §5 接口还没全部就绪时，训练页仍能走完整个流程做界面验收。
// 规则：只有「调用失败」或「没拿到约定字段」才切换到模拟；界面必须写明「模拟数据」。
// 这里的数值只用于演示，不能当成实测参数对外引用。
import type { Slot } from "./trainModel"

export const MOCK_NOTE = "模拟数据（后端 §5 接口未就绪），仅用于看界面，不代表真实训练"

export function mockHw() {
  return {
    ok: true,
    mock: true,
    gpu: { name: "NVIDIA GeForce RTX 3060（模拟）", vram_mb: 12288 },
    cpu_threads: 16,
    ram_gb: 32,
    tier: "mid",
    train: { batch_size: 8, num_workers: 4, amp: "fp16" },
    sep: { batch_size: 4, use_autocast: true },
    note: MOCK_NOTE,
  }
}

function clip(id: string, slot: string, dur: number, status = "ok", reason = "") {
  return { id: id, slot: slot, src: "mock", path: "mock/" + id + ".wav", dur_s: dur,
           peak: 0.82, status: status, reason: reason }
}

export function mockCorpus(extra: { slot: string; seconds: number; job: string }) {
  const song = 96 + (extra && extra.slot === "song" ? extra.seconds : 0)
  const dry = 48 + (extra && extra.slot === "dry_sing" ? extra.seconds : 0)
  const speech = 30 + (extra && extra.slot === "speech" ? extra.seconds : 0)
  const clips = [
    clip("mock-song-1", "song", 12.0),
    clip("mock-song-2", "song", 9.5),
    clip("mock-song-3", "song", 11.2, "rejected", "几乎无声"),
    clip("mock-dry-1", "dry_sing", 14.0),
    clip("mock-dry-2", "dry_sing", 13.4),
    clip("mock-dry-3", "dry_sing", 12.8),
    clip("mock-speech-1", "speech", 8.0),
  ]
  return {
    ok: true,
    mock: true,
    slots: {
      song: { files: ["mock/整首歌.mp3"], clips: clips.filter((c) => c.slot === "song"), valid_s: song },
      dry_sing: { files: ["mock/清唱.wav", "mock/清唱2.wav"],
                  clips: clips.filter((c) => c.slot === "dry_sing"), valid_s: dry },
      speech: { files: ["mock/朗读.wav"], clips: clips.filter((c) => c.slot === "speech"), valid_s: speech },
    },
    total_valid_s: song + dry + speech,
    clips: clips,
  }
}

/** 模拟一次投递：每调用一次 progress 秒数往上走。 */
export function mockJob(jobId: string, pct: number): any {
  const phase = pct >= 100 ? "done" : pct > 60 ? "checking" : pct > 25 ? "dereverb" : "separating"
  return {
    ok: true,
    mock: true,
    job_id: jobId,
    phase: phase,
    pct: pct,
    stage_text: phase === "done" ? "完成" : "处理中 " + pct + "%",
    error: "",
  }
}

export function mockRuns() {
  return {
    ok: true,
    mock: true,
    active: "mock-run-001",
    total: 1,
    runs: [{ run_id: "mock-run-001", ok: true, name: "小夏（模拟）", phase: "awaiting_choice",
             step: 8000, updated_at: "2026-11-03 10:00:00" }],
  }
}

/** 模拟训练状态：step 由界面传进来，越往后阶段越高。 */
export function mockStatus(step: number, total: number) {
  const milestones = []
  for (let v = 2000; v <= total; v += 2000) milestones.push(v)
  const previews = milestones.filter((m) => m <= step).map((m) => ({
    step: m,
    wav: "mock/wav_" + m + ".wav",
    val_loss: Math.round((1.42 - m / total * 0.6) * 10000) / 10000,
    flag: m === 4000 ? "clip" : "ok",
    seconds: 8,
  }))
  const next = milestones.find((m) => m > step) || null
  const done = Math.max(0, ...milestones.filter((m) => m <= step))
  const pct = next && next > done ? Math.round((step - done) * 100 / (next - done)) : 100
  const phase = step >= total ? "awaiting_choice" : "training"
  return {
    ok: true,
    mock: true,
    run_id: "mock-run-001",
    name: "小夏（模拟）",
    phase: phase,
    step: step,
    steps_total: total,
    next_milestone: next,
    milestones: milestones,
    previews: previews,
    alive: true,
    auto: false,
    best_step: done || null,
    best_val_loss: previews.length ? previews[previews.length - 1].val_loss : null,
    speed: { steps_per_sec: 1.4 },
    progress: {
      step: step, target: next, pct: pct, eta_s: next ? Math.round((next - step) / 1.4) : null,
      batch_per_s: 0.46, stage_text: phase === "training" ? "" : "已到节点，等你试听决定",
    },
    dataset: { clips_train: 128, clips_val: 14, total_sec: 744, dropped: [{}, {}] },
    params: {
      batch_size: 8, duration: 5.0, amp: "fp16", use_pitch_aug: true, weight_decay: 0.0001,
      val_ratio: 0.1, auto_cap: total, init_from: "auto",
      reasons: { batch_size: "按 12GB 显存", amp: "3090/3060 走 fp16" },
    },
    advice: phase === "training" ? {} : { code: "can_stop", text: "val loss 已连续 3 个节点没有再降，建议停下挑一个节点保留。" },
    // 手动模式 + 过拟合迹象：界面应弹窗询问是否继续
    overfit_prompt: step >= total ? { step: step, val_loss: 1.0312, best_step: done - 2000, best_val: 0.9812 } : undefined,
  }
}

/** 模拟试听 WAV（0.6 秒 440Hz，免得面板里点试听什么都听不到）。 */
export function mockPreview() {
  const rate = 8000
  const n = Math.round(rate * 0.6)
  const bytes = new Uint8Array(44 + n * 2)
  const view = new DataView(bytes.buffer)
  const ascii = (off: number, text: string) => {
    for (let i = 0; i < text.length; i++) bytes[off + i] = text.charCodeAt(i)
  }
  ascii(0, "RIFF"); view.setUint32(4, 36 + n * 2, true); ascii(8, "WAVE")
  ascii(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true)
  view.setUint16(22, 1, true); view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true)
  view.setUint16(32, 2, true); view.setUint16(34, 16, true)
  ascii(36, "data"); view.setUint32(40, n * 2, true)
  for (let i = 0; i < n; i++) {
    const v = Math.sin(2 * Math.PI * 440 * i / rate) * 8000 * Math.min(1, i / 400, (n - i) / 400)
    view.setInt16(44 + i * 2, v, true)
  }
  let bin = ""
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i])
  return { ok: true, mock: true, mime: "audio/wav", audio_base64: btoa(bin) }
}
