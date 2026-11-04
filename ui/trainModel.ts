// 为她而唱 · 训练声音模型（向导）的纯函数与常量（2026-11-03）
//
// 一切数值口径以总方案 §3/§4/§5 为准；本文件不放任何写数据/训练的逻辑，
// 只做「按接口返回值算界面该显示什么」，方便单独核对。
//
// 语料库接口契约（名字不得改）：
//   ms_corpus_requirements(steps) → {steps, min_s, rec_s, reasons[]}
//   ms_corpus_list()              → {slots:{song,dry_sing,speech}, total_valid_s, clips[]}
//   ms_corpus_add(slot, path|data_b64+filename) → {job_id}
//   ms_corpus_job(job_id)         → {phase, pct, stage_text, error}
//   ms_corpus_clip(clip_id, action) / ms_corpus_remove(source_id)
//   ms_train_start(steps, from_corpus=true, auto) / ms_train_status / ms_train_keep / ms_train_set…
import { eta } from "./api"

export type Slot = "song" | "dry_sing" | "speech"

export type SlotMeta = {
  key: Slot
  title: string
  need: string
  how: string
  note: string
}

export const SLOTS: Array<SlotMeta> = [
  {
    key: "song", title: "带伴奏的完整歌曲",
    need: "普通的歌：有人声、有伴奏都行。同一个人唱的最好。",
    how: "导入完整音频文件；系统会先分离出人声，再去和声、去混响、去噪。",
    note: "分离最花时间；每首歌 3～5 分钟。声线杂的歌（合唱、和声多）效果会差一些。",
  },
  {
    key: "dry_sing", title: "清唱 / 纯人声",
    need: "没有伴奏、没有人声混响的纯人声（手机清唱、剪掉伴奏的人声都可以）。",
    how: "直接导入；系统只做去混响 + 去噪（可选），不分离人声。",
    note: "质量最好的一类素材，时长按 1 : 1 计入有效时长。",
  },
  {
    key: "speech", title: "平时说话的录音",
    need: "正常语速说话、朗读；不要刻意模仿，也不要唱歌。",
    how: "直接导入；系统只做去噪。",
    note: "单独计时、默认权重低；唱歌时用不上，只在素材太少时补一点。",
  },
]

/** 界面上统一叫法：素材页和训练页用同一个名字，不再出现「整首歌曲（带伴奏）」这类两种叫法。 */
export function materialTitle(slot: any): string {
  const key = String(slot || "")
  for (const m of SLOTS) if (m.key === key) return m.title
  return key || "素材"
}

export const STEPS_CHOICES = [4000, 8000, 12000, 16000]

// 步数按素材时长自动算（和后端 _STEPS_PER_SECOND 保持一致）。
// 旧固定档位的比例就是 44 步/秒：90s→4000、180s→8000、300s→12000、360s→16000。
export const STEPS_PER_SECOND = 44
export const STEPS_MIN = 4000
export const STEPS_MAX = 30000        // 自动推荐上限
export const STEPS_HARD_MAX = 200000
export const STEPS_TARGET = 8000    // 标准档：等效 batch 24 时 ≈ barbara 的 4000×48 样本量
// 三档：速度 / 标准 / 质量。步数是「参数更新次数」，不是 micro-batch 数。
export const STEPS_MODES = [
  { key: "speed", label: "快一些", steps: 6000, hint: "约 15 分钟/千步量级，先听个大概" },
  { key: "normal", label: "标准（推荐）", steps: 8000, hint: "对齐他们 2000~4000 步（batch 48）的样本量" },
  { key: "quality", label: "细致一些", steps: 14000, hint: "更稳、更像，但耗时约翻倍，过拟合风险也更高" },
]
const REC_RATIO = 1.65

/** 有效时长（秒）→ 推荐训练步数：向下取整到 1000，限制在 4000~30000。 */
export function autoStepsFor(seconds: any): number {
  const s = Math.max(0, num(seconds))
  if (!s) return STEPS_MIN
  const raw = Math.floor((s * STEPS_PER_SECOND) / 1000) * 1000
  return Math.min(STEPS_MAX, Math.max(STEPS_MIN, raw))
}

// §4 经验值：后端 ms_corpus_requirements 拿不到时按这张表显示（界面会标注「经验值」）。
// 宿主的 export 扫描器把 `Record<number, {…}>` 里的逗号当成声明分隔符（2026-11-03），故用别名代替泛型标注。
// 同上：泛型参数里不能再带逗号，用 STEPS_REC_LIST 代替。
const STEPS_REC = null
const STEPS_REC_LIST = null
export const STEPS_TABLE: Record<STEPS_REC_LIST> = {
  4000: { min: 90, rec: 180 },
  8000: { min: 180, rec: 300 },
  12000: { min: 300, rec: 480 },
  16000: { min: 360, rec: 600 },
}

export const PHASE_TEXT = {
  queued: "排队", separating: "分离人声 / 伴奏", dereverb: "去混响", denoise: "去噪",
  slicing: "切片", checking: "质检", done: "完成", error: "出错",
  preparing: "准备中", preprocessing: "整理数据 / 提特征", training: "正在训练",
  previewing: "正在生成试听", awaiting_choice: "已到节点，等你试听决定",
  stopping: "正在停止", finished: "已结束", failed: "失败", stopped: "已停止",
}

export function phaseText(phase: any): string {
  return PHASE_TEXT[String(phase || "")] || short2(phase)
}

export function short2(value: any): string {
  return value === undefined || value === null || value === "" ? "未知" : String(value)
}

export function num(value: any): number {
  const n = Number(value)
  return Number.isFinite(n) ? n : 0
}

export type Requirement = { min: number; rec: number; reasons: string[]; source: string }

/** 后端给的值优先；没拿到就用 §4 经验值，并把来源写出来给界面标注。 */
export function requirementOf(req: any, steps: number): Requirement {
  // 表里有的用旧档位值；表外的（自动算出来的步数）按同一比例推。
  const derivedMin = Math.round(steps / STEPS_PER_SECOND)
  const fallback = STEPS_TABLE[steps] || { min: derivedMin, rec: Math.round(derivedMin * REC_RATIO) }
  const hasMin = req && req.min_s !== undefined && req.min_s !== null
  const hasRec = req && req.rec_s !== undefined && req.rec_s !== null
  return {
    min: hasMin ? num(req.min_s) : fallback.min,
    rec: hasRec ? num(req.rec_s) : fallback.rec,
    reasons: Array.isArray(req && req.reasons) ? req.reasons.map((r: any) => String(r)) : [],
    source: hasMin || hasRec ? "接口返回" : "经验值（接口未返回）",
  }
}

export type SlotView = { files: number; hidden_files: number; clips: number; valid_s: number; files_s: number; raw_files: any[]; raw_clips: any[] }

export function slotView(list: any, slot: Slot): SlotView {
  const raw = (list && list.slots && list.slots[slot]) || {}
  const rawClips = Array.isArray(raw.clips) ? raw.clips : []
  const rawFiles = Array.isArray(raw.files) ? raw.files : []
  // 隐藏的素材不计入「N 个文件」，但仍然留在 raw_files 里，方便面板「显示已隐藏」。
  const shown = rawFiles.filter((f: any) => !(f && f.hidden))
  let filesS = num(raw.files_s)
  if (!filesS) for (const f of shown) filesS += sourceSeconds(f)
  return { files: shown.length, hidden_files: rawFiles.length - shown.length,
           clips: rawClips.length, valid_s: num(raw.valid_s), files_s: Math.round(filesS * 10) / 10,
           raw_files: rawFiles, raw_clips: rawClips }
}

/** 一个素材源里第一个文件的时长（上传时 ffprobe 量好的，秒）。 */
export function sourceSeconds(f: any): number {
  const inner = f && Array.isArray(f.files) && f.files.length ? f.files[0] : f
  return inner ? num(inner.dur_s) : 0
}

/** 所有没被隐藏的素材文件的实际总时长（秒）——训练步数按这个算。 */
export function totalFileSeconds(list: any): number {
  let sum = 0
  for (const slot of ["song", "dry_sing", "speech"] as Slot[]) {
    for (const f of slotView(list, slot).raw_files) {
      if (f && !f.hidden) sum += sourceSeconds(f)
    }
  }
  return Math.round(sum * 10) / 10
}

/** 有效时长 = 质检后切片总长（不含 speech 时以 song+dry_sing 计，§4）。 */
export function effectiveSeconds(list: any, countSpeech: boolean): number {
  const song = slotView(list, "song").valid_s
  const dry = slotView(list, "dry_sing").valid_s
  const speech = slotView(list, "speech").valid_s
  return Math.round((song + dry + (countSpeech ? speech : 0)) * 10) / 10
}

/** 差多少 → 「还差 23 秒」；够了返回空串。 */
export function deficitText(have: number, need: number): string {
  const gap = Math.round((need - have) * 10) / 10
  if (gap <= 0) return ""
  return gap < 60 ? "还差 " + gap + " 秒" : "还差 " + (gap / 60).toFixed(1) + " 分钟"
}

export type ClipView = {
  id: string
  slot: string
  src: string
  dur_s: number
  peak: number | null
  status: string
  reason: string
}

export function clipList(list: any, slot?: Slot): ClipView[] {
  const all = Array.isArray(list && list.clips) ? list.clips : []
  return all
    .filter((c: any) => c && typeof c === "object")
    .filter((c: any) => (slot ? String(c.slot) === slot : true))
    .map((c: any) => ({
      id: String(c.id || ""),
      slot: String(c.slot || ""),
      src: String(c.src || ""),
      dur_s: num(c.dur_s),
      peak: c.peak === undefined || c.peak === null ? null : num(c.peak),
      status: String(c.status || "ok"),
      reason: String(c.reason || ""),
    }))
}

/** 质检剔掉的切片理由（后端 reason 原文；界面不替它编新理由）。 */
export function clipReason(clip: ClipView): string {
  if (clip.reason) return clip.reason
  return clip.status === "rejected" ? "未通过质检" : ""
}

export type MilestoneView = {
  step: number
  val_loss: number | null
  flag: string
  seconds: number | null
  ready: boolean
  /** 已越过这个节点但没留下试听 */
  missed: boolean
}

/** 每 2000 步一个节点；到点没出试听的也先占位显示「生成中」。 */
export function milestoneList(status: any): MilestoneView[] {
  const total = num(status && status.steps_total) || num(status && (status.params || {}).total_steps)
  const plan: number[] = Array.isArray(status && status.milestones) && status.milestones.length
    ? status.milestones.map((m: any) => num(m)).filter((m: number) => m > 0)
    : (total ? range(2000, total, 2000) : [])
  const made = new Map<number, any>()
  const previews = Array.isArray(status && status.previews) ? status.previews : []
  for (const p of previews) if (p && p.step !== undefined) made.set(num(p.step), p)
  const steps = plan.slice()
  for (const key of Array.from(made.keys())) if (steps.indexOf(key) < 0) steps.push(key)
  steps.sort((a, b) => a - b)
  const reached = num(status && status.step)
  return steps.map((step) => {
    const p = made.get(step) || {}
    const ready = Boolean(p && p.wav)
    return {
      step: step,
      val_loss: p.val_loss === undefined || p.val_loss === null ? null : num(p.val_loss),
      flag: String(p.flag || ""),
      seconds: p.seconds === undefined || p.seconds === null ? null : num(p.seconds),
      ready: ready,
      // 还没有试听、也没走到 → 待生成；已越过却没有 → 未生成
      missed: !ready && reached > step,
    }
  })
}

export function range(from: number, to: number, step: number): number[] {
  const out: number[] = []
  for (let v = from; v <= to; v += step) out.push(v)
  return out
}

/** 「第 X / Y 步 · 约剩 12 分钟」。Y 优先用后端 prog.target，其次下一节点。 */
export function progressText(status: any): {
  step: number; target: number | null; pct: number; etaText: string; bps: string
} {
  const prog = (status && status.progress) || {}
  const live = (status && status.live) || {}
  const step = prog.step !== undefined && prog.step !== null ? num(prog.step) : num(status && status.step)
  const target = prog.target !== undefined && prog.target !== null ? num(prog.target)
    : (status && status.next_milestone ? num(status.next_milestone) : null)
  let pct = prog.pct !== undefined && prog.pct !== null ? num(prog.pct) : -1
  if (pct < 0) {
    const total = num(status && status.steps_total)
    pct = total > 0 ? Math.max(0, Math.min(100, (step / total) * 100)) : 0
  }
  const bps = live.batch_per_s ? num(live.batch_per_s).toFixed(2) + " batch/s" : ""
  return {step: step, target: target, pct: Math.round(pct), etaText: eta(prog.eta_s), bps: bps}
}

export function flagText(flag: string): { text: string; tone: "success" | "warning" | "error" | "neutral" } {
  if (flag === "silent") return {text: "异常：几乎无声", tone: "error"}
  if (flag === "clip") return {text: "异常：削波", tone: "error"}
  if (flag === "ok") return {text: "正常", tone: "success"}
  return {text: "未标记", tone: "neutral"}
}

export function adviceTone(code: any): "warning" | "error" | "info" {
  const c = String(code || "")
  if (c.indexOf("abnormal") >= 0 && c === "stop_abnormal") return "error"
  if (c === "stop_overfit") return "error"
  if (["stop_plateau", "stop_reached_cap", "can_stop", "overfit", "plateau"].indexOf(c) >= 0) return "warning"
  return "info"
}

/** 过拟合提示：后端在手动模式给出 overfit_prompt:{step,val_loss,best_step,best_val}。 */
export function overfitOf(status: any): any | null {
  const p = status && status.overfit_prompt
  if (!p || typeof p !== "object") return null
  return p
}

/** 能不能「保留这一步」：训练没停时不允许（与 ms_train_keep 的拦截一致）。 */
export function canKeep(status: any): boolean {
  if (!status) return false
  const phase = String(status.phase || "")
  return ["training", "preparing", "preprocessing", "previewing"].indexOf(phase) < 0
}
