// 为她而唱 · 素材库（2026-11-04 重做）
//
// 「准备素材」页和「训练声音」第 1 步共用这一份状态：素材列表、投递、时长统计。
// 接口照旧（ms_corpus_list / ms_corpus_add / ms_corpus_job / ms_corpus_clip / ms_corpus_requirements），
// 这里只把返回值整理成人话，不改任何数据流。模块级单例，切页面不丢正在跑的投递任务。
import { useEffect, useState } from "@neko/plugin-ui"
import { callEntry, uploadFileChunked } from "./api"

export type Material = { slot: string; title: string; lead: string; how: string; ok: string }

// 方案第 2 节的卡片文案：只讲「投什么、有什么用」，不出现任何字段名
export const MATERIALS: Material[] = [
  {
    slot: "song", title: "带伴奏的完整歌曲",
    lead: "直接放平常听的歌，我们会自动去掉伴奏、只留人声。",
    how: "最省事：把常听的歌拖进来就行。",
    ok: "适合：素材最省事",
  },
  {
    slot: "dry_sing", title: "清唱 / 纯人声",
    lead: "没有伴奏的清唱录音，效果最好。",
    how: "手机录的清唱、或剪掉伴奏的人声都可以。",
    ok: "效果最好",
  },
  {
    slot: "speech", title: "平时说话的录音",
    lead: "聊天、朗读都行。默认不算进「有效时长」，可以打开「也算进去」。",
    how: "正常说话就好，别刻意模仿，也别唱歌。",
    ok: "只在素材不够时补一点",
  },
]

/** 界面上统一叫法（训练页也用它，不再出现「整首歌曲（带伴奏）」这类不同名字）。 */
export function materialTitle(slot: string): string {
  for (const m of MATERIALS) if (m.slot === slot) return m.title
  return slot
}

export type PendingUpload = {
  id: string; slot: string; name: string; size: number
  pct: number; state: "upload" | "queued"; at: number
}

export type CorpusJob = { job_id: string; slot: string; file: string; pct?: number; phase?: string; mock?: boolean; error?: string }

/**
 * 一个槽位的整体进度：多文件时按「已完成的条数 + 当前这条的百分比」算，
 * 而不是只显示第一条的进度。
 */
export function jobProgress(jobs: CorpusJob[], slot: string) {
  const list = (jobs || []).filter((j) => j && j.slot === slot)
  const total = list.length
  const finished = list.filter((j) => j.phase === "done" || j.phase === "error").length
  const active = list.find((j) => j.phase !== "done" && j.phase !== "error") || null
  const pct = total
    ? Math.min(100, Math.round(((finished + (active ? Math.max(0, Math.min(100, Number(active.pct || 0))) / 100 : 0)) / total) * 100))
    : 0
  return { total, finished, pct, active: Boolean(active), current: active }
}

export type CorpusState = {
  list: any
  busy: boolean
  error: string
  jobs: CorpusJob[]
  pending: PendingUpload[]
  notice: { tone: string; text: string } | null
  req: any
  countSpeech: boolean
  setCountSpeech: (v: boolean) => void
  setNotice: (n: any) => void
  refresh: () => Promise<void>
  /** 按路径投递；成功返回 true */
  addPath: (slot: string, path: string) => Promise<boolean>
  /** 选择/拖入的文件；返回收下的个数 */
  addFiles: (slot: string, files: FileList | File[] | null) => Promise<number>
  removeSource: (id: string) => void
  setClip: (clipId: string, action: "reject" | "restore" | "delete") => Promise<void>
}

let STATE: CorpusState | null = null
const SUBS = new Set<() => void>()
export function materialSubs() { return SUBS }

function emit() { SUBS.forEach((f) => f()) }

/** 模块级单例：第一次用到时创建，之后所有页面共用。 */
let CORPUS_POLL: any = null

export function useCorpus(api: any): CorpusState {
  const [, force] = useState(0)
  useEffect(() => {
    const f = () => force((n) => n + 1)
    SUBS.add(f)
    return () => { SUBS.delete(f) }
  }, [])
  if (!STATE) STATE = createState(api)
  // 素材在后台是单线程队列慢慢处理的；页面不刷新的话，进度会一直停在刚打开时那一眼。
  if (!CORPUS_POLL && typeof window !== "undefined") {
    CORPUS_POLL = setInterval(() => {
      const s = STATE
      if (!s || s.busy || (s as any)._refreshing) return
      ;(s as any)._refreshing = true
      s.refresh().finally(() => { (s as any)._refreshing = false })
    }, 4000)
  }
  return STATE
}

function createState(api: any): CorpusState {
  const s: CorpusState = {
    list: null, busy: false, error: "", jobs: [], pending: [], notice: null, req: null, countSpeech: false,
    setCountSpeech: (v: boolean) => { s.countSpeech = v; emit() },
    setNotice: (n: any) => { s.notice = n; emit() },
    refresh: async () => {
      try {
        const r: any = await callEntry(api, "ms_corpus_list", {}, 60000)
        s.list = r
        s.error = ""
        prunePending()
      } catch (e: any) {
        s.error = String((e && e.message) || e)
      }
      emit()
    },
    addPath: async (slot: string, path: string) => {
      s.busy = true; emit()
      try {
        const r: any = await callEntry(api, "ms_corpus_add", { slot, path }, 120000)
        const name = path.split(/[\\/]/).pop() || path
        if (r && r.job_id) track(slot, name, String(r.job_id))
        else s.setNotice({ tone: "error", text: "没能收下这个文件：" + ((r && r.error) || "未返回任务号") })
        await s.refresh()
        return true
      } catch (e: any) {
        s.setNotice({ tone: "error", text: "没能收下这个文件：" + String((e && e.message) || e) })
        return false
      } finally {
        s.busy = false; emit()
      }
    },
    addFiles: async (slot: string, files: FileList | File[] | null) => {
      const arr: File[] = files ? Array.prototype.slice.call(files) : []
      if (!arr.length) return 0
      // 选完立刻把整批列出来（带上传进度），不要等一个个传完才出现
      const now = Date.now()
      const rows: PendingUpload[] = arr.map((f: File, i: number) => ({
        id: "up-" + now + "-" + i, slot, name: f.name, size: f.size, pct: 0,
        state: "upload", at: now,
      }))
      s.pending = s.pending.concat(rows)
      s.busy = true; emit()
      let ok = 0
      // 一个一个来：给用户"处理哪个才读哪个"，一次只在内存里放一个文件的一片。
      // 选文件时只拿 File 句柄（指向磁盘，本身不占内存），不预先读成 base64。
      for (let i = 0; i < arr.length; i++) {
        const f = arr[i]
        const row = rows[i]
        try {
          const r: any = await uploadFileChunked(api, slot, f, 300000, (pct: number) => {
            row.pct = pct
            emit()
          })
          if (r && r.job_id) {
            row.state = "queued"; row.pct = 100
            track(slot, f.name, String(r.job_id)); ok += 1
          } else if (r && r.duplicate) {
            dropPending(row)
            s.setNotice({ tone: "info", text: "「" + f.name + "」已经在素材里了，跳过" })
          } else {
            dropPending(row)
            s.setNotice({ tone: "error", text: "「" + f.name + "」没能收下：" + ((r && r.error) || "未返回任务号") })
          }
        } catch (e: any) {
          dropPending(row)
          s.setNotice({ tone: "error", text: "「" + f.name + "」没能收下：" + String((e && e.message) || e) })
        }
        emit()
      }
      // 上传完就把这批文件句柄放掉，避免浏览器一直攥着几十个 File 对象
      arr.length = 0
      s.busy = false
      await s.refresh()
      prunePending()
      emit()
      return ok
    },
    removeSource: (id: string) => {
      callEntry(api, "ms_corpus_remove", { source_id: id }, 30000).then(() => s.refresh()).catch((e: any) => {
        s.setNotice({ tone: "error", text: String((e && e.message) || e) })
      })
    },
    setClip: async (clipId: string, action: "reject" | "restore" | "delete") => {
      try {
        await callEntry(api, "ms_corpus_clip", { clip_id: clipId, action }, 120000)
        await s.refresh()
      } catch (e: any) {
        s.setNotice({ tone: "error", text: String((e && e.message) || e) })
      }
    },
  }

  // 投递任务：排队轮询 ms_corpus_job，一条处理完再轮询下一条
  const queue: CorpusJob[] = []
  let pumping = false
  function track(slot: string, file: string, jobId: string) {
    const j: CorpusJob = { job_id: jobId, slot, file, pct: 0, phase: "queued" }
    s.jobs.push(j)
    queue.push(j)
    emit()
    pump()
  }
  async function pump() {
    if (pumping) return
    pumping = true
    while (queue.length) {
      const j = queue[0]
      let failed = ""
      for (let i = 0; i < 2400; i++) {
        await new Promise((res) => setTimeout(res, 900))
        let rec: any = null
        try { rec = await callEntry(api, "ms_corpus_job", { job_id: j.job_id }, 30000) } catch (e: any) { rec = null }
        if (!rec) continue
        // 任务在插件里已经不存在（插件重启/被终止）→ 立刻停下来，不要对着死任务一直轮询
        if (rec.ok === false) { failed = String(rec.error || "任务不存在"); break }
        j.pct = Number(rec.pct || 0)
        j.phase = String(rec.phase || "")
        if (rec.error) { failed = String(rec.error); break }
        if (rec.phase === "done") break
        emit()
      }
      if (failed) {
        const lost = /找不到任务|不存在/.test(failed)
        s.setNotice({
          tone: "error",
          text: lost
            ? "「" + j.file + "」的处理被中断了（插件重启过）。文件还在素材库里，重新添加或点「恢复未入库文件」即可。"
            : "「" + j.file + "」处理失败：" + failed,
        })
        j.phase = "error"; j.error = failed; j.pct = 100
      } else {
        j.phase = "done"; j.pct = 100
      }
      queue.shift()
      emit()
      s.refresh()
      // 留着这条 8 秒，让「整体进度」能走到 100%，之后自动从列表里收掉
      setTimeout(() => {
        const k = s.jobs.indexOf(j)
        if (k >= 0) { s.jobs.splice(k, 1); emit() }
      }, 8000)
    }
    pumping = false
  }

  /** 移除一条上传占位行（失败/重复时） */
  function dropPending(row: PendingUpload) {
    s.pending = s.pending.filter((p) => p !== row)
  }

  /**
   * 刷新后清掉已经变成正式素材的占位行：
   * 优先按「槽位 + 文件名 + 大小」对；对不上的（同名被自动改名等）最多留 2 分钟。
   */
  function prunePending() {
    const keys = new Set<string>()
    const slots = (s.list && s.list.slots) || {}
    for (const k of Object.keys(slots)) {
      for (const src of ((slots as any)[k] && (slots as any)[k].files) || []) {
        for (const f of (src && src.files) || []) {
          if (f && f.name) keys.add(k + "|" + String(f.name).toLowerCase() + "|" + String(f.bytes || 0))
        }
      }
    }
    const now = Date.now()
    s.pending = s.pending.filter((p) => {
      if (now - p.at > 120000) return false
      return !keys.has(p.slot + "|" + p.name.toLowerCase() + "|" + String(p.size))
    })
  }

  // 有素材在处理时 2.5 秒刷一次（进度条才走得动），空闲时 20 秒一次
  const anyProcessing = () => {
    const slots = (s.list && s.list.slots) || {}
    for (const k of Object.keys(slots)) {
      for (const src of ((slots as any)[k] && (slots as any)[k].files) || []) {
        if (src && src.status === "processing") return true
      }
    }
    return false
  }
  let timer: any = null
  const tick = async () => {
    if (!s.busy) {
      await s.refresh()
      emit() // 刷新后必须通知界面重画，否则进度条不会更新
    }
    timer = setTimeout(tick, anyProcessing() ? 2500 : 20000)
  }
  tick()
  if (typeof window !== "undefined") {
    // 预览页卸载时清掉定时器，避免重复请求
    ;(window as any).__msCorpusStop = () => clearTimeout(timer)
  }
  return s
}
