// 为她而唱 · 训练声音模型（分步向导，2026-11-03 改造版）
//
// 结构：Step 1 素材准备 → Step 2 训练设置 → Step 3 训练监控 → Step 4 保存模型，
// 分步只靠本页的 `step` 状态切换，不改任何接口与数据流（mock 拦截原样保留）。
// 「开始训练」先弹极简确认框（步数 / 素材时长 / 预计耗时），确认后才真正发 start；
// 到试听节点弹「试听卡片」，可「保留这一步」或「继续训练」，也能直接关掉继续看别的节点。
// 参数、硬件、切片明细等收进默认折叠的「高级网络设置」<details>，主界面只留该点的东西。
// 接口名严格按 §5；后端没就绪时自动切到 mock，页头只留一个小气泡写明「模拟演示」。
import { Alert, Divider, StatusBadge, useEffect, useRef, useState } from "@neko/plugin-ui"
import { callEntry, dataUriOf, fileToDataUri, refreshContext } from "./api"
import { MOCK_NOTE, mockCorpus, mockHw, mockJob, mockPreview, mockRuns, mockStatus } from "./mock"
import { CLIP_TYPE, SLOT_TYPE, TONE_OF } from "./kinds"
import {
  SLOTS, STEPS_CHOICES, canKeep, clipList, deficitText, effectiveSeconds, flagText, milestoneList,
  overfitOf, phaseText, progressText, requirementOf, slotView,
} from "./trainModel"
import { AudioBox, Bar, Btn, Col, Confirm, DropZone, Empty, Err, FileRow, Input, KV, Muted, Panel, Pill, Row, Select, Toggle, fmtEta, fmtTime } from "./ui"
import { BRAND, BRAND_DEEP, INK, INK_2, LINE } from "./ui"


type CorpusJob = { job_id: string; slot: string; file: string }

// ---- 纯样式（视觉极简改造集中在这里） -------------------------------------
const PRIMARY = { border: 0, background: BRAND, color: "#fff", borderRadius: 12, boxShadow: "0 6px 16px rgba(224,45,118,0.22)" }
const PRIM_TXT = { background: "none", border: 0, color: BRAND, padding: "4px 2px", fontWeight: 600 }
const CARD_FILL = { backgroundImage: "linear-gradient(150deg,#ffffff 0%,#fffdfe 46%,#fdf7fa 100%)", border: "1px solid rgba(255,255,255,0.94)" }
const CARD = { padding: 24, borderRadius: 16, ...CARD_FILL, boxShadow: "0 7px 24px rgba(142,159,187,0.06)" }
const LIGHT = { padding: 24, borderRadius: 16, ...CARD_FILL, boxShadow: "0 7px 24px rgba(142,159,187,0.06)" }
// 投递卡片里的小信息条
const GREY = { padding: 16, background: "rgba(255,246,250,0.86)", border: "1px solid rgba(224,45,118,0.07)", borderRadius: 14 }
const STEP_OF = ["素材准备", "训练设置", "训练监控", "保存模型"]

function Steps(props: { cur: number; ready: boolean[]; onPick: (i: number) => void }) {
  return (
    <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
      {STEP_OF.map((t: string, i: number) => {
        const no = i + 1
        const on = props.cur === no
        const ok = props.ready[i]
        return (
          <button key={no} onClick={() => props.onPick(no)} title={"第 " + no + " 步：" + t} style={{
            display: "inline-flex", alignItems: "center", gap: 6, padding: "6px 12px", cursor: "pointer",
            fontSize: 13, borderRadius: 999, fontFamily: "inherit",
            border: "0", background: on ? BRAND : "transparent", color: on ? "#fff" : INK_2,
            fontWeight: on ? 700 : 400, boxShadow: on ? "0 6px 16px rgba(224,45,118,0.22)" : "none",
          }}>
            <i style={{
              width: 18, height: 18, borderRadius: 999, fontSize: 11, lineHeight: "18px", textAlign: "center",
              fontStyle: "normal", color: on ? BRAND : ok ? "#fff" : "#b9b3c1",
              background: on ? "#fff" : ok ? BRAND : "rgba(224,45,118,0.12)",
            }}>{ok && !on ? "✓" : no}</i>
            {t}
          </button>
        )
      })}
    </div>
  )
}

/** 折叠区：默认收起，只在展开时才挂 children（省得铺一屏小字）。 */
function Fold(props: { title: string; children: () => any }) {
  return (
    <details style={{ fontSize: 12.5 }}>
      <summary style={{ cursor: "pointer", color: BRAND_DEEP, padding: "4px 0", fontWeight: 600 }}>{props.title}</summary>
      <div style={{ padding: "6px 0 0 2px" }}>{props.children()}</div>
    </details>
  )
}

/** 跨步导航：上一步 / 下一步（下一步可以带主色）。 */
function Foot(props: { prev?: number; next?: number; nextText?: string; nextMain?: boolean; onGo: (i: number) => void; hint?: any }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, marginTop: 18, paddingTop: 12, borderTop: "1px solid " + LINE }}>
      <div>{props.prev ? <Btn style={PRIM_TXT} onClick={() => props.onGo(props.prev as number)}>← 上一步</Btn> : null}</div>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        {props.hint || null}
        {props.next ? (
          <Btn style={props.nextMain ? PRIMARY : undefined} onClick={() => props.onGo(props.next as number)}>{props.nextText || "下一步"}</Btn>
        ) : null}
      </div>
    </div>
  )
}


export function Train(props: { api: any; snap: any; onRefresh: () => void }) {
  const { api, snap } = props

  // ---- 向导分步（只影响显示，不做任何数据请求） -------------------------
  const [step, setStep] = useState<number>(1)
  const [confirmOpen, setConfirmOpen] = useState<boolean>(false)
  const [nodeOpen, setNodeOpen] = useState<boolean>(false)
  const [nodeStep, setNodeStep] = useState<number>(0)
  // 试听卡片是「到点弹出」：每遇到一个新节点自动打开一次；手动关掉后不再自己弹回来
  const autoNode = useRef<string>("")

  // ---- 首次向导（本页内的一步流程，用本地状态记住进度） -------------------
  const [steps, setSteps] = useState<number>(8000)
  const [name, setName] = useState<string>("")
  const [auto, setAuto] = useState<boolean>(false)

  // ---- 硬件与语料 -------------------------------------------------------
  const [hw, setHw] = useState<any>(null)
  const [hwMock, setHwMock] = useState<boolean>(false)
  const [req, setReq] = useState<any>(null)
  const [corpus, setCorpus] = useState<any>(null)
  const [corpusMock, setCorpusMock] = useState<boolean>(false)
  const [loadErr, setLoadErr] = useState<string>("")
  const [countSpeech, setCountSpeech] = useState<boolean>(false)

  // ---- 投递 -------------------------------------------------------------
  const [addText, setAddText] = useState<any>({ song: "", dry_sing: "", speech: "" })
  const [jobs, setJobs] = useState<CorpusJob[]>([])
  const [jobState, setJobState] = useState<any>({})

  // ---- 训练 -------------------------------------------------------------
  const [runs, setRuns] = useState<any>(null)
  const [runId, setRunId] = useState<string>("")
  const [status, setStatus] = useState<any>(null)
  const [statusMock, setStatusMock] = useState<boolean>(false)
  const [pick, setPick] = useState<number>(0)
  const [keepName, setKeepName] = useState<string>("")
  const [keepMsg, setKeepMsg] = useState<string>("")
  const [keepCkpt, setKeepCkpt] = useState<string>("")
  const [previews, setPreviews] = useState<any>({})
  const [busy, setBusy] = useState<string>("")
  const [msg, setMsg] = useState<{ tone: string; text: string } | null>(null)
  const [overfitSeen, setOverfitSeen] = useState<string>("")
  const mockStep = useRef<number>(0)
  const mockTotal = useRef<number>(8000)

  const mock = hwMock || corpusMock || statusMock

  // ---- 读取：硬件 -------------------------------------------------------
  useEffect(() => {
    let alive = true
    callEntry(api, "ms_hw_profile", {}, 60000).then((r: any) => {
      if (!alive) return
      if (r && r.ok && r.train) { setHw(r); setHwMock(Boolean(r.mock)) }
      else { setHw(mockHw()); setHwMock(true) }
    }).catch(() => { if (alive) { setHw(mockHw()); setHwMock(true) } })
    return () => { alive = false }
  }, [])

  // ---- 读取：所需时长（随步数变化） ------------------------------------
  useEffect(() => {
    let alive = true
    callEntry(api, "ms_corpus_requirements", { steps }, 60000).then((r: any) => {
      if (alive) setReq(r && r.ok ? r : null)
    }).catch(() => { if (alive) setReq(null) })
    return () => { alive = false }
  }, [steps])

  // ---- 读取：语料库 -----------------------------------------------------
  const loadCorpus = async (extra?: any) => {
    try {
      const r: any = await callEntry(api, "ms_corpus_list", {}, 60000)
      if (r && r.ok && r.slots) { setCorpus(r); setCorpusMock(Boolean(r.mock)); return }
      setCorpus(mockCorpus(extra)); setCorpusMock(true)
    } catch (e: any) {
      setLoadErr(String((e && e.message) || e))
      setCorpus(mockCorpus(extra)); setCorpusMock(true)
    }
  }
  useEffect(() => { loadCorpus() }, [])

  // ---- 读取：训练任务与状态 --------------------------------------------
  const loadRuns = async (keep?: string) => {
    try {
      const r: any = await callEntry(api, "ms_train_runs", {}, 60000)
      if (r && r.ok && Array.isArray(r.runs)) {
        setRuns(r)
        const cur = keep || runId || r.active || (r.runs[0] && r.runs[0].run_id) || ""
        setRunId(cur)
        return
      }
    } catch (e: any) { /* 落到 mock */ }
    setRuns(mockRuns())
    if (!keep && !runId) setRunId("mock-run-001")
  }
  useEffect(() => { loadRuns() }, [])

  const pollStatus = async () => {
    if (statusMock || String(runId).startsWith("mock-")) {
      const st = mockStatus(mockStep.current, mockTotal.current)
      setStatus(st)
      setStatusMock(true)
      return
    }
    if (!runId) { setStatus(null); return }
    try {
      const r: any = await callEntry(api, "ms_train_status", { run_id: runId }, 60000)
      if (r && r.ok && r.phase) { setStatus(r); setStatusMock(false); return }
      if (r && r.ok === false && r.error) { setStatus(r); setStatusMock(false); return }
    } catch (e: any) { /* 落到 mock */ }
    setStatus(mockStatus(mockStep.current || 8000, mockTotal.current))
    setStatusMock(true)
  }
  useEffect(() => { pollStatus() }, [runId])

  // 训练中每 2 秒刷一次状态、每 1 秒推进模拟进度
  useEffect(() => {
    const id = setInterval(() => {
      const phase = String((status && status.phase) || "")
      if (["training", "preparing", "preprocessing", "previewing", "stopping"].indexOf(phase) >= 0) {
        if (statusMock || String(runId).startsWith("mock-")) {
          mockStep.current = Math.min(mockTotal.current, mockStep.current + 60)
        }
        pollStatus()
      }
    }, 2000)
    return () => clearInterval(id)
  }, [status, runId, statusMock])

  // ---- 投递任务的进度轮询 ----------------------------------------------
  useEffect(() => {
    if (!jobs.length) return
    let alive = true
    const tick = async () => {
      for (const j of jobs) {
        const cur = jobState[j.job_id]
        if (cur && (cur.phase === "done" || cur.phase === "error")) continue
        let rec: any
        if (j.job_id.startsWith("mock-")) {
          const pct = Math.min(100, Number((cur && cur.pct) || 0) + 25)
          rec = mockJob(j.job_id, pct)
        } else {
          try { rec = await callEntry(api, "ms_corpus_job", { job_id: j.job_id }, 30000) } catch (e: any) {
            rec = { phase: "error", pct: 0, stage_text: String((e && e.message) || e), error: String((e && e.message) || e) }
          }
        }
        if (!alive) return
        setJobState((old: any) => ({ ...old, [j.job_id]: rec }))
        if (rec && rec.phase === "done") loadCorpus()
      }
    }
    tick()
    const id = setInterval(tick, 2000)
    return () => { alive = false; clearInterval(id) }
  }, [jobs.length, jobState])

  // ---- 过拟合弹窗 -------------------------------------------------------
  const ov = overfitOf(status)
  const ovKey = ov ? String((ov.step || "") + "|" + (ov.val_loss || "")) : ""

  // ---- 动作 -------------------------------------------------------------
  function startMockRun(total: number) {
    mockTotal.current = total
    mockStep.current = 0
    setStatusMock(true)
    setRunId("mock-run-001")
    setRuns(mockRuns())
    setStatus(mockStatus(0, total))
  }

  const startTrain = async () => {
    const have = effectiveSeconds(corpus, countSpeech)
    const need = requirementOf(req, steps)
    if (have < need.min) { setMsg({ tone: "error", text: "有效时长不够：还差 " + deficitText(have, need.min) }); return }
    setBusy("start")
    setMsg(null)
    let started = false
    try {
      const r: any = await callEntry(api, "ms_train_start", { steps, from_corpus: true, auto, name: name.trim() || undefined }, 300000)
      if (r && r.ok && r.run_id) {
        setRunId(r.run_id)
        setStatusMock(false)
        setMsg({ tone: "info", text: "已开始训练：" + (r.name || r.run_id) + (auto ? "（全自动）" : "（到节点会停下等你试听）") })
        await loadRuns(r.run_id)
        await pollStatus()
        started = true
      } else if (!r || !r.ok) {
        // 后端明确报错就照实显示，不再假装在训练
        setMsg({ tone: "error", text: "启动失败：" + ((r && r.error) || "未知错误") })
      }
    } catch (e: any) {
      const text = String((e && e.message) || e)
      setMsg({ tone: "error", text: "启动失败（后端未就绪，改为模拟演示）：" + text })
      startMockRun(steps)
      started = true
    }
    setBusy("")
    // 真正跑起来了才切到「训练监控」；没启动时留在原地，错误就在原处显示
    if (started) { autoNode.current = ""; setStep(3) }
  }

  const control = async (action: string) => {
    setBusy(action)
    try {
      if (String(runId).startsWith("mock-")) {
        setMsg({ tone: "info", text: action === "continue" ? "（模拟）继续训练" : "（模拟）已停止，可以选节点保留" })
        if (action === "continue") mockStep.current = Math.min(mockTotal.current, mockStep.current + 1000)
        else { mockStep.current = mockTotal.current }
        setStatus(mockStatus(mockStep.current, mockTotal.current))
      } else {
        await callEntry(api, "ms_train_control", { run_id: runId, action }, 60000)
        await pollStatus()
        setMsg({ tone: "info", text: action === "continue" ? "已下指令继续训练，到下一节点会再停下。" : "已下指令停止；停下后可以选一个节点保留。" })
      }
    } catch (e: any) {
      setMsg({ tone: "error", text: "指令失败：" + String((e && e.message) || e) })
    }
    setBusy("")
  }

  const playPreview = async (nodeStep: number): Promise<boolean> => {
    setPick(nodeStep)
    if (previews[nodeStep] || previews[String(nodeStep)]) return true
    try {
      let r: any
      if (String(runId).startsWith("mock-")) r = mockPreview()
      else r = await callEntry(api, "ms_train_preview", { run_id: runId, step: nodeStep }, 120000)
      if (r && r.ok && r.audio_base64) {
        // 用同一个地址拼装函数，保证每条试听都是独立的 data: 地址（同一地址再点不会空放）
        setPreviews((old: any) => ({ ...old, [nodeStep]: dataUriOf(r.audio_base64, r.mime || "audio/wav") }))
        return true
      }
      setMsg({ tone: "error", text: "取试听失败：" + ((r && r.error) || "无音频") })
    } catch (e: any) {
      setMsg({ tone: "error", text: "取试听失败：" + String((e && e.message) || e) })
    }
    return false
  }

  const keep = async (nodeStep?: number): Promise<boolean> => {
    if (nodeStep) setPick(nodeStep)
    const at = nodeStep || pick
    if (!at) { setKeepMsg("先在节点上选一步。"); return false }
    if (!canKeep(status)) { setKeepMsg("这条训练还在跑：先点「停止训练」，等状态变成「已到节点」再保留。"); return false }
    setBusy("keep")
    setKeepMsg("")
    let ok = false
    try {
      if (String(runId).startsWith("mock-")) {
        setKeepCkpt("/mock/model_" + at + ".pt")
        setKeepMsg("（模拟）已保留第 " + at + " 步 → /mock/model_" + at + ".pt（未改动当前模型）")
        ok = true
      } else {
        const r: any = await callEntry(api, "ms_train_keep", {
          run_id: runId, step: at, name: keepName.trim() || undefined, set_current: false,
        }, 600000)
        if (r && r.ok) {
          setKeepCkpt(String(r.ckpt || ""))
          setKeepMsg("已保留第 " + at + " 步 → " + (r.ckpt || "（路径未返回）") +
            "。当前变声模型没有改动；要启用请看「设为当前模型」。")
          await loadRuns(runId)
          ok = true
        } else {
          setKeepMsg("保留失败：" + ((r && r.error) || "未知错误"))
        }
      }
    } catch (e: any) {
      setKeepMsg("保留失败：" + String((e && e.message) || e))
    }
    setBusy("")
    return ok
  }

  const setCurrent = async () => {
    if (!keepCkpt) return
    setBusy("current")
    try {
      if (keepCkpt.startsWith("/mock/")) { setKeepMsg("（模拟）已把 " + keepCkpt + " 设为当前变声模型。"); }
      else {
        const r: any = await callEntry(api, "ms_set_active_model", { ckpt: keepCkpt }, 60000)
        setKeepMsg(r && r.ok ? "已设为当前变声模型：" + keepCkpt : "切换失败：" + ((r && r.error) || "未知错误"))
      }
      await refreshContext(api)
      props.onRefresh()
    } catch (e: any) {
      setKeepMsg("切换失败：" + String((e && e.message) || e))
    }
    setBusy("")
  }

  const addTo = async (slot: SLOT_TYPE, files?: FileList | null) => {
    const text = String(addText[slot] || "").trim()
    if (!files || !files.length) {
      if (!text) { setMsg({ tone: "error", text: "填本地路径，或用下面的「选择文件」投递。" }); return }
      try {
        const r: any = await callEntry(api, "ms_corpus_add", { slot, path: text }, 120000)
        if (r && r.job_id) { setJobs((old) => old.concat([{ job_id: r.job_id, slot, file: text }])); setAddText({ ...addText, [slot]: "" }) }
        else setMsg({ tone: "error", text: "投递失败：" + ((r && r.error) || "未返回 job_id") })
      } catch (e: any) {
        setMsg({ tone: "error", text: "投递失败：" + String((e && e.message) || e) })
      }
      return
    }
    for (let i = 0; i < files.length; i++) {
      const f = files[i]
      try {
        const uri = await fileToDataUri(f)
        const b64 = uri.slice(uri.indexOf(",") + 1)
        const r: any = await callEntry(api, "ms_corpus_add", { slot, data_b64: b64, filename: f.name }, 300000)
        if (r && r.job_id) setJobs((old) => old.concat([{ job_id: r.job_id, slot, file: f.name }]))
        else setMsg({ tone: "error", text: "投递失败：" + ((r && r.error) || "未返回 job_id") })
      } catch (e: any) {
        const m = String((e && e.message) || e)
        setMsg({ tone: "error", text: "投递失败（后端未就绪，用模拟任务演示）：" + m })
        setJobs((old) => old.concat([{ job_id: "mock-" + slot + "-" + Date.now() + "-" + i, slot, file: f.name }]))
      }
    }
  }

  const clipAction = async (c: CLIP_TYPE, action: string) => {
    if (corpusMock) {
      if (action === "preview") {
        const r: any = mockPreview()
        setPreviews((p: any) => ({ ...p, ["clip-" + c.id]: dataUriOf(r.audio_base64, r.mime) }))
        return
      }
      setCorpus((old: any) => {
        const next = JSON.parse(JSON.stringify(old))
        next.clips = next.clips.map((x: any) =>
          x.id === c.id ? { ...x, status: action === "reject" ? "rejected" : "ok", reason: action === "reject" ? (x.reason || "手动剔除") : "" } : x)
        return next
      })
      return
    }
    try {
      const r: any = await callEntry(api, "ms_corpus_clip", { clip_id: c.id, action }, 120000)
      if (r && r.audio_base64) setPreviews((p: any) => ({ ...p, ["clip-" + c.id]: dataUriOf(r.audio_base64, r.mime || "audio/wav") }))
      loadCorpus()
    } catch (e: any) {
      setMsg({ tone: "error", text: "切片操作失败：" + String((e && e.message) || e) })
    }
  }

  // ---- 渲染 -------------------------------------------------------------
  const need = requirementOf(req, steps)
  const have = effectiveSeconds(corpus, countSpeech)
  const gap = deficitText(have, need.min)
  const canStart = !!corpus && have >= need.min && !busy
  const prog = progressText(status)
  const nodes = milestoneList(status)
  const clips = clipList(corpus)
  const runPhase = String((status && status.phase) || "")
  const ended = ["finished", "failed", "stopped", "awaiting_choice"].indexOf(runPhase) >= 0
  const corpusClips = clips.filter((c: CLIP_TYPE) => !c.status || c.status === "ok")

  const nodeCard = (() => {
    if (!nodeStep) return null
    for (const n of nodes) if (n.step === nodeStep) return n
    return null
  })()

  // 到点弹一次试听卡片：每个节点只自动弹一次，手动关掉后不再自己弹回来
  const reached = nodes.filter((n: any) => n.ready && n.step <= Number((status && status.step) || 0))
  const newest = reached.length ? reached[reached.length - 1].step : 0
  const newestKey = newest ? String(runId) + "|" + newest : ""
  useEffect(() => {
    if (!newestKey || autoNode.current === newestKey) return
    autoNode.current = newestKey
    setNodeStep(newest)
    playPreview(newest)
    setNodeOpen(true)
  }, [newestKey])

  const goto = (i: number) => { setStep(Math.max(1, Math.min(4, i))); }
  // 手动打开节点：要等试听拿到再弹（或提示失败），不然卡片里会一直是空的
  const openNode = async (s: number) => {
    setPick(s)
    const ok = await playPreview(s)
    if (ok) { setNodeStep(s); setNodeOpen(true) }
    else setMsg({ tone: "error", text: "第 " + s + " 步的试听还没生成好，稍后再试。" })
  }
  const keepFromCard = () => {
    if (busy) return
    keep(nodeStep).then((ok: boolean) => { if (ok) setNodeOpen(false) })
  }
  // 卡片开着时点了「继续训练」：关卡片 + 下发继续；关卡片本身只影响显示，没有任何请求
  const continueTrain = () => {
    setNodeOpen(false)
    if (runPhase === "awaiting_choice" || runPhase === "stopped") control("continue")
  }

  const readyFlags = [
    !!corpus && have > 0,
    have > 0 && !gap,
    !!status,
    !!keepCkpt,
  ]
  const showStart = step === 3 && !status

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>

      {/* 页头：标题 + 步骤条 + 一处「模拟演示」小气泡 */}
      <div>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, flexWrap: "wrap" }}>
          <div>
            <div style={{ fontSize: 17, fontWeight: 600 }}>训练声音模型</div>
            <div style={{ fontSize: 12, color: "#8a97ad", marginTop: 3 }}>第 {step} 步 / 4 · {STEP_OF[step - 1]}</div>
          </div>
          {mock ? (
            <span title={MOCK_NOTE} style={{
              fontSize: 11.5, color: "#c8861a", background: "#fff6e8", padding: "3px 10px",
              borderRadius: 999, cursor: "help", whiteSpace: "nowrap",
            }}>模拟演示</span>
          ) : null}
        </div>
        <div style={{ marginTop: 10 }}>
          <Steps cur={step} ready={readyFlags} onPick={goto} />
        </div>
      </div>

      {loadErr ? <Alert tone="warning">读取语料库失败：{loadErr}</Alert> : null}
      {msg ? <Alert tone={msg.tone === "error" ? "warning" : "info"}>{msg.text}</Alert> : null}
      {overfitSeen && ovKey && overfitSeen !== ovKey ? (
        <Alert tone="warning">检测到过拟合迹象（第 {ov.step} 步），请在弹窗里选择继续或停止；停止后可以挑一个更早的节点保留。</Alert>
      ) : null}

      {/* ===== Step 1：素材准备 ===== */}
      {step === 1 ? (
        <div style={CARD}>
          <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 10, flexWrap: "wrap" }}>
            <div style={{ fontSize: 14.5, fontWeight: 600 }}>素材准备</div>
            <div style={{ fontSize: 12, color: "#8a97ad" }}>
              所需 {steps} 步：最少 {fmtTime(need.min)} · 推荐 {fmtTime(need.rec)}
            </div>
          </div>
          <div style={{ fontSize: 12.5, color: INK_2, margin: "4px 0 14px" }}>
            三类素材按需投递即可，只有人声类会被算进「有效时长」；拖进文件也行，填路径也行。
          </div>

          {/* 三张投递卡片：响应式并排；选中文件后中间变成文件列表 */}
          <div style={{ display: "grid", gap: 24, gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))" }}>
            {SLOTS.map((meta: any) => {
              const view = slotView(corpus, meta.key)
              const slotClips = clips.filter((c: CLIP_TYPE) => c.slot === meta.key)
              const slotFiles = (corpus && corpus.slots && corpus.slots[meta.key] && Array.isArray(corpus.slots[meta.key].files))
                ? corpus.slots[meta.key].files : []
              const rows = slotFiles.length ? slotFiles.length
                : (typeof view.files === "number" && view.files > 0 ? view.files : 0)
              const fileLi: any[] = []
              for (let i = 0; i < rows; i++) fileLi.push(slotFiles[i] || "素材 #" + (i + 1))
              return (
                <div key={meta.key} style={{ ...LIGHT, padding: 20, display: "flex", flexDirection: "column", gap: 12 }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
                    <div style={{ fontSize: 15, fontWeight: 600 }}>{meta.title}</div>
                    <Pill tone={TONE_OF[meta.key]}>{view.clips} 段切片</Pill>
                  </div>
                  <div style={{ fontSize: 12, color: "#8a97ad" }}>{meta.note}</div>

                  <DropZone onFiles={(f: any) => addTo(meta.key, f)} />
                  <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <Input placeholder="或粘贴本地路径…" value={addText[meta.key] || ""} style={{ flex: 1, minWidth: 0 }}
                           onChange={(e: any) => setAddText({ ...addText, [meta.key]: e.target.value })} />
                    <Btn style={PRIM_TXT} onClick={() => addTo(meta.key)}>投递</Btn>
                  </div>

                  {fileLi.length ? (
                    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                      {fileLi.map((name: any, i: number) => {
                        const cl = slotClips[i]
                        return (
                          <FileRow key={name + i} title={name}
                                   meta={cl ? fmtTime(cl.dur_s) + (cl.status === "rejected" ? " · 已剔除" : "") : "已投递 · 质检中"}
                                   right={cl ? <Btn style={PRIM_TXT} onClick={() => clipAction(cl, "preview")}>试听</Btn> : null} />
                        )
                      })}
                    </div>
                  ) : (
                    <div style={{ fontSize: 12, color: "#9aa6bb" }}>
                      {view.files} 个文件 · 有效 {fmtTime(view.valid_s)} · 还没投递素材
                    </div>
                  )}

                  <div style={{ fontSize: 11.5, color: "#9aa6bb", lineHeight: 1.6 }}>{meta.need}</div>
                  {slotClips.length ? (
                    <Fold title={"质检明细（" + slotClips.length + " 段 · 可试听 / 剔除）"}>
                      {() => slotClips.map((c: CLIP_TYPE) => (
                        <div key={c.id} style={{ display: "flex", alignItems: "center", gap: 8, padding: "3px 0", flexWrap: "wrap" }}>
                          <StatusBadge tone={c.status === "rejected" ? "error" : "success"}>{c.status === "rejected" ? "已剔除" : "通过"}</StatusBadge>
                          <Muted>{fmtTime(c.dur_s)} · 峰值 {c.peak ?? "—"}{c.reason ? " · " + c.reason : ""}</Muted>
                          <Btn style={PRIM_TXT} onClick={() => clipAction(c, "preview")}>试听</Btn>
                          <Btn style={PRIM_TXT} onClick={() => clipAction(c, c.status === "rejected" ? "restore" : "reject")}>
                            {c.status === "rejected" ? "恢复" : "剔除"}
                          </Btn>
                          {previews["clip-" + c.id] ? <AudioBox src={previews["clip-" + c.id]} keyId={"clip-" + c.id} /> : null}
                        </div>
                      ))}
                    </Fold>
                  ) : null}
                </div>
              )
            })}
          </div>

          {jobs.length ? (
            <div style={{ marginTop: 14 }}>
              <Muted>投递进度</Muted>
              <div style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 4 }}>
                {jobs.map((j: CorpusJob) => {
                  const st = jobState[j.job_id] || {}
                  const phase = String(st.phase || "queued")
                  const pct = Number(st.pct || 0)
                  return (
                    <div key={j.job_id} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                        <Pill tone={phase === "done" ? "success" : phase === "error" ? "error" : "neutral"}>{phaseText(phase)}</Pill>
                        <Muted>{j.file} · {pct}%</Muted>
                      </div>
                      <Bar pct={pct} tone={phase === "error" ? "#d64545" : undefined} />
                      {st.stage_text ? <Muted>{st.stage_text}</Muted> : null}
                      {st.error ? <Err>{st.error}</Err> : null}
                    </div>
                  )
                })}
              </div>
            </div>
          ) : null}

          {/* 有效时长统计 */}
          <div style={{ marginTop: 16, padding: "12px 14px", background: "#f2f6ff", borderRadius: 10 }}>
            <div style={{ display: "flex", alignItems: "baseline", gap: 10, flexWrap: "wrap" }}>
              <div style={{ fontSize: 22, fontWeight: 600, color: BRAND_DEEP }}>{fmtTime(have)}</div>
              <span style={{ fontSize: 12.5, color: INK_2 }}>有效时长 · 最少 {fmtTime(need.min)}</span>
              {gap
                ? <StatusBadge tone="warning">{gap}</StatusBadge>
                : <StatusBadge tone="success">已够 {steps} 步</StatusBadge>}
            </div>
            <Bar pct={need.min > 0 ? Math.min(100, (have / need.min) * 100) : 0} />
            <div style={{ marginTop: 6 }}>
              <Fold title="有效时长怎么算">
                {() => (
                  <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                    <Muted>按质检后的切片总长计（整曲 + 清唱{countSpeech ? " + 说话" : ""}），被剔除的切片不算。</Muted>
                    <Toggle on={countSpeech} onChange={(v: boolean) => setCountSpeech(v)}
                            label="把说话 / 朗读也算进有效时长（默认不算）" />
                  </div>
                )}
              </Fold>
            </div>
          </div>

          <Foot next={2} nextText="下一步：训练设置" nextMain onGo={goto}
                hint={<Muted>{gap ? gap + "，可以先去投递" : "时长够了"}</Muted>} />
        </div>
      ) : null}

      {/* ===== Step 2：训练设置 ===== */}
      {step === 2 ? (
        <Panel title="训练设置" extra={<Muted>每 2000 步出一个试听节点</Muted>}>
          <div style={{ fontSize: 12.5, color: INK_2, margin: "4px 0 14px" }}>步数越多越像，也越久。</div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {STEPS_CHOICES.map((v: number) => (
              <Btn key={v} style={v === steps ? PRIMARY : undefined}
                   onClick={() => { setSteps(v); if (String(runId).startsWith("mock-")) mockTotal.current = v }}>
                {v} 步
              </Btn>
            ))}
          </div>

          <div style={{ marginTop: 16, display: "flex", flexDirection: "column", gap: 8 }}>
            <div style={{ fontSize: 13, fontWeight: 600 }}>音色名字</div>
            <Input value={name} placeholder="可留空，按时间自动命名（例如 小夏）" style={{ width: 260 }}
                   onChange={(e: any) => setName(e.target.value)} />
          </div>

          <label style={{ display: "flex", gap: 8, alignItems: "flex-start", fontSize: 13, marginTop: 16, cursor: "pointer" }}>
            <input type="checkbox" checked={auto} style={{ marginTop: 3 }} onChange={(e) => setAuto(e.target.checked)} />
            <span>
              全自动
              <div style={{ fontSize: 12, color: "#8a97ad" }}>到节点不停下，按防过拟合建议自动继续/停止。</div>
            </span>
          </label>

          <div style={{ marginTop: 14, display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
            <Muted>素材有效时长 <b style={{ color: BRAND_DEEP }}>{fmtTime(have)}</b> · 最少 {fmtTime(need.min)}</Muted>
            {gap ? <StatusBadge tone="warning">{gap}</StatusBadge> : <StatusBadge tone="success">够了</StatusBadge>}
          </div>

          <div style={{ marginTop: 10 }}>
            <Fold title="高级网络设置（时长估算与硬件）">
              {() => (
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  <KV items={[
                    ["所需时长", "最少 " + fmtTime(need.min) + " · 推荐 " + fmtTime(need.rec)],
                    ["来源", need.source],
                    ["已有有效时长", fmtTime(have) + (gap ? "（" + gap + "）" : "（够了）")],
                  ]} />
                  {need.reasons.length ? <Muted>{need.reasons.join("；")}</Muted> : null}
                  <label style={{ display: "flex", gap: 6, alignItems: "center", fontSize: 12.5 }}>
                    {/* 原生勾选框藏在 label 里：点文字也能切，屏幕阅读器也有名字 */}
                    <input type="checkbox" aria-label="把说话/朗读也算进有效时长" checked={countSpeech} onChange={(e) => setCountSpeech(e.target.checked)} />
                    <span>把说话/朗读也算进有效时长（默认不算，唱歌用不上）</span>
                  </label>
                  {hw ? (
                    <Muted>
                      本机：{hw.gpu ? hw.gpu.name + " · " + Math.round(Number(hw.gpu.vram_mb || 0) / 1024) + "GB" : "无独立显卡（CPU 训练会很慢）"}
                      {" · " + hw.cpu_threads + " 线程 · " + hw.ram_gb + "GB 内存 · 档位 " + hw.tier}
                      {" · 训练 batch " + ((hw.train || {}).batch_size ?? "?") + " · " + ((hw.train || {}).amp ?? "?")}
                    </Muted>
                  ) : <Muted>正在探测硬件…</Muted>}
                </div>
              )}
            </Fold>
          </div>

          <Foot prev={1} next={3} nextMain nextText="下一步：训练监控" onGo={goto}
                hint={<Muted>开始训练会再确认一次</Muted>} />
        </Panel>
      ) : null}

      {/* ===== Step 3：训练监控 ===== */}
      {step === 3 ? (
        <div style={CARD}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, flexWrap: "wrap" }}>
            <div style={{ fontSize: 14.5, fontWeight: 600 }}>训练监控</div>
            <div style={{ display: "flex", gap: 4 }}>
              <Btn style={PRIM_TXT} onClick={() => loadRuns(runId).then(() => pollStatus())}>刷新</Btn>
              {(runs && runs.runs && runs.runs.length) ? (
                <Select value={runId} onChange={(e: any) => { setRunId(e.target.value); setStatus(null); setPick(0) }}>
                  {runs.runs.map((r: any) => (
                    <option key={r.run_id} value={r.run_id}>
                      {r.run_id} · {r.name || ""} · {r.ok === false ? "读取失败" : phaseText(r.phase)}
                    </option>
                  ))}
                </Select>
              ) : null}
            </div>
          </div>

          {status ? (
            status.ok === false ? <div style={{ marginTop: 12 }}><Err>{status.error || "读取训练状态失败"}</Err></div> : (
              <div style={{ display: "flex", flexDirection: "column", gap: 14, marginTop: 14 }}>
                <div>
                  <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                    <StatusBadge tone={runPhase === "failed" ? "error" : runPhase === "awaiting_choice" ? "warning" : runPhase === "finished" ? "success" : "neutral"}>
                      {phaseText(runPhase)}
                    </StatusBadge>
                    <span style={{ fontSize: 13, fontWeight: 600 }}>{status.name || status.run_id}</span>
                    {status.auto ? <Muted>全自动</Muted> : null}
                    {status.best_step ? <Muted>最佳第 {status.best_step} 步（val loss {status.best_val_loss != null ? Number(status.best_val_loss).toFixed(4) : "—"}）</Muted> : null}
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 10 }}>
                    <Bar pct={prog.pct} tone={runPhase === "failed" ? "#d64545" : undefined} />
                    <span style={{ fontSize: 13, fontWeight: 600, color: BRAND_DEEP }}>{prog.pct}%</span>
                  </div>
                  <div style={{ fontSize: 13, marginTop: 6 }}>
                    第 {prog.step} 步 / {prog.target ?? status.steps_total ?? "—"} · 约剩 {fmtEta((status.progress || {}).eta_s)}
                    {prog.bps ? " · " + prog.bps : ""}
                  </div>
                  {(status.progress || {}).stage_text ? <Muted>{(status.progress || {}).stage_text}</Muted> : null}
                  {status.error ? <Err>{status.error}</Err> : null}
                </div>

                {(() => {
                  const adv = status.advice || {}
                  if (!adv.text && !adv.keep_text) return null
                  return (
                    <Alert tone={["stop_overfit", "stop_abnormal", "stop_flt", "overfit", "abnormal"].indexOf(String(adv.code || "")) >= 0 ? "warning" : "info"}>
                      {(status.auto ? "自动判定：" : "系统建议：") + (adv.text || "")}{adv.keep_text ? "　" + adv.keep_text : ""}
                    </Alert>
                  )
                })()}

                <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                  <Btn style={PRIMARY} disabled={!!busy || ended} onClick={continueTrain}>{busy === "continue" ? "下发中…" : "继续训练"}</Btn>
                  <Btn disabled={!!busy || ended} onClick={() => control("stop")}>{busy === "stop" ? "下发中…" : "停止训练"}</Btn>
                  <Muted>停止后状态会变成「已到节点」，再到「保存模型」挑一步。</Muted>
                </div>

                {/* 试听节点 */}
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>
                    试听节点 <Muted>每 2000 步自动生成一段；到点会弹试听卡片</Muted>
                  </div>
                  {nodes.length === 0 ? <Empty>训练到第一个节点（2000 步）后，这里会出现试听卡片。</Empty> : (
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(230px, 1fr))", gap: 10 }}>
                      {nodes.map((n: any) => {
                        const f = flagText(n.flag)
                        const chosen = pick === n.step
                        return (
                          <div key={n.step} style={{
                            borderRadius: 12, padding: "10px 12px", display: "flex", flexDirection: "column", gap: 6,
                            background: chosen ? "#f2f6ff" : "#f7f9fd",
                          }}>
                            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 6 }}>
                              <div style={{ fontSize: 14, fontWeight: 600 }}>第 {n.step} 步</div>
                              <StatusBadge tone={n.ready ? f.tone : n.missed ? "warning" : "neutral"}>
                                {n.ready ? f.text : n.missed ? "未生成" : "待生成"}
                              </StatusBadge>
                            </div>
                            <Muted>val loss {n.val_loss == null ? "未跑" : n.val_loss.toFixed(4)} · 时长 {n.seconds ?? "—"}s</Muted>
                            <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
                              <Btn style={PRIM_TXT} disabled={!n.ready || !!busy} onClick={() => openNode(n.step)}>打开试听 →</Btn>
                              {chosen ? <Muted>已选</Muted> : null}
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  )}
                </div>

                <Fold title="高级网络设置（本次训练参数）">
                  {() => {
                    const d = status.dataset || status.data || {}
                    const p = status.params || {}; const why = p.reasons || {}
                    const rows: any[] = [
                      ["批大小", p.batch_size, why.batch_size],
                      ["片段时长", p.duration != null ? p.duration + "s" : null, why.duration],
                      ["混合精度", p.amp, why.amp],
                      ["音高增强", p.use_pitch_aug == null ? null : (p.use_pitch_aug ? "开" : "关"), why.use_pitch_aug],
                      ["权重衰减", p.weight_decay, why.weight_decay],
                      ["验证占比", p.val_ratio != null ? Math.round(Number(p.val_ratio) * 100) + "%" : null, why.val_split],
                      ["步数上限", p.auto_cap, why.cap_reason],
                      ["起点", p.init_from, why.init_from],
                    ].filter((r) => r[1] !== null && r[1] !== undefined && r[1] !== "").map((r) => [r[0], String(r[1]) + (r[2] ? "（" + r[2] + "）" : "")])
                    return (
                      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                        <Muted>训练片段 {d.clips_train ?? "—"} 条 · 验证片段 {d.clips_val ?? "—"} 条 · 素材总时长 {(d.total_sec ? Number(d.total_sec).toFixed(1) + "s" : "—")}{(d.dropped && d.dropped.length) ? " · 丢弃 " + d.dropped.length + " 段" : ""}</Muted>
                        {rows.length ? <KV items={rows} /> : null}
                      </div>
                    )
                  }}
                </Fold>
              </div>
            )
          ) : (
            <div style={{ marginTop: 12 }}>
              <Empty>还没有训练任务。回到「训练设置」点「开始训练」。</Empty>
            </div>
          )}

          <Foot prev={2} next={4} nextMain nextText="下一步：保存模型" onGo={goto}
                hint={!keepCkpt ? <Muted>训练中也可以先上去看</Muted> : undefined} />
        </div>
      ) : null}

      {/* ===== Step 4：保存模型 ===== */}
      {step === 4 ? (
        <div style={CARD}>
          <div style={{ fontSize: 14.5, fontWeight: 600 }}>保存模型</div>
          <div style={{ fontSize: 12.5, color: INK_2, margin: "4px 0 14px" }}>
            挑一步保留成正式模型；「保留这一步」只存文件，不会改你正在用的模型。
          </div>

          {nodes.length === 0 ? <Empty>训练到第一个节点（2000 步）后，这里会出现可选节点。</Empty> : (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {nodes.map((n: any) => {
                const f = flagText(n.flag)
                const chosen = pick === n.step
                return (
                  <div key={n.step} onClick={() => { if (n.ready) setPick(n.step) }} style={{
                    display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", padding: "10px 12px", borderRadius: 10,
                    background: chosen ? "#f2f6ff" : "#f7f9fd", cursor: n.ready ? "pointer" : "default",
                  }}>
                    <span style={{ width: 16, color: BRAND }}>{chosen ? "●" : ""}</span>
                    <span style={{ fontSize: 14, fontWeight: 600, minWidth: 86 }}>第 {n.step} 步</span>
                    <StatusBadge tone={n.ready ? f.tone : n.missed ? "warning" : "neutral"}>
                      {n.ready ? f.text : n.missed ? "未生成" : "待生成"}
                    </StatusBadge>
                    <Muted>val loss {n.val_loss == null ? "未跑" : n.val_loss.toFixed(4)}</Muted>
                    {n.ready ? <Btn style={PRIM_TXT} onClick={(e: any) => { if (e && e.stopPropagation) e.stopPropagation(); openNode(n.step) }}>试听</Btn> : null}
                  </div>
                )
              })}
            </div>
          )}

          <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap", marginTop: 16 }}>
            <Muted>已选：{pick ? "第 " + pick + " 步" : "未选"}</Muted>
            <Input placeholder="正式模型名（可留空）" value={keepName} style={{ width: 200 }} onChange={(e: any) => setKeepName(e.target.value)} />
            <Btn style={PRIMARY} disabled={!pick || !!busy || !canKeep(status)} onClick={() => keep()}>{busy === "keep" ? "保存中…" : "保留这一步"}</Btn>
            <Btn disabled={!keepCkpt || !!busy} onClick={setCurrent}>{busy === "current" ? "切换中…" : "设为当前模型"}</Btn>
          </div>
          <div style={{ fontSize: 11.5, color: "#9aa6bb", marginTop: 8 }}>
            「设为当前模型」是单独的一步，点了才生效；其余检查点会移进回收站（不永久删除）。
          </div>
          {keepMsg ? <div style={{ marginTop: 6 }}><Muted>{keepMsg}</Muted></div> : null}
          {keepCkpt ? <div style={{ marginTop: 4 }}><Muted style={{ wordBreak: "break-all" }}>模型：{keepCkpt}</Muted></div> : null}
          {!canKeep(status) && pick ? <div style={{ marginTop: 4 }}><Muted style={{ color: "#c8861a" }}>训练还没停：先到「训练监控」点「停止训练」再保留。</Muted></div> : null}

          <Foot prev={3} onGo={goto} />
        </div>
      ) : null}

      {/* Step 3 里还没有任务时，把「开始训练」放在这里（确认框仍在下面统一处理） */}
      {showStart ? (
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <Btn style={PRIMARY} disabled={!canStart} onClick={() => setConfirmOpen(true)}>{busy === "start" ? "正在启动…" : "开始训练"}</Btn>
          {gap ? <Muted style={{ color: "#c8861a" }}>时长不足：{gap}</Muted> : null}
        </div>
      ) : null}

      {/* 主按钮：训练设置里开始训练（先弹确认） */}
      {step === 2 ? (
        <div style={{ display: "flex", justifyContent: "flex-end", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          {gap ? <Muted style={{ color: "#c8861a" }}>时长不足：{gap}</Muted> : null}
          <Btn style={PRIMARY} disabled={!canStart} onClick={() => setConfirmOpen(true)}>{busy === "start" ? "正在启动…" : "开始训练"}</Btn>
        </div>
      ) : null}

      {/* 确认框：极简，只看预计耗时和素材时长 */}
      <Confirm
        open={confirmOpen}
        title="开始训练？"
        okText={busy === "start" ? "正在启动…" : "确认开始"}
        cancelText="再想想"
        onCancel={() => setConfirmOpen(false)}
        onOk={() => {
          if (busy) return
          setConfirmOpen(false)
          // startTrain 自带时长校验；这里不再重复判断，避免两处口径不一致
          startTrain()
        }}
      >
        <KV items={[
          ["步数", steps + " 步（共 " + Math.round(steps / 2000) + " 个试听节点）"],
          ["素材时长", fmtTime(have) + " · 最少 " + fmtTime(need.min)],
          ["预计耗时", fmtEta((status && (status.progress || {}).eta_s) || (hw && hw.eta_4000_s ? Number(hw.eta_4000_s) * steps / 4000 : NaN))],
          ["方式", auto ? "全自动（不停下）" : "到节点停下等试听"],
        ]} />
        <div style={{ fontSize: 12, color: "#8a97ad", marginTop: 10 }}>
          预计耗时按本机档位估算，实际以训练进度为准；期间可以随时停止。
        </div>
      </Confirm>

      {/* 弹窗层：交给 ui.tsx 的 DialogLayer 排队，同一时间只会出现一张。
          （原来用 z-index 10/11 叠两张，2026-11-04 方案第 4 节要求去掉。） */}
      {/* 试听卡片：到点弹出，也能从节点列表/保存模型里打开 */}
      <Confirm
        open={nodeOpen && !!nodeCard}
        title={nodeCard ? "第 " + nodeCard.step + " 步试听" : "节点试听"}
        okText={busy === "keep" ? "保存中…" : "保留这一步"}
        // 还没到节点时不能直接继续：关掉卡片即可，否则下发的「继续」也没意义
        cancelText={canKeep(status) ? "先关掉" : "继续训练"}
        onOk={keepFromCard}
        onCancel={canKeep(status) ? () => setNodeOpen(false) : continueTrain}
      >
        {nodeCard ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
              <StatusBadge tone={flagText(nodeCard.flag).tone}>{flagText(nodeCard.flag).text}</StatusBadge>
              <Muted>val loss {nodeCard.val_loss == null ? "未跑" : nodeCard.val_loss.toFixed(4)} · 时长 {nodeCard.seconds ?? "—"}s</Muted>
            </div>
            {previews[nodeCard.step]
              ? <AudioBox src={previews[nodeCard.step]} keyId={"card-" + nodeCard.step} />
              : <Muted>正在取试听…</Muted>}
            <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
              <Input placeholder="正式模型名（可留空）" value={keepName} style={{ flex: 1, minWidth: 140 }} onChange={(e: any) => setKeepName(e.target.value)} />
              <Btn style={PRIM_TXT} onClick={() => playPreview(nodeCard.step)}>重新试听</Btn>
            </div>
            {keepMsg && !canKeep(status) ? <Muted>{keepMsg}</Muted> : null}
            {!canKeep(status) ? (
              <Muted style={{ color: "#c8861a" }}>训练还在跑：可以先把试听听完、继续训练，停下后再回来保留。</Muted>
            ) : null}
          </div>
        ) : <Muted>这个节点还没有试听。</Muted>}
      </Confirm>

      {ov ? (
        <Confirm
          open={true}
          title="检测到过拟合，是否继续训练？"
          okText="继续训练"
          cancelText="停止并挑节点"
          onCancel={() => { setOverfitSeen(ovKey); control("stop") }}
          onOk={() => { setOverfitSeen(ovKey); control("continue") }}
        >
          第 {ov.step} 步的 val loss 为 {ov.val_loss != null ? Number(ov.val_loss).toFixed(4) : "—"}，
          最佳是第 {ov.best_step ?? "—"} 步（{ov.best_val != null ? Number(ov.best_val).toFixed(4) : "—"}）。
          继续大概率只会更像但更失真；停止后建议保留最佳那一步。
        </Confirm>
      ) : null}

      {/* 底部一行浅灰：本机环境（没有就整行不出现） */}
      {hw ? (
        <div style={{ fontSize: 11, color: "#9aa6bb", textAlign: "center", paddingBottom: 4 }}
             title={"档位 " + hw.tier + " · 训练 batch " + ((hw.train || {}).batch_size ?? "?") + " · " + ((hw.train || {}).amp ?? "?")}>
          {hw.gpu ? hw.gpu.name + " " + Math.round(Number(hw.gpu.vram_mb || 0) / 1024) + "GB" : "CPU 训练"}
          {" · " + hw.cpu_threads + " 线程 · " + hw.ram_gb + "GB 内存"}
        </div>
      ) : null}
    </div>
  )
}
