// 为她而唱 · 首次使用向导（2026-11-04，方案第 3 节）
//
// 打开插件先调 ms_setup_check；有任何必需项没就绪，就显示这个全屏向导挡住其它页面。
//   第 1 步 欢迎     —— 要下载什么、多大、多久
//   第 2 步 安装位置 —— 预填建议目录 + 该盘剩余空间；放不下就标红、下一步禁用
//   第 3 步 安装     —— 每项一行状态图标 + 进度条，顶部总进度 + 当前在做什么（每 1 秒轮询）
//   第 4 步 完成     —— 「全部就绪」；失败时显示原因 + 「重试这一项」
// 只用既有接口：ms_setup_check / ms_setup_install / ms_setup_status。安装任务在后端线程里跑，
// 所以关掉向导不会中断安装；再打开时会从 ms_setup_status 接着显示。
// 模拟演示：预览环境（?bundle=tmp 或后端不通）下自动用一份示例清单走完 4 步，页面标「模拟演示」。
import { Alert, StatusBadge, useEffect, useRef, useState } from "@neko/plugin-ui"
import { callEntry } from "./api"
import { BRAND, Btn, INK, INK_2, INK_3, LINE, Muted, Panel, Pill, Row, Bar, fmtEta } from "./ui"

type Item = { key: string; label: string; status: string; ok: boolean; required: boolean; installable: boolean; size_gb: number; detail: string; fix: string }
type Report = { ready: boolean; items: Item[]; suggested_root: string; total_missing_gb: number; [k: string]: any }

const STEPS = ["欢迎", "安装位置", "安装", "完成"]

/** 只有开发预览（?wizard= / ?bundle=）才允许用示例清单，正式使用必须读真实检测结果。 */
function previewMode(): boolean {
  try {
    const q = String((window as any).location && (window as any).location.search || "")
    return /[?&](wizard|bundle)=/.test(q)
  } catch (e) { return false }
}

// ---- 模拟：preview 里没有安装接口时给一份示例清单，只为验收界面 ------------
function mockReport(root: string, ready: boolean): Report {
  const items: Item[] = [
    { key: "base_python", label: "基础 Python（建虚拟环境用）", status: "ready", ok: true, required: true, installable: false, size_gb: 0, detail: "3.10.11 · 系统已装", fix: "" },
    { key: "synth_env", label: "DDSP-SVC 环境", status: "ready", ok: true, required: true, installable: false, size_gb: 0, detail: "已就绪", fix: "" },
    { key: "sep_env", label: "人声分离环境", status: "ready", ok: true, required: true, installable: false, size_gb: 0, detail: "已就绪", fix: "" },
    { key: "ddsp_repo", label: "人声模型程序", status: "ready", ok: true, required: true, installable: false, size_gb: 0, detail: "已就绪", fix: "" },
    { key: "ddsp_contentvec", label: "声音特征模型（ContentVec）", status: ready ? "ready" : "missing", ok: ready, required: true, installable: !ready, size_gb: 0.2, detail: ready ? "已就绪" : "缺少 1/1 个文件", fix: "自动下载，约 0.2GB" },
    { key: "ddsp_vocoder", label: "音色合成模型（NSF-HiFiGAN）", status: ready ? "ready" : "missing", ok: ready, required: true, installable: !ready, size_gb: 0.05, detail: ready ? "已就绪" : "缺少 1/1 个文件", fix: "自动下载，约 0.05GB" },
    { key: "ddsp_rmvpe", label: "音高识别模型（RMVPE）", status: ready ? "ready" : "missing", ok: ready, required: true, installable: !ready, size_gb: 0.06, detail: ready ? "已就绪" : "缺少 1/1 个文件", fix: "自动下载，约 0.06GB" },
    { key: "sep_karaoke", label: "去伴奏模型（Karaoke）", status: ready ? "ready" : "missing", ok: ready, required: true, installable: !ready, size_gb: 1.75, detail: ready ? "已就绪" : "缺少 1/1 个文件", fix: "自动下载，约 1.75GB" },
  ]
  const total = items.filter((i) => !i.ok).reduce((a, i) => a + i.size_gb, 0)
  return { ready, items, suggested_root: root, total_missing_gb: Math.round(total * 10) / 10,
           mock: true, hardware_brief: "显卡：NVIDIA GeForce RTX 3060（模拟）" }
}

// 模拟安装进度：按清单顺序逐项跑，每项大约 6 秒
function mockProgress(report: Report, t: number) {
  const todo = report.items.filter((i) => i.installable && i.status !== "ready")
  const items: any = {}
  for (const i of report.items) items[i.key] = { label: i.label, state: i.status === "ready" ? "ready" : "wait", pct: null }
  const per = 6
  const idx = Math.floor(t / per)
  for (let k = 0; k < Math.min(idx, todo.length); k++) items[todo[k].key] = { ...items[todo[k].key], state: "done", pct: 100 }
  if (idx < todo.length) {
    const cur = todo[idx]
    const p = Math.min(96, Math.round(((t % per) / per) * 96))
    items[cur.key] = { ...items[cur.key], state: "active", pct: p,
                       done_mb: Math.round(cur.size_gb * 1024 * p / 100), total_mb: Math.round(cur.size_gb * 1024) }
  }
  const doneAll = idx >= todo.length
  return {
    items, current: doneAll ? "" : todo[idx].key,
    job_state: doneAll ? "done" : "running",
    detail: doneAll ? "安装完成" : "下载 " + todo[idx].label + "：" + Math.round((t % per) / per * (todo[idx].size_gb * 1024)) + "/" + Math.round(todo[idx].size_gb * 1024) + " MB",
  }
}

const STATE_TEXT: { [k: string]: { text: string; tone: any } } = {
  ready: { text: "已就绪", tone: "success" },
  wait: { text: "等待", tone: "neutral" },
  active: { text: "安装中", tone: "warning" },
  done: { text: "完成", tone: "success" },
  failed: { text: "失败", tone: "error" },
}

function StateIcon(props: { state: string }) {
  const st = props.state || "wait"
  const color = st === "ready" || st === "done" ? "#1c8a58" : st === "failed" ? "#c93c3c" : st === "active" ? BRAND : "#b9b3c1"
  const bg = st === "ready" || st === "done" ? "#e8f7ef" : st === "failed" ? "#fdecec" : st === "active" ? "#ffe5ef" : "rgba(40,37,50,0.06)"
  return (
    <span aria-hidden style={{ width: 22, height: 22, borderRadius: 999, background: bg, color, display: "inline-grid", placeItems: "center", flex: "0 0 auto" }}>
      {st === "ready" || st === "done" ? (
        <svg width="12" height="12" viewBox="0 0 12 12"><path d="M2 6.4 4.8 9 10 3.2" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
      ) : st === "failed" ? <b style={{ fontSize: 12, lineHeight: 1 }}>!</b>
        : st === "active" ? <i style={{ width: 8, height: 8, borderRadius: 999, background: color, display: "block" }} />
          : <i style={{ width: 6, height: 6, borderRadius: 999, background: color, display: "block" }} />}
    </span>
  )
}

const primaryBtn = { border: 0, background: BRAND, color: "#fff", borderRadius: 12, boxShadow: "0 6px 16px rgba(224,45,118,0.22)" }

export function SetupWizard(props: { api: any; onDone: () => void; startAt?: number }) {
  const [step, setStep] = useState<number>(props.startAt || 1)
  const [report, setReport] = useState<Report | null>(null)
  const [mockMode, setMockMode] = useState(false)
  const [path, setPath] = useState("")
  const [freeGb, setFreeGb] = useState<number | null>(null)
  const [busy, setBusy] = useState("")
  const [err, setErr] = useState("")
  const [job, setJob] = useState<any>(null)
  const [prog, setProg] = useState<any>(null)
  const [mockT, setMockT] = useState(0)
  const jobId = useRef("")

  // 第 1 步：检测
  const check = async () => {
    setBusy("check")
    setErr("")
    try {
      const r: any = await callEntry(props.api, "ms_setup_check", {}, 180000)
      if (!r || !Array.isArray(r.items)) throw Error("没有拿到清单")
      setReport(r)
      setMockMode(false)
      setPath(String(r.suggested_root || ""))
    } catch (e: any) {
      if (previewMode()) {
        setReport(mockReport("", false))
        setMockMode(true)
        setPath("")
      } else {
        setReport(null)
        setErr("检测失败：" + String((e && e.message) || e))
      }
    }
    setBusy("")
  }
  useEffect(() => { check() }, [])

  // 第 2 步：安装位置剩余空间（按安装根目录所在盘）
  // 数据来源：ms_setup_check 的 report.hardware.drives（探测时已经列过每块盘的剩余空间），
  // 没有再退回宿主提供的 window.__msDiskFree；都没有就只提示「读不到」，不拦着继续。
  useEffect(() => {
    if (step !== 2 || !path) return
    const drive = (path.match(/^[A-Za-z]:/) || [""])[0].toUpperCase()
    if (!drive) { setFreeGb(null); return }
    const drives = (report && report.hardware && report.hardware.drives) || []
    const hit = drives.filter((d: any) => String(d.drive || "").toUpperCase() === drive)[0]
    if (hit && hit.free_gib !== undefined) { setFreeGb(Number(hit.free_gib)); return }
    const s: any = (window as any).__msDiskFree
    if (typeof s === "function") {
      Promise.resolve(s(drive)).then((g: any) => setFreeGb(Number(g))).catch(() => setFreeGb(null))
    } else setFreeGb(null)
  }, [step, path, report])

  // 第 3 步：每 1 秒轮询
  useEffect(() => {
    if (step !== 3) return
    let stop = false
    let tick = 0
    const loop = async () => {
      while (!stop) {
        await new Promise((r) => setTimeout(r, 1000))
        if (stop) break
        tick += 1
        if (mockMode || !jobId.current) {
          setMockT(tick)
          const p = mockProgress(report || mockReport(path, false), tick)
          setProg(p)
          if (p.job_state === "done") { setStep(4); return }
          continue
        }
        try {
          const st: any = await callEntry(props.api, "ms_setup_status", {}, 30000)
          setJob(st.job || {})
          setProg(st.progress || null)
          if (st.setup && st.setup.report) setReport(st.setup.report)
          const s = String((st.job || {}).state || "")
          if (s === "done" || s === "failed") { setStep(4); return }
        } catch (e: any) { /* 单次失败不影响，下一秒再试 */ }
      }
    }
    loop()
    return () => { stop = true }
  }, [step, mockMode])

  const startInstall = async (only?: string) => {
    setBusy("install")
    setErr("")
    try {
      const r: any = await callEntry(props.api, "ms_setup_install", only ? { path: path.trim(), items: [only] } : { path: path.trim() }, 60000)
      if (r && r.already_ready) {
        setProg({ job_state: "done", items: {}, detail: "环境已就绪" })
        setStep(4)
        return
      }
      if (r && r.job_id) {
        jobId.current = String(r.job_id)
        setJob({ state: "running", progress: 0, detail: "正在准备…" })
        setMockT(0)
      } else throw Error(String((r && r.error) || "没能启动安装"))
      setStep(3)
    } catch (e: any) {
      if (mockMode) { setMockT(0); setStep(3); return }
      setErr(String((e && e.message) || e))
    }
    setBusy("")
  }

  const items = (report && report.items) || []
  const todo = items.filter((i) => i.installable && i.status !== "ready")
  const needGb = report ? Number(report.total_missing_gb || 0) : 0
  const lack = freeGb !== null && needGb > 0 && freeGb < Math.max(0.5, needGb * 1.15)

  // ---- 各步内容 --------------------------------------------------------
  const welcome = (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ fontSize: 13.5, lineHeight: 1.85, color: INK }}>
        欢迎使用「为她而唱」。第一次打开需要先准备好她会用到的程序、模型和音频处理工具，之后就不用再装了。
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(150px,1fr))", gap: 12 }}>
        {[
          ["要下载什么", "运行程序 + 声音模型 + 去伴奏模型"],
          ["大约多大", report && needGb ? needGb + " GB" : "约 2 GB（以检测结果为准）"],
          ["大概多久", report && needGb ? fmtEta(needGb * 900) : "看网速，一般 10～40 分钟"],
        ].map((row: string[], i: number) => (
          <div key={i} style={{ padding: 14, borderRadius: 16, background: "rgba(255,229,239,0.5)", border: "1px solid rgba(224,45,118,0.10)" }}>
            <div style={{ fontSize: 11.5, color: INK_2 }}>{row[0]}</div>
            <div style={{ fontSize: 14, fontWeight: 700, color: "#9c204e", marginTop: 4 }}>{row[1]}</div>
          </div>
        ))}
      </div>
      <Muted>安装过程中可以继续用电脑；关掉这个窗口不会中断下载，下次打开会接着装。</Muted>
      <Row style={{ justifyContent: "flex-end" }}>
        <Btn style={primaryBtn} disabled={busy === "check"} onClick={() => setStep(2)}>{busy === "check" ? "检测中…" : "下一步：选安装位置"}</Btn>
      </Row>
    </div>
  )

  const place = (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div style={{ fontSize: 13.5, color: INK, lineHeight: 1.8 }}>
        程序、模型和临时文件都会放在这里。选一个空间充足的盘就行，装在 D 盘、E 盘都可以。
      </div>
      <label style={{ display: "flex", flexDirection: "column", gap: 6, fontSize: 12, color: INK_2 }}>
        <span>安装到哪里</span>
        <input value={path} onChange={(e: any) => setPath(e.target.value)} style={{
          border: "1px solid " + (lack ? "#e8a3a3" : "transparent"), borderRadius: 12, padding: "10px 14px", fontSize: 13,
          background: lack ? "#fff6f6" : "rgba(248,250,252,0.95)", color: INK, outline: "none",
        }} />
      </label>
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
        {freeGb !== null ? (
          lack
            ? <StatusBadge tone="error">这个盘只剩 {freeGb.toFixed(1)} GB，还要约 {needGb} GB，换一个盘吧</StatusBadge>
            : <StatusBadge tone="success">这个盘还剩 {freeGb.toFixed(1)} GB · 需要约 {needGb} GB</StatusBadge>
        ) : <Muted>（无法读取剩余空间，也可以继续）</Muted>}
      </div>
      {mockMode ? <Muted>安装向导暂时读不到检测结果。</Muted> : null}
      <Row style={{ justifyContent: "space-between" }}>
        <Btn onClick={() => setStep(1)}>上一步</Btn>
        <Btn style={primaryBtn} disabled={lack || !path.trim() || busy === "install"} onClick={() => startInstall()}>
          {busy === "install" ? "正在启动…" : "开始安装"}
        </Btn>
      </Row>
      {err ? <Alert tone="warning">{err}</Alert> : null}
    </div>
  )

  // 第 3 步：逐项列表
  const rows = items.filter((i) => i.status !== "ready" || (prog && prog.items && prog.items[i.key]))
  const showing = rows.length ? rows : items
  const overall = (() => {
    if (!prog) return 0
    const list = items.filter((i) => i.status !== "ready")
    if (!list.length) return 100
    let sum = 0
    for (const i of list) {
      const st = (prog.items || {})[i.key] || {}
      sum += st.state === "done" || st.state === "ready" ? 100 : Number(st.pct || 0)
    }
    return Math.round(sum / list.length)
  })()
  const installing = (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        <Row style={{ justifyContent: "space-between" }}>
          <div style={{ fontSize: 13.5, fontWeight: 700, color: INK }}>{prog && prog.job_state === "done" ? "快好了…" : "正在下载并安装"}</div>
          <Muted>{overall}%</Muted>
        </Row>
        <Bar pct={overall} />
        <div style={{ fontSize: 12, color: INK_2, minHeight: 18 }}>{(prog && prog.detail) || (job && job.detail) || "正在准备…"}</div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {showing.map((i) => {
          const st = (prog && prog.items && prog.items[i.key]) || { state: i.status === "ready" ? "ready" : "wait", pct: null }
          const tx = STATE_TEXT[st.state] || STATE_TEXT.wait
          const isActive = st.state === "active"
          const isDownload = st.total_mb && st.done_mb !== undefined && st.done_mb !== null
          return (
            <div key={i.key} style={{ display: "grid", gridTemplateColumns: "22px minmax(0,1fr) auto", gap: "4px 10px", alignItems: "center", padding: "9px 12px", borderRadius: 14, background: isActive ? "rgba(255,229,239,0.55)" : "rgba(248,250,252,0.9)", border: "1px solid " + (isActive ? "rgba(224,45,118,0.18)" : LINE) }}>
              <StateIcon state={st.state} />
              <div style={{ minWidth: 0 }}>
                <div style={{ fontSize: 12.5, color: INK, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{i.label}</div>
                <div style={{ fontSize: 11.5, color: INK_3 }}>
                  {isDownload ? st.done_mb + " / " + st.total_mb + " MB" : i.size_gb ? "约 " + i.size_gb + " GB" : ""}
                </div>
              </div>
              <Pill tone={tx.tone}>{tx.text}</Pill>
              <div style={{ gridColumn: "2 / 4" }}>
                <Bar pct={st.state === "done" || st.state === "ready" ? 100 : Number(st.pct || 0)} tone={st.state === "failed" ? "#e06666" : st.state === "active" ? undefined : "rgba(224,45,118,0.35)"} />
              </div>
            </div>
          )
        })}
      </div>
      <Muted>可以先去忙别的：关掉窗口不会中断，回来还能接着看。</Muted>
    </div>
  )

  // 第 4 步：完成 / 失败
  const failed = prog && prog.job_state === "failed"
  const failLabels = items.filter((i) => {
    const st = prog && prog.items && prog.items[i.key]
    return st && st.state === "failed"
  })
  const finish = (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {failed ? (
        <>
          <Alert tone="warning">{(prog && prog.detail) || "安装没能全部完成，看看下面哪一项出了问题。"}</Alert>
          {failLabels.length ? failLabels.map((i) => (
            <div key={i.key} style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "center", padding: "10px 12px", borderRadius: 14, background: "#fdecec" }}>
              <div style={{ minWidth: 0 }}>
                <div style={{ fontSize: 12.5, color: INK }}>{i.label}</div>
                <div style={{ fontSize: 11.5, color: "#c93c3c", wordBreak: "break-all" }}>{((prog.items[i.key] || {}).error) || i.fix}</div>
              </div>
              <Btn onClick={() => startInstall(i.key)}>重试这一项</Btn>
            </div>
          )) : <Btn onClick={() => startInstall()}>重新安装缺失项</Btn>}
        </>
      ) : (
        <>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <StateIcon state="done" />
            <div style={{ fontSize: 17, fontWeight: 700, color: INK }}>全部就绪</div>
          </div>
          <div style={{ fontSize: 13, color: INK_2, lineHeight: 1.8 }}>以后打开插件不用再装东西了。先把几首歌放进「准备素材」，就能开始训练她的声音。</div>
          <Row style={{ justifyContent: "flex-end" }}>
            <Btn style={primaryBtn} onClick={props.onDone}>开始使用</Btn>
          </Row>
        </>
      )}
    </div>
  )

  const body = step === 1 ? welcome : step === 2 ? place : step === 3 ? installing : finish

  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 1100, display: "flex", alignItems: "center", justifyContent: "center", padding: 20, background: "rgba(248,250,252,0.92)", backdropFilter: "blur(10px)", WebkitBackdropFilter: "blur(10px)", overflow: "auto" }}>
      <div style={{ width: "100%", maxWidth: 620 }}>
        <Panel style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          <div>
            <div style={{ fontSize: 18, fontWeight: 700, color: INK }}>首次使用 · 安装向导</div>
            <div style={{ fontSize: 12, color: INK_2, marginTop: 3 }}>第 {step} 步 / 共 4 步 · {STEPS[step - 1]}</div>
          </div>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {STEPS.map((t: string, i: number) => {
              const no = i + 1
              const on = step === no
              const done = step > no
              return (
                <span key={no} style={{ display: "inline-flex", alignItems: "center", gap: 6, padding: "5px 11px", borderRadius: 999, fontSize: 12.5, background: on ? BRAND : "transparent", color: on ? "#fff" : done ? "#9c204e" : INK_2, fontWeight: on ? 700 : 400 }}>
                  <i style={{ fontStyle: "normal", width: 17, height: 17, lineHeight: "17px", textAlign: "center", borderRadius: 999, fontSize: 10.5, background: on ? "#fff" : done ? "rgba(224,45,118,0.14)" : "rgba(40,37,50,0.06)", color: on ? BRAND : "inherit" }}>{done ? "✓" : no}</i>
                  {t}
                </span>
              )
            })}
          </div>
          <div>{body}</div>
        </Panel>
        <div style={{ fontSize: 11, color: INK_3, textAlign: "center", marginTop: 10 }}>安装文件只来自官方发布地址，放在你选定的目录里；不会改动系统设置。</div>
      </div>
    </div>
  )
}
