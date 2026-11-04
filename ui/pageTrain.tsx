// 为她而唱 · 训练声音（2026-11-04，方案第 2、5 节）
//
// 四个小步：① 放素材 → ② 训练设置 → ③ 看着她唱 → ④ 保存声音。
// 素材三张卡和「准备素材」页共用 materials.ts 的单例（同一批文件两边都能看到）；
// 训练接口和原来 train.tsx 完全一致：ms_corpus_* / ms_train_start / ms_train_status /
// ms_train_control(continue|stop) / ms_train_preview / ms_train_keep(run_id, step, set_current)。
// 界面上不出现 步数 / ckpt / run_id 这些字段名：只说「快一些 / 推荐 / 细致一些」「保存这个声音」。
import { Alert, StatusBadge, useEffect, useRef, useState } from "@neko/plugin-ui"
import { audioSrcOf, callEntry, dataUriOf } from "./api"
import { MATERIALS, useCorpus } from "./materials"
import {
  STEPS_CHOICES, STEPS_MAX, STEPS_MIN, STEPS_PER_SECOND, STEPS_MODES, STEPS_HARD_MAX,
  clipList, deficitText, effectiveSeconds, flagText, materialTitle, milestoneList, num,
  overfitOf, phaseText, progressText, requirementOf, slotView, totalFileSeconds,
} from "./trainModel"
import {
  BRAND, BRAND_DEEP, INK, INK_2, LINE, AudioBox, Bar, Btn, Col, Empty, Err, FileRow, FoldCard,
  Grid, Input, Muted, Panel, PRIMARY_BTN, Pill, Row, Section, Select, Soft, Toggle, fmtEta, fmtTime, DeletePrompt,
} from "./ui"

const STEP_NAMES = ["放素材", "训练设置", "看着她唱", "保存声音"]
const ENDED = ["finished", "failed", "stopped"]
// 只有这些阶段才在真正消耗显卡；awaiting_choice 只是进程在等你试听
const ACTIVE_PHASES = ["training", "preparing", "preprocessing", "previewing"]

/** 训练页的「上次看到哪」：模块级，跨页面切换不丢 */
const TRAIN_UI = { step: 1, maxStep: 1, runId: "", name: "" }

export function PageTrain(props: { api: any; snap: any; onRefresh: () => void; onGo: (page: string) => void }) {
  const { api, snap } = props
  const c = useCorpus(api)

  // 切到别的页面时这个组件会被卸载，useState 里的东西全没了。
  // 把「正在第几步、在看哪个训练」放到模块级：切回来自动回到原来的位置；
  // 连插件关掉再打开（模块重载、这里归零）时，下面会从后端的 active 训练自动接回来。
  const [step, _setStep] = useState<number>(TRAIN_UI.step)
  const setStep = (v: number) => {
    TRAIN_UI.step = v
    if (v > TRAIN_UI.maxStep) TRAIN_UI.maxStep = v   // 记住走到过的最远一步，退回去还能再点回来
    _setStep(v)
  }
  const [name, _setName] = useState<string>(TRAIN_UI.name)
  const setName = (v: string) => { TRAIN_UI.name = v; _setName(v) }
  const [extraSteps, setExtraSteps] = useState<string>("")
  const [stepMode, setStepMode] = useState<string>("normal")
  const [customSteps, setCustomSteps] = useState<string>("10000")
  const [customOpen, setCustomOpen] = useState<boolean>(false)
  const [extraOpen, setExtraOpen] = useState<boolean>(false)
  const [openAdv, setOpenAdv] = useState<boolean>(false)
  const [msg, setMsg] = useState<any>(null)
  const [busy, setBusy] = useState<string>("")

  const [runs, setRuns] = useState<any>(null)
  const [runId, _setRunId] = useState<string>(TRAIN_UI.runId)
  const setRunId = (v: string) => { TRAIN_UI.runId = v; _setRunId(v) }
  const [status, setStatus] = useState<any>(null)
  const [previews, setPreviews] = useState<any>({})
  const [pickStep, setPickStep] = useState<number>(0)
  const [keepName, setKeepName] = useState<string>("")
  const [keepMsg, setKeepMsg] = useState<string>("")
  const [delRun, setDelRun] = useState<boolean>(false)

  const [nodeOpen, setNodeOpen] = useState<boolean>(false)
  const [overOpen, setOverOpen] = useState<boolean>(false)
  const nodeSeen = useRef<string>("")
  const overSeen = useRef<string>("")

  const list = c.list
  const have = effectiveSeconds(list, c.countSpeech)
  // 训练步数跟每个素材文件的实际时长走（不是文件个数，也不是处理后的切片）
  const filesHave = totalFileSeconds(list)
  const fileCount = slotView(list, "song").files + slotView(list, "dry_sing").files + slotView(list, "speech").files
  // 三档 + 自定义：步数是参数更新次数（已计入梯度累积）。
  const steps = stepMode === "custom"
    ? Math.max(STEPS_MIN, Math.min(STEPS_HARD_MAX, Number(customSteps) || STEPS_MIN))
    : (STEPS_MODES.find((m: any) => m.key === stepMode) || STEPS_MODES[1]).steps
  // 能不能开始训练仍按 8000 步这一档判：至少 3 分钟、推荐 5 分钟（和「开始」页一致）。
  const engTrain: any = (((props.snap || {}).engine || {}).train) || {}
  const effectiveBatch = (Number(engTrain.batch) || 12) * (Number(engTrain.accum) || 1)
  const modeHint = stepMode === "custom"
    ? "自定义步数：建议 6000–16000；超过 20000 通常不会更好，只会更慢、更容易过拟合。"
    : (STEPS_MODES.find((m: any) => m.key === stepMode) || STEPS_MODES[1]).hint
  const need = requirementOf(c.req, STEPS_CHOICES[1])
  const gap = have >= need.min ? "" : deficitText(have, need.min)
  const clips = clipList(list)
  const phase = String((status && status.phase) || "")
  const running = phase === "training" || phase === "preparing" || phase === "preprocessing" || phase === "previewing"
  const prog = progressText(status)
  const milestones = status ? milestoneList(status) : []
  const ready = milestones.filter((m: any) => m.ready)

  // ---- 训练记录与进度（原来 train.tsx 的节奏不变：4 秒一次） -------------
  const loadRuns = async (keep?: string, keepStep?: boolean) => {
    try {
      const r: any = await callEntry(api, "ms_train_runs", {}, 60000)
      if (r && r.ok && Array.isArray(r.runs)) {
        setRuns(r)
        // 注意：后端返回的 r.active 是「正在跑的训练数量」，不是 run_id。
        // 要从列表里按 alive / 未结束的 phase 找正在跑的那一个。
        const list = r.runs as any[]
        const liveRun = list.find((x: any) => x && ACTIVE_PHASES.indexOf(String(x.phase)) >= 0)
          || list.find((x: any) => x && ENDED.indexOf(String(x.phase)) < 0)
        // 记住的 run_id 可能已经被删了：只认还存在的，否则会留一个空壳页。
        // 刚点「开始训练」时必须优先用新 run_id（keep）：这里闭包里的 runId 还是旧值，
        // 直接用它会导致新训练页显示上一次训练的模型。
        const ids = list.map((x: any) => String(x.run_id))
        // keep 是「刚创建的那条训练」，无条件优先；否则用记住的、正在跑的、最新的。
        const keepId = String(keep || "").trim()
        const want = keepId || String(TRAIN_UI.runId || "").trim()
        const remembered = want && ids.indexOf(want) >= 0 ? want : ""
        const cur = keepId || remembered || (liveRun && liveRun.run_id) || ids[0] || ""
        setRunId(cur)
        if (cur) TRAIN_UI.maxStep = Math.max(TRAIN_UI.maxStep, 3)
        // 进训练页时：有正在跑的训练就去看它；没有就回到第 1 步准备素材，
        // 不要停在上一轮已经结束的「看着她唱」页面。
        if (!keepStep) {
          if (liveRun) setStep(3)
          else setStep(1)
        }
        return
      }
    } catch (e: any) { /* 没有就空着 */ }
    setRuns({ runs: [] })
  }
  useEffect(() => { loadRuns() }, [])

  const loadStatus = async (id?: string) => {
    const rid = id || runId
    if (!rid) return
    try {
      const r: any = await callEntry(api, "ms_train_status", { run_id: rid }, 60000)
      if (r && !r.ok && r.error) return
      if (r && r.ok) setStatus(r)
    } catch (e: any) { /* 单次失败不影响，下一次轮询再试 */ }
  }
  useEffect(() => {
    if (!runId) return
    // 换到另一条训练时先清掉上一条的进度，避免「看着她唱」页短暂显示旧模型的步数。
    setStatus(null)
    loadStatus(runId)
    const id = setInterval(() => loadStatus(runId), 4000)
    return () => clearInterval(id)
  }, [runId])

  // 到了新节点：只弹一次「可以试听」。手动模式会在 awaiting_choice 停下等选择。
  useEffect(() => {
    if (!status) return
    const last = milestones.length ? String(milestones[milestones.length - 1].step) : ""
    if (phase === "awaiting_choice" && last && last !== nodeSeen.current) {
      nodeSeen.current = last
      setNodeOpen(true)
    }
  }, [status])

  // 过拟合提醒（后端只在手动模式给）：排在节点卡之后，仍只有一个弹窗。
  useEffect(() => {
    const of = overfitOf(status)
    if (!of) return
    const key = String(of.step || "")
    if (key && key === overSeen.current) return
    overSeen.current = key
    setOverOpen(true)
  }, [status])

  async function run(key: string, fn: () => Promise<any>) {
    if (busy) return null
    setBusy(key)
    try { return await fn() } catch (e: any) {
      setMsg({ tone: "error", text: String((e && e.message) || e) })
      return null
    } finally { setBusy("") }
  }

  const deleteRun = async (withFiles: boolean) => {
    setDelRun(false)
    if (!runId) return
    try {
      const r: any = await callEntry(api, "ms_delete_run", { run_id: runId, delete_files: withFiles }, 120000)
      if (r && r.ok === false) {
        setMsg({ tone: "error", text: String(r.error || "没删成功") })
        return
      }
      setMsg({ tone: "info", text: "训练记录已删除" + (withFiles ? "（文件已移到回收站）" : "") })
      TRAIN_UI.runId = ""
      setRunId("")
      setStatus(null)
      loadRuns()
    } catch (e: any) {
      setMsg({ tone: "error", text: String((e && e.message) || e) })
    }
  }

  const previewClip = (clip: any) => run("pv:" + clip.id, async () => {
    const r: any = await callEntry(api, "ms_corpus_clip", { clip_id: clip.id, action: "preview" }, 120000)
    const src = audioSrcOf(r)
    if (src) setPreviews((p: any) => ({ ...p, [clip.id]: src }))
    else setMsg({ tone: "error", text: "这段暂时读不出来" })
  })

  const start = () => run("start", async () => {
    if (have < need.min) { setMsg({ tone: "error", text: "素材还不够：" + gap }); return }
    if (stepMode === "custom" && (steps < STEPS_MIN || steps > STEPS_HARD_MAX)) {
      setMsg({ tone: "error", text: "自定义步数要填 " + STEPS_MIN + "～" + STEPS_HARD_MAX + " 之间" })
      return
    }
    const r: any = await callEntry(api, "ms_train_start", {
      steps, name: name.trim() || undefined, from_corpus: true, auto: false,
    }, 120000)
    if (r && r.ok && r.run_id) {
      setRunId(String(r.run_id))
      setMsg({ tone: "info", text: "开始训练了。每 2000 步会出试听，到目标步数才停；觉得够像了可以提前「停下来」保存。" })
      setStep(3)
      loadRuns(String(r.run_id), true)
      loadStatus(String(r.run_id))
    } else setMsg({ tone: "error", text: "没能开始训练：" + ((r && r.error) || "未知错误") })
  })

  const control = (action: "continue" | "stop", extra?: number) => run("ctl:" + action, async () => {
    const add = extra && extra > 0 ? Math.floor(extra) : 0
    const r: any = await callEntry(api, "ms_train_control",
      { run_id: runId, action, steps: add }, 60000)
    setMsg({
      tone: r && r.ok ? "info" : "error",
      text: r && r.ok
        ? (action === "stop" ? "已经让它停下来，稍等一下。"
           : ("继续训练" + (add ? "：再练 " + add + " 步，每 2000 步出试听、中途不停" : "")))
        : "操作没成功：" + ((r && r.error) || "未知错误"),
    })
    setNodeOpen(false)
    setExtraSteps("")
    loadStatus(runId)
  })

  const browse = (m: any) => run("pv:" + m.step, async () => {
    const r: any = await callEntry(api, "ms_train_preview", { run_id: runId, step: Number(m.step) }, 120000)
    const src = audioSrcOf(r)
    if (src) setPreviews((p: any) => ({ ...p, ["node-" + m.step]: src }))
    else setMsg({ tone: "error", text: "这个节点还听不了：" + ((r && r.error) || "") })
  })

  const keep = () => run("keep", async () => {
    if (!pickStep) { setKeepMsg("先挑一个你听着最像的节点"); return }
    const r: any = await callEntry(api, "ms_train_keep", { run_id: runId, step: pickStep, name: keepName.trim() || undefined, set_current: true }, 180000)
    setKeepMsg(r && r.ok ? "保存好了，以后唱歌就用这个声音。可以去「我的声音」里看看。" : "保存失败：" + ((r && r.error) || "未知错误"))
    if (r && r.ok) { loadRuns(undefined, true); props.onRefresh() }
  })

  // 步骤导航（不用表格、不用复杂组件）。素材不够也能先去看设置，只是「开始训练」会拦住。
  // 走到的步数只增不减：退回第 1/2 步后，仍然可以直接点回去看训练进度。
  const reach = (no: number) => no === 1 || no === 2 || no <= TRAIN_UI.maxStep
  const nav = (
    <Row style={{ flexWrap: "wrap", gap: 8 }}>
      {STEP_NAMES.map((t: string, i: number) => {
        const no = i + 1
        const on = step === no
        const ok = reach(no)
        return (
          <button key={no} type="button" disabled={!ok} onClick={() => { if (ok) setStep(no) }} style={{
            border: "1px solid " + (on ? "rgba(224,45,118,0.28)" : LINE), borderRadius: 999, padding: "6px 14px",
            fontSize: 12.5, cursor: ok ? "pointer" : "default", fontFamily: "inherit",
            background: on ? "rgba(255,229,239,0.9)" : "rgba(255,255,255,0.9)",
            color: on ? BRAND_DEEP : INK_2, fontWeight: on ? 700 : 400, opacity: ok ? 1 : 0.55,
          }}>{no}. {t}</button>
        )
      })}
    </Row>
  )

  return (
    <Col style={{ gap: 24 }}>
      {msg ? <Alert tone={msg.tone === "error" ? "warning" : "info"}>{msg.text}</Alert> : null}

      <Row style={{ justifyContent: "space-between", flexWrap: "wrap", gap: 10 }}>
        {nav}
        <Muted>{running ? "正在训练 · 可以先去别的页面，回来接着看" : ready.length ? "挑一个节点保存成她的声音" : ""}</Muted>
      </Row>

      {/* ① 放素材 */}
      {step === 1 ? (
        <Col style={{ gap: 24 }}>
          {c.notice ? <Soft style={{ background: c.notice.tone === "error" ? "#fff1f1" : "#f2f7ff", fontSize: 12.5, color: c.notice.tone === "error" ? "#b23b3b" : INK_2 }}>{c.notice.text}</Soft> : null}
          <Panel>
            <div style={{ display: "flex", gap: 24, flexWrap: "wrap", alignItems: "center", justifyContent: "space-between" }}>
              <div style={{ flex: "1 1 260px", minWidth: 220 }}>
                <div style={{ fontSize: 15.5, fontWeight: 700, color: INK }}>已经准备好 {fmtTime(have)} 可用人声</div>
                <div style={{ fontSize: 13, color: INK_2, marginTop: 6, lineHeight: 1.9 }}>
                  至少需要 {fmtTime(need.min)} · 推荐 {fmtTime(need.rec)} 以上
                </div>
                <Row style={{ marginTop: 8 }}>
                  {gap
                    ? <StatusBadge tone="warning">{gap}就够训练</StatusBadge>
                    : <StatusBadge tone="success">素材够了，可以开始训练</StatusBadge>}
                </Row>
              </div>
              <Col style={{ gap: 8, flex: "0 0 auto" }}>
                <Btn style={PRIMARY_BTN} onClick={() => setStep(2)}>下一步：训练设置</Btn>
                <Btn onClick={() => props.onGo("materials")}>去「准备素材」页</Btn>
              </Col>
            </div>
            <div style={{ marginTop: 12 }}><Bar pct={need.min > 0 ? Math.min(100, (have / need.min) * 100) : 0} /></div>
            <Row style={{ marginTop: 10 }}>
              <Toggle on={c.countSpeech} onChange={c.setCountSpeech} label="把说话 / 朗读录音也算进可用时长（默认不算）" />
            </Row>
          </Panel>

          <Grid cols={3} min={280} gap={24}>
            {MATERIALS.map((m: any) => {
              const v = slotView(list, m.slot)
              const mine = clips.filter((x: any) => x.slot === m.slot)
              const usable = mine.filter((x: any) => x.status !== "rejected")
              const openKey = "open:" + m.slot
              return (
                <Panel key={m.slot} style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  <Row style={{ justifyContent: "space-between", gap: 8 }}>
                    <div style={{ fontSize: 15, fontWeight: 700, color: INK }}>{m.title}</div>
                    <Pill tone={m.slot === "dry_sing" ? "success" : m.slot === "speech" ? "warning" : "neutral"}>{m.ok}</Pill>
                  </Row>
                  <div style={{ fontSize: 12.5, color: INK_2, lineHeight: 1.8, minHeight: 40 }}>{m.lead}</div>
                  {/* 上传只在「准备素材」页做，这里只读：避免两个入口让人以为素材不通用 */}
                  <Row style={{ justifyContent: "space-between", gap: 8, flexWrap: "wrap" }}>
                    <Muted>{v.files} 个文件 · 可用 {fmtTime(v.valid_s)}</Muted>
                    <Btn onClick={() => props.onGo("materials")}>{v.files ? "去准备素材页管理" : "去准备素材页添加"}</Btn>
                  </Row>
                  {mine.length ? (
                    <FoldCard open={Boolean(previews[openKey])} onToggle={() => setPreviews({ ...previews, [openKey]: !previews[openKey] })}
                              title={"其中 " + usable.length + " 段能用（可以试听）"} sub={<Muted>{mine.length > usable.length ? "已自动剔除 " + (mine.length - usable.length) + " 段" : "都通过质检了"}</Muted>}>
                      <Col style={{ gap: 6 }}>
                        {mine.slice(-6).reverse().map((cl: any) => {
                          const bad = cl.status === "rejected"
                          return (
                            <FileRow key={cl.id} icon={bad ? "×" : "♪"}
                                     title={(bad ? "已剔除 · " : "") + fmtTime(cl.dur_s) + (cl.reason ? " · " + cl.reason : "")}
                                     meta={materialTitle(m.slot)}
                                     right={
                                       <Row style={{ gap: 4, flexWrap: "wrap", justifyContent: "flex-end" }}>
                                         {!bad ? <Btn onClick={() => previewClip(cl)}>{busy === "pv:" + cl.id ? "读取中…" : "试听"}</Btn> : null}
                                         <Btn onClick={() => c.setClip(cl.id, bad ? "restore" : "reject")}>{bad ? "恢复" : "剔除"}</Btn>
                                         <Btn style={{ color: "#c93c3c", borderColor: "rgba(201,60,60,0.28)" }}
                                              onClick={() => c.setClip(cl.id, "delete")}>删除</Btn>
                                       </Row>
                                     } />
                          )
                        })}
                        {Object.keys(previews).filter((k: string) => k.indexOf("open:") !== 0 && k.indexOf("node-") !== 0).map((k: string) => (
                          <AudioBox key={k} keyId={k} src={previews[k]} />
                        ))}
                      </Col>
                    </FoldCard>
                  ) : <Muted>还没有投递这一类的素材。</Muted>}
                </Panel>
              )
            })}
          </Grid>
        </Col>
      ) : null}

      {/* ② 训练设置 */}
      {step === 2 ? (
        <Panel title="训练设置" extra={<Muted>按这台电脑的显存自动选参数</Muted>}>
          <Col style={{ gap: 16 }}>
            <div style={{ fontSize: 13, color: INK_2, lineHeight: 1.85 }}>
              她会拿已经收好的素材来学（带伴奏的歌会在后台先把人声分出来）。取个名字，以后在「我的声音」里好认。
            </div>
            <Row style={{ justifyContent: "space-between", gap: 10, flexWrap: "wrap" }}>
              <Muted>训练强度</Muted>
              <Pill tone="success">{steps} 步 · 等效 batch {effectiveBatch}</Pill>
            </Row>
            <Row style={{ gap: 8, flexWrap: "wrap" }}>
              {STEPS_MODES.map((m: any) => (
                <Btn key={m.key} style={stepMode === m.key ? PRIMARY_BTN : undefined}
                     onClick={() => setStepMode(m.key)}>
                  {m.label} · {m.steps} 步
                </Btn>
              ))}
              <Btn style={stepMode === "custom" ? PRIMARY_BTN : undefined} onClick={() => setCustomOpen(true)}>
                自定义{stepMode === "custom" ? " · " + customSteps + " 步" : "…"}
              </Btn>
            </Row>
            <Muted>{modeHint}</Muted>
            <Muted>
              每 2000 步自动生成一段试听（中间不停），到目标步数才停下来让你确认；之后还能追加步数。
            </Muted>
            <Muted>试听会用固定的干声样例（{(((props.snap || {}).engine || {}).preview_src) ? "已配置" : "自动从素材里挑一段"}），每到一个节点合成一段，方便比较像不像。</Muted>
            <Row style={{ gap: 8 }}>
              <Muted>音色名字</Muted>
              <Input value={name} placeholder="例如 小夏（留空按时间自动命名）" style={{ flex: 1, minWidth: 0, maxWidth: 340 }}
                     onChange={(e: any) => setName(e.target.value)} />
            </Row>
            <FoldCard open={openAdv} onToggle={() => setOpenAdv(!openAdv)} title="高级设置（一般不用改）"
                      sub={<Muted>一共 {steps} 步 · 每 {steps > 30000 ? 5000 : 2000} 步停下来让你试听</Muted>}>
              <Col style={{ gap: 8 }}>
                <Muted>步数不是固定的：素材越多，自动算出来的步数越多，练得也更久。</Muted>
                <Muted>现在这一档的参考：至少 {fmtTime(need.min)} 素材，推荐 {fmtTime(need.rec)}。</Muted>
              </Col>
            </FoldCard>
            <Row style={{ justifyContent: "space-between" }}>
              <Btn onClick={() => setStep(1)}>上一步</Btn>
              <Row>
                <Muted>{gap ? gap + "，可以回去再放一点素材" : "素材够用，随时可以开始"}</Muted>
                <Btn style={PRIMARY_BTN} disabled={!!busy || !!gap} onClick={start}>{busy === "start" ? "正在开始…" : "开始训练"}</Btn>
              </Row>
            </Row>
          </Col>
        </Panel>
      ) : null}

      {/* ③ 看着她唱 */}
      {step === 3 ? (
        <Col style={{ gap: 24 }}>
          <Panel title="她在跟着素材学怎么唱"
                 extra={<Select value={runId} onChange={(e: any) => setRunId(e.target.value)} style={{ maxWidth: 300 }}>
                   {((runs && runs.runs) || []).length === 0 ? <option value={runId}>{runId ? "刚开始的训练" : "（还没有训练记录）"}</option> : null}
                   {((runs && runs.runs) || []).map((r: any) => <option key={r.run_id} value={r.run_id}>{(r.name || r.run_id) + (ACTIVE_PHASES.indexOf(String(r.phase)) >= 0 ? " · 训练中" : String(r.phase) === "awaiting_choice" ? " · 等你试听" : "")}</option>)}
                 </Select>}>
            {status ? (
              <Col style={{ gap: 12 }}>
                <Row style={{ justifyContent: "space-between", flexWrap: "wrap" }}>
                  <div style={{ fontSize: 14, fontWeight: 700, color: INK }}>
                    已经学到第 {prog.step}{prog.target ? " 步，下一个节点 " + prog.target : ""} 步
                  </div>
                  <Pill tone={running ? "warning" : ENDED.indexOf(phase) >= 0 ? "success" : "neutral"}>
                    {phase === "awaiting_choice" ? "到节点了，等你试听" : phaseText(phase)}
                  </Pill>
                </Row>
                <Bar pct={prog.pct} />
                <Row style={{ flexWrap: "wrap" }}>
                  {running ? <Muted>预计还要 {prog.etaText || fmtEta(num((status.progress || {}).eta_s))}</Muted> : null}
                  {phase === "awaiting_choice" ? (
                    <Row style={{ gap: 6, alignItems: "center", flexWrap: "wrap" }}>
                      <Muted>还要再练</Muted>
                      <Btn disabled={!!busy} onClick={() => control("continue", 4000)}>+4000 步</Btn>
                      <Btn disabled={!!busy} onClick={() => control("continue", 8000)}>+8000 步</Btn>
                      <Btn disabled={!!busy} onClick={() => setExtraOpen(true)}>自定义…</Btn>
                    </Row>
                  ) : null}
                  {running || phase === "awaiting_choice" ? <Btn disabled={!!busy} onClick={() => control("stop")}>停下来</Btn> : null}
                  {ENDED.indexOf(phase) >= 0 && ready.length ? <Btn style={PRIMARY_BTN} onClick={() => setStep(4)}>下一步：保存声音</Btn> : null}
                  <Btn onClick={() => loadStatus(runId)}>刷新一下</Btn>
                  {runId ? (
                    <Btn style={{ color: "#c93c3c", borderColor: "rgba(201,60,60,0.28)" }}
                         onClick={() => setDelRun(true)}>删除这条训练</Btn>
                  ) : null}
                </Row>
                {status.error ? <Err>{status.error}</Err> : null}
                {phase === "awaiting_choice" ? <Muted>听一听下面最新的节点，觉得像了就「停下来」再保存；不满意就继续练。</Muted> : null}
              </Col>
            ) : <Empty>还没有训练记录。回到第一步放好素材，就能开始。</Empty>}
          </Panel>

          <Section title="每个节点的试听" extra={<Muted>节点越大越像她 · 一共 {ready.length} 个听得了</Muted>}>
            {ready.length ? (
              <Grid cols={2} min={300} gap={16}>
                {ready.slice().reverse().map((m: any) => {
                  const fl = flagText(m.flag)
                  return (
                    <div key={m.step} style={{ padding: 14, borderRadius: 16, background: "rgba(248,250,252,0.95)", border: "1px solid " + LINE }}>
                      <Row style={{ justifyContent: "space-between", gap: 8, flexWrap: "wrap" }}>
                        <div style={{ fontSize: 13, fontWeight: 700, color: INK }}>第 {m.step} 步的录音</div>
                        <Row style={{ gap: 6 }}>
                          <StatusBadge tone={fl.tone}>{fl.text}</StatusBadge>
                          <Btn onClick={() => browse(m)}>{busy === "pv:" + m.step ? "读取中…" : "试听"}</Btn>
                          <Btn style={pickStep === m.step ? PRIMARY_BTN : undefined} onClick={() => setPickStep(Number(m.step))}>
                            {pickStep === m.step ? "已选这个" : "选这个"}
                          </Btn>
                        </Row>
                      </Row>
                      <Row style={{ marginTop: 8 }}>
                        <Muted>越接近又不失真就是刚好；听起来发闷、像被掐住了就是练过头了。</Muted>
                      </Row>
                      {previews["node-" + m.step] ? <AudioBox key={"node-" + m.step} keyId={"node-" + m.step} src={previews["node-" + m.step]} /> : null}
                    </div>
                  )
                })}
              </Grid>
            ) : <Muted>每 2000 步会出一个可以试听的节点，出来后会出现在这里。</Muted>}
          </Section>

          <Row style={{ justifyContent: "space-between" }}>
            <Btn onClick={() => setStep(2)}>上一步</Btn>
            <Btn style={PRIMARY_BTN} disabled={running || !ready.length} onClick={() => setStep(4)}>{running ? "练完这一段再保存声音" : "下一步：保存声音"}</Btn>
          </Row>
        </Col>
      ) : null}

      {/* ④ 保存声音 */}
      {step === 4 ? (
        <Panel title="保存成一个声音" extra={<Muted>不会删掉任何训练结果，随时可以在「我的声音」里换</Muted>}>
          <Col style={{ gap: 14 }}>
            <div style={{ fontSize: 13, color: INK_2, lineHeight: 1.85 }}>
              挑一个你听着最像的节点保存成一个声音。别的节点仍然留在电脑上，以后想换回来在「我的声音」里点一下就行。
            </div>
            <Row style={{ flexWrap: "wrap" }}>
              <Muted>选中的节点</Muted>
              <Pill tone={pickStep ? "success" : "neutral"}>{pickStep ? "第 " + pickStep + " 步" : "还没选"}</Pill>
              <Select value={pickStep ? String(pickStep) : ""} onChange={(e: any) => setPickStep(Number(e.target.value) || 0)} style={{ minWidth: 200, maxWidth: 340 }}>
                <option value="">（挑一个听过的节点）</option>
                {ready.map((m: any) => <option key={m.step} value={String(m.step)}>第 {m.step} 步</option>)}
              </Select>
            </Row>
            <Row style={{ gap: 8, flexWrap: "wrap" }}>
              <Input value={keepName} placeholder="给这个声音取个名字，例如 小夏" style={{ flex: 1, minWidth: 0, maxWidth: 340 }}
                     onChange={(e: any) => setKeepName(e.target.value)} />
              <Btn style={PRIMARY_BTN} disabled={!!busy || !pickStep} onClick={keep}>{busy === "keep" ? "保存中…" : "保存这个声音"}</Btn>
            </Row>
            {keepMsg ? <Alert tone={keepMsg.indexOf("失败") >= 0 || keepMsg.indexOf("先挑") >= 0 ? "warning" : "info"}>{keepMsg}</Alert> : null}
            <Row style={{ justifyContent: "space-between" }}>
              <Btn onClick={() => setStep(3)}>上一步</Btn>
              <Btn onClick={() => props.onGo("models")}>去「我的声音」看看 →</Btn>
            </Row>
          </Col>
        </Panel>
      ) : null}

      <Soft style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap", justifyContent: "space-between" }}>
        <Muted>训练用的还是同一批素材；换声音、试听都在「我的声音」里。不要的训练记录可以删掉。</Muted>

      <DeletePrompt open={delRun}
                    title="删除这条训练记录？"
                    message={(runId || "") + " —— 连本地文件一起删会把它的检查点（model_*.pt）和实验目录移进回收站；"
                             + "只删记录则保留检查点，之后还能在「我的声音」里看到。"}
                    recordLabel="只删记录（检查点保留）"
                    onCancel={() => setDelRun(false)}
                    onRecordOnly={() => deleteRun(false)}
                    onWithFiles={() => deleteRun(true)} />
        <Btn onClick={() => props.onGo("models")}>我的声音</Btn>
      </Soft>

      {/* 弹窗：永远只有一个。先弹「到节点了」，过拟合提醒排在它后面 */}
      {nodeOpen ? (
        <Soft style={{ position: "fixed", inset: 0, zIndex: 1150, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(40,37,50,0.38)" }}>
          <div style={{ background: "#fff", borderRadius: 20, padding: 22, maxWidth: 420, boxShadow: "0 18px 44px rgba(40,37,50,0.22)" }}>
            <div style={{ fontSize: 15, fontWeight: 700, color: INK }}>有一个新节点可以试听</div>
            <div style={{ fontSize: 12.5, color: INK_2, lineHeight: 1.85, margin: "10px 0 16px" }}>
              已经学到第 {prog.step} 步{milestones.length ? "（最近节点 " + milestones[milestones.length - 1].step + " 步）" : ""}。
              听一句像不像，可以继续练，也可以停下来保存现在这个。
            </div>
            <Row style={{ justifyContent: "flex-end" }}>
              <Btn disabled={!!busy} onClick={() => control("stop")}>停下来保存</Btn>
              <Btn style={PRIMARY_BTN} disabled={!!busy} onClick={() => { setNodeOpen(false); setStep(3) }}>听听看</Btn>
            </Row>
          </div>
        </Soft>
      ) : null}

      {customOpen ? (
        <Soft style={{ position: "fixed", inset: 0, zIndex: 1160, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(40,37,50,0.38)" }}>
          <div style={{ background: "#fff", borderRadius: 20, padding: 22, maxWidth: 440, boxShadow: "0 18px 44px rgba(40,37,50,0.22)" }}>
            <div style={{ fontSize: 15, fontWeight: 700, color: INK }}>自定义训练步数</div>
            <div style={{ fontSize: 12.5, color: INK_2, lineHeight: 1.85, margin: "10px 0 14px" }}>
              步数不是越多越好：练过头会把训练集"背下来"，出现发闷、发尖、丢原味；通常 6000–16000 就够，
              超过 20000 多数情况下不会更好，只会更慢。
            </div>
            <Row style={{ gap: 8, alignItems: "center", marginBottom: 14 }}>
              <Input value={customSteps} style={{ width: 120 }}
                     onChange={(e: any) => setCustomSteps(String(e.target.value || "").replace(/[^0-9]/g, ""))} />
              <Muted>步（参数更新次数，已计入梯度累积）</Muted>
            </Row>
            <Row style={{ justifyContent: "flex-end" }}>
              <Btn onClick={() => setCustomOpen(false)}>取消</Btn>
              <Btn style={PRIMARY_BTN} disabled={!Number(customSteps)} onClick={() => {
                setStepMode("custom"); setCustomOpen(false)
              }}>确定</Btn>
            </Row>
          </div>
        </Soft>
      ) : null}

      {extraOpen ? (
        <Soft style={{ position: "fixed", inset: 0, zIndex: 1160, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(40,37,50,0.38)" }}>
          <div style={{ background: "#fff", borderRadius: 20, padding: 22, maxWidth: 440, boxShadow: "0 18px 44px rgba(40,37,50,0.22)" }}>
            <div style={{ fontSize: 15, fontWeight: 700, color: INK }}>再练多少步？</div>
            <div style={{ fontSize: 12.5, color: INK_2, lineHeight: 1.85, margin: "10px 0 14px" }}>
              这是在当前模型上追加步数。建议先听最近的节点，再决定加 4000–8000 步；
              连续加 20000 步以上很可能开始过拟合，音色会变尖、失真。
            </div>
            <Row style={{ gap: 8, alignItems: "center", marginBottom: 14 }}>
              <Input value={extraSteps} placeholder="例如 4000" style={{ width: 120 }}
                     onChange={(e: any) => setExtraSteps(String(e.target.value || "").replace(/[^0-9]/g, ""))} />
              <Muted>步</Muted>
            </Row>
            <Row style={{ justifyContent: "flex-end" }}>
              <Btn onClick={() => setExtraOpen(false)}>取消</Btn>
              <Btn style={PRIMARY_BTN} disabled={!!busy || !Number(extraSteps)} onClick={() => {
                setExtraOpen(false); control("continue", Number(extraSteps))
              }}>继续训练</Btn>
            </Row>
          </div>
        </Soft>
      ) : null}

      {!nodeOpen && overOpen ? (
        <Soft style={{ position: "fixed", inset: 0, zIndex: 1150, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(40,37,50,0.38)" }}>
          <div style={{ background: "#fff", borderRadius: 20, padding: 22, maxWidth: 440, boxShadow: "0 18px 44px rgba(40,37,50,0.22)" }}>
            <div style={{ fontSize: 15, fontWeight: 700, color: INK }}>好像练得有点过了</div>
            <div style={{ fontSize: 12.5, color: INK_2, lineHeight: 1.85, margin: "10px 0 16px" }}>
              再练下去可能只会唱训练时听过的歌，别的歌就不像她了。
              {pickStep ? "建议就保存第 " + pickStep + " 步。" : "建议挑前面一个节点保存。"}
            </div>
            <Row style={{ justifyContent: "flex-end" }}>
              <Btn onClick={() => { setOverOpen(false); setStep(3) }}>回去挑一个</Btn>
              <Btn style={PRIMARY_BTN} disabled={!!busy} onClick={() => { setOverOpen(false); control("stop") }}>停下来</Btn>
            </Row>
          </div>
        </Soft>
      ) : null}
    </Col>
  )
}
