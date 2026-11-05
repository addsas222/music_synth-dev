// 为她而唱 · 原创歌曲（ACE-Step text2music，2026-11-05）
//
// 流程：风格描述 + 歌词 → ACE-Step 本地生成整首歌 → 入库成一条曲目。
// 「生成并让她唱」会直接复用现有链路：分离 → DDSP 变声 → 混音。
import { Alert, StatusBadge, useEffect, useRef, useState } from "@neko/plugin-ui"
import { callEntry } from "./api"
import { BRAND_DEEP, INK, INK_2, Btn, Col, Field, FoldCard, Grid, Input, Muted, Panel, PRIMARY_BTN, Pill, Row, Section, Select, Toggle } from "./ui"

const TA_STYLE = {
  border: "1px solid transparent", borderRadius: 12, padding: "9px 14px", fontSize: 13,
  background: "rgba(248,244,248,0.95)", color: INK, minWidth: 0, outline: "none",
  boxShadow: "inset 0 1px 2px rgba(40,37,50,0.04)", fontFamily: "inherit",
  resize: "vertical", lineHeight: 1.7, width: "100%", boxSizing: "border-box",
} as any

const LANGS = [
  ["auto", "自动判断"], ["zh", "中文"], ["ja", "日语"], ["en", "英语"], ["ko", "韩语"],
]

export function PageCreate(props: { api: any; snap: any; onRefresh: () => void; onGo: (page: string) => void }) {
  const { api } = props
  const [status, setStatus] = useState<any>(null)
  const [prompt, setPrompt] = useState("")
  const [lyrics, setLyrics] = useState("")
  const [name, setName] = useState("")
  const [duration, setDuration] = useState("0")
  const [bpm, setBpm] = useState("0")
  const [language, setLanguage] = useState("auto")
  const [instrumental, setInstrumental] = useState(false)
  const [lora, setLora] = useState("")
  const [busy, setBusy] = useState("")
  const [jobId, setJobId] = useState("")
  const [job, setJob] = useState<any>(null)
  const [notice, setNotice] = useState<any>(null)
  const poll = useRef<any>(null)
  const [trainOpen, setTrainOpen] = useState(false)
  const [trainName, setTrainName] = useState("")
  const [trainEpochs, setTrainEpochs] = useState("10")
  const [trainRank, setTrainRank] = useState("32")
  const [trainBusy, setTrainBusy] = useState("")

  const check = async () => {
    try {
      const r: any = await callEntry(api, "ms_ace_status", {}, 30000)
      setStatus(r)
    } catch (e: any) {
      setStatus({ ok: false, problems: [String((e && e.message) || e)] })
    }
  }
  useEffect(() => { check(); return () => { if (poll.current) clearInterval(poll.current) } }, [])

  useEffect(() => {
    if (!jobId) return
    const tick = async () => {
      try {
        const r: any = await callEntry(api, "ms_ace_job", { job_id: jobId }, 30000)
        const j = (r && r.job) || null
        setJob(j)
        if (j && (j.state === "done" || j.state === "failed")) {
          if (poll.current) { clearInterval(poll.current); poll.current = null }
          setBusy("")
          setTrainBusy("")
          if (j.state === "done" && j.lora_path) {
            const parts = String(j.lora_path).split(/[\\/]/)
            setLora(parts[parts.length - 1] || "")
          }
          check()
          props.onRefresh()
        }
      } catch (e) { /* 下一拍再试 */ }
    }
    tick()
    poll.current = setInterval(tick, 3000)
    return () => { if (poll.current) { clearInterval(poll.current); poll.current = null } }
  }, [jobId])

  const start = async (sing: boolean) => {
    if (busy) return
    if (!prompt.trim() && !lyrics.trim()) {
      setNotice({ tone: "error", text: "至少写一句风格描述或歌词" })
      return
    }
    setBusy(sing ? "sing" : "gen")
    setNotice(null)
    setJob(null)
    setJobId("")
    try {
      const r: any = await callEntry(api, "ms_ace_generate", {
        prompt: prompt.trim(),
        lyrics: instrumental ? "" : lyrics.trim(),
        name: name.trim(),
        duration: Number(duration) || 0,
        bpm: Number(bpm) || 0,
        language,
        instrumental,
        auto_sing: sing && !lora.trim(),
        inference_steps: 12,
        guidance_scale: 7.0,
        lora,
      }, 60000)
      if (r && r.ok && r.job_id) {
        setJobId(String(r.job_id))
        setNotice({ tone: "info", text: "开始生成：" + (r.name || "") + " · 约 " + (r.duration || "?") + " 秒" })
      } else {
        setBusy("")
        setNotice({ tone: "error", text: "没能开始：" + ((r && r.error) || "未知错误") })
      }
    } catch (e: any) {
      setBusy("")
      setNotice({ tone: "error", text: String((e && e.message) || e) })
    }
  }

  const install = async () => {
    if (busy) return
    setBusy("install")
    setNotice(null)
    setJob(null)
    setJobId("")
    try {
      const r: any = await callEntry(api, "ms_ace_install", {}, 30000)
      if (r && r.ok && r.job_id) {
        setJobId(String(r.job_id))
        setNotice({ tone: "info", text: "开始安装 ACE 依赖（约 100MB，需要联网）…" })
      } else {
        setBusy("")
        setNotice({ tone: "error", text: "没能开始安装：" + ((r && r.error) || "未知错误") })
      }
    } catch (e: any) {
      setBusy("")
      setNotice({ tone: "error", text: String((e && e.message) || e) })
    }
  }

  const ready = status && status.ok
  const loras = (status && Array.isArray(status.loras)) ? status.loras : []

  const startTrain = async () => {
    if (trainBusy) return
    if (!ready) { setNotice({ tone: "error", text: "ACE 环境还没就绪" }); return }
    setTrainBusy("start")
    setNotice(null)
    setJob(null)
    setJobId("")
    try {
      const r: any = await callEntry(api, "ms_ace_train", {
        name: trainName.trim(),
        epochs: Number(trainEpochs) || 10,
        rank: Number(trainRank) || 32,
      }, 60000)
      if (r && r.ok && r.job_id) {
        setJobId(String(r.job_id))
        setNotice({ tone: "info", text: "开始训练：" + (r.name || "") + "，素材 " + (r.samples || 0) + " 段" })
      } else {
        setTrainBusy("")
        setNotice({ tone: "error", text: "没能开始训练：" + ((r && r.error) || "未知错误") })
      }
    } catch (e: any) {
      setTrainBusy("")
      setNotice({ tone: "error", text: String((e && e.message) || e) })
    }
  }

  const stateText = !job ? "" : job.state === "queued" ? "排队中" : job.state === "running" ? "生成中" : job.state === "done" ? "已完成" : job.state === "failed" ? "失败" : job.state
  const isTrainJob = Boolean(job && job.job === "ace-train")

  return (
    <Col style={{ gap: 24 }}>
      {notice ? <Alert tone={notice.tone === "error" ? "warning" : "info"}>{notice.text}</Alert> : null}
      {status && !status.ok ? (
        <Alert tone="warning">
          <Col style={{ gap: 8 }}>
            <div>原创生成还没就绪：{(status.problems || []).join("、")}</div>
            <Row style={{ gap: 8, flexWrap: "wrap" }}>
              <Btn disabled={!!busy} onClick={install}>{busy === "install" ? "安装中…" : "安装 ACE 依赖"}</Btn>
              <Muted>依赖约 100MB；模型会在第一次生成时自动下载到 {status.paths ? status.paths.checkpoints : "计算根目录"}。</Muted>
            </Row>
          </Col>
        </Alert>
      ) : null}

      <Panel title="原创歌曲（ACE-Step）"
             extra={<Pill tone={ready ? "success" : "warning"}>{ready ? "本地模型已就绪" : "环境未就绪"}</Pill>}>
        <Col style={{ gap: 14 }}>
          <div style={{ fontSize: 12.5, color: INK_2, lineHeight: 1.8 }}>
            用文字生成一首完整的歌（人声 + 伴奏），生成后可以：
            <b>只生成</b>（先试听原曲），或 <b>生成并让她唱</b>（直接走分离 → 变声 → 混音，用当前音色）。
          </div>
          <Grid cols={2} min={240} gap={14}>
            <Field label="歌名（留空自动命名）">
              <Input value={name} placeholder="例如 夏夜的风" onChange={(e: any) => setName(e.target.value)} />
            </Field>
            <Field label="时长（秒；0 = 按歌词估算）">
              <Input type="number" min={10} max={600} step={5} value={duration} onChange={(e: any) => setDuration(e.target.value)} />
            </Field>
          </Grid>
          <Field label="风格 / 编曲描述">
            <textarea style={TA_STYLE} rows={3} value={prompt}
                      placeholder="例如：温暖的女声流行，钢琴 + 弦乐，中速，情绪逐渐上扬"
                      onChange={(e: any) => setPrompt(e.target.value)} />
          </Field>
          <Field label="歌词（留空或勾选纯音乐则生成纯伴奏）">
            <textarea style={TA_STYLE} rows={7} value={lyrics}
                      placeholder={"支持 [verse] / [chorus] 结构标签\n第一句歌词…\n第二句歌词…"}
                      onChange={(e: any) => setLyrics(e.target.value)} />
          </Field>
          <Grid cols={3} min={180} gap={14}>
            <Field label="BPM（0 = 自动）">
              <Input type="number" min={0} max={300} step={1} value={bpm} onChange={(e: any) => setBpm(e.target.value)} />
            </Field>
            <Field label="演唱语言">
              <Select value={language} onChange={(e: any) => setLanguage(e.target.value)}>
                {LANGS.map(([v, t]: any) => <option key={v} value={v}>{t}</option>)}
              </Select>
            </Field>
            <Field label="生成方式">
              <Row style={{ paddingTop: 6 }}>
                <Toggle on={instrumental} onChange={setInstrumental} label="纯音乐（不唱）" />
              </Row>
            </Field>
          </Grid>
          <Field label="ACE 音色（LoRA；训练后可直接选）">
            <Select value={lora} onChange={(e: any) => setLora(e.target.value)}>
              <option value="">默认音色（不加载 LoRA）</option>
              {loras.map((l: any) => <option key={l.name} value={l.name}>{l.name}</option>)}
            </Select>
          </Field>
          {lora ? (
            <Muted>已选 ACE 音色：生成结果直接就是该音色，不会再走 DDSP 二次变声。</Muted>
          ) : null}
          <Row style={{ flexWrap: "wrap", alignItems: "center", gap: 14 }}>
            <Muted>首次加载 ACE 模型会慢一些；同一时间只跑一个生成/唱的任务。</Muted>
          </Row>
          <Row style={{ flexWrap: "wrap", gap: 8 }}>
            <Btn disabled={!!busy || !ready} onClick={() => start(false)}>只生成</Btn>
            <Btn style={PRIMARY_BTN} disabled={!!busy || !ready} onClick={() => start(true)}>
              {busy ? "正在生成…" : (lora ? "用这个 ACE 音色生成" : "生成并让她唱")}
            </Btn>
            <Btn disabled={!!busy} onClick={check}>重新检查环境</Btn>
          </Row>
        </Col>
      </Panel>

      <FoldCard open={trainOpen} onToggle={() => setTrainOpen(!trainOpen)}
                title="训练 ACE 音色（LoRA）"
                sub={<Muted>让 ACE 直接唱出目标音色，而不是每次生成后再变声</Muted>}>
        <Col style={{ gap: 14 }}>
          <Muted>
            素材固定取「准备素材」里已经处理好的切片
            （当前 {status && typeof status.train_clips === "number" ? status.train_clips : "?"} 段）。
            风格由系统根据音色自动判断，歌词使用通用元音，不需要你填英文。
            训练会长时间占用显卡，期间翻唱/生成会排队。
          </Muted>
          <Grid cols={3} min={200} gap={14}>
            <Field label="音色名（留空自动命名）">
              <Input value={trainName} placeholder="例如 梦梦pro" onChange={(e: any) => setTrainName(e.target.value)} />
            </Field>
            <Field label="训练轮数（1～100）">
              <Input type="number" min={1} max={100} step={1} value={trainEpochs} onChange={(e: any) => setTrainEpochs(e.target.value)} />
            </Field>
            <Field label="LoRA rank（4～128）">
              <Input type="number" min={4} max={128} step={4} value={trainRank} onChange={(e: any) => setTrainRank(e.target.value)} />
            </Field>
          </Grid>
          <Row style={{ flexWrap: "wrap", alignItems: "center", gap: 12 }}>
            <Btn style={PRIMARY_BTN} disabled={!ready || !!trainBusy || !!busy} onClick={startTrain}>
              {trainBusy ? "正在训练…" : "开始训练"}
            </Btn>
            <Muted>首次训练约 10～40 分钟，取决于素材量和轮数；显卡显存占用约 6～10GB。</Muted>
          </Row>
        </Col>
      </FoldCard>

      {job ? (
        <Section title={isTrainJob ? "训练进度" : "生成进度"} extra={<Muted>{stateText}</Muted>}>
          <Panel>
            <Col style={{ gap: 10 }}>
              <Row style={{ justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
                <div style={{ fontSize: 14, fontWeight: 700, color: INK }}>
                  {job.state === "done" ? (isTrainJob ? "训练完成" : "生成完成")
                    : job.state === "failed" ? (isTrainJob ? "训练失败" : "生成失败")
                    : (isTrainJob ? "ACE 音色训练中" : "ACE-Step 正在生成")}
                </div>
                <StatusBadge tone={job.state === "done" ? "success" : job.state === "failed" ? "error" : "warning"}>
                  {stateText}
                </StatusBadge>
              </Row>
              <div style={{ height: 8, borderRadius: 999, background: "rgba(40,37,50,0.08)", overflow: "hidden" }}>
                <div style={{ width: (Number(job.progress) || (job.state === "done" ? 100 : 15)) + "%", height: "100%", background: BRAND_DEEP }} />
              </div>
              <Muted>{job.detail || job.message || "正在准备…"}</Muted>
              {job.state === "done" ? (
                <Row style={{ flexWrap: "wrap", gap: 8 }}>
                  {isTrainJob ? <Btn style={PRIMARY_BTN} onClick={check}>刷新音色列表</Btn> : null}
                  <Btn style={isTrainJob ? undefined : PRIMARY_BTN} onClick={() => props.onGo("sing")}>去「让她唱歌」看看</Btn>
                  <Btn onClick={() => props.onGo("models")}>我的声音</Btn>
                </Row>
              ) : null}
            </Col>
          </Panel>
        </Section>
      ) : null}

      <Section title="说明" extra={<Muted>本地运行，不联网、不需要账号</Muted>}>
        <Panel>
          <Col style={{ gap: 8 }}>
            <Muted>· ACE 生成的是“整首歌”；「生成并让她唱」会自动分离人声和伴奏，再用当前音色变声并混回去。</Muted>
            <Muted>· 歌词越完整、风格描述越具体，生成越稳定；纯音乐请勾选“纯音乐”。</Muted>
            <Muted>· 同一时间只跑一个重活；正在训练或唱别的歌时，这里会排队等待。</Muted>
            {status && status.paths ? <Muted>· 模型目录：{status.paths.checkpoints}</Muted> : null}
          </Col>
        </Panel>
      </Section>
    </Col>
  )
}
