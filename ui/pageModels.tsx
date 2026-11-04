// 为她而唱 · 我的声音（2026-11-04）
//
// 把她训练好的声音排成卡片：点「设为当前」= ms_set_active_model（之后再多存一次 ms_set_config{ddsp_ckpt}，
// 让设置页里的模型选择也跟过来），点「试听」= ms_train_preview。正在训练中的声音单独一组，标注「还没训完」。
// 只读：这里不删除任何训练结果。
import { Alert, StatusBadge, useState } from "@neko/plugin-ui"
import { audioSrcOf, callEntry, dataUriOf } from "./api"
import { BRAND, BRAND_DEEP, INK, INK_2, LINE, Bar, Btn, Col, DeletePrompt, Empty, Grid, Muted, Panel, PRIMARY_BTN, Pill, Row, Section } from "./ui"

export function PageModels(props: { api: any; snap: any; onRefresh: () => void; onGo: (page: string) => void }) {
  const { api, snap, onRefresh } = props
  const s = snap || {}
  const eng = s.engine || {}
  const all = Array.isArray(eng.models) ? eng.models : []
  const ready = all.filter((m: any) => !m.training)
  const working = all.filter((m: any) => m.training)
  const currentVoice = all.find((m: any) => m && m.current) || null
  const currentName = currentVoice ? String(currentVoice.exp_dir || "") : ""
  const [busy, setBusy] = useState("")
  const [notice, setNotice] = useState<any>(null)
  const [open, setOpen] = useState("")
  const [audio, setAudio] = useState<any>(null)
  const [delTarget, setDelTarget] = useState<any>(null)
  const [exported, setExported] = useState<any>(null)

  const setCurrent = async (m: any) => {
    setBusy("cur:" + m.path)
    setNotice(null)
    try {
      const r: any = await callEntry(api, "ms_set_active_model", { ckpt: m.path }, 60000)
      const saved: any = await callEntry(api, "ms_set_config", { ddsp_ckpt: m.path }, 30000)
      const ok = !!(r && r.ok) || !!(saved && saved.ok)
      setNotice({ tone: ok ? "info" : "error", text: ok ? "以后唱歌都用「" + (m.exp_dir || "训练好的声音") + "」了" : "没切换成功：" + ((saved && saved.error) || (r && r.error) || "未知错误") })
      onRefresh()
    } catch (e: any) {
      setNotice({ tone: "error", text: String((e && e.message) || e) })
    }
    setBusy("")
  }

  const exportModel = async (m: any) => {
    setBusy("exp:" + m.path)
    setNotice(null)
    try {
      const r: any = await callEntry(api, "ms_export_model", { path: m.path }, 600000)
      if (r && r.ok) {
        setExported(r)
        setNotice({ tone: "info", text: "已导出「" + (r.name || m.exp_dir || "声音") + "」" })
      } else {
        setNotice({ tone: "error", text: "导出失败：" + ((r && r.error) || "未知错误") })
      }
    } catch (e: any) {
      setNotice({ tone: "error", text: "导出失败：" + String((e && e.message) || e) })
    }
    setBusy("")
  }

  const openExportDir = async () => {
    try {
      const r: any = await callEntry(api, "ms_open_export_dir", {}, 30000)
      if (r && r.ok === false) setNotice({ tone: "error", text: String(r.error || "打不开导出文件夹") })
    } catch (e: any) {
      setNotice({ tone: "error", text: String((e && e.message) || e) })
    }
  }

  const deleteModel = async (withFiles: boolean) => {
    const m = delTarget
    setDelTarget(null)
    if (!m) return
    setBusy("del:" + m.path)
    setNotice(null)
    try {
      const r: any = await callEntry(api, "ms_delete_model", { path: m.path, delete_files: withFiles }, 60000)
      if (r && r.ok === false) setNotice({ tone: "error", text: String(r.error || "没删成功") })
      else setNotice({ tone: "info", text: (m.exp_dir || "训练好的声音") + " 已删除" + (withFiles ? "（文件已移到回收站）" : "") })
      onRefresh()
    } catch (e: any) {
      setNotice({ tone: "error", text: String((e && e.message) || e) })
    }
    setBusy("")
  }

  const preview = async (m: any) => {
    setBusy("pv:" + m.path)
    setNotice(null)
    try {
      const r: any = await callEntry(api, "ms_train_preview", { run_id: String(m.exp_dir || ""), step: Number(m.steps || 0) }, 120000)
      const src = audioSrcOf(r)
      if (src) setAudio({ id: m.path, title: m.exp_dir || "训练好的声音", src })
      else setNotice({ tone: "error", text: "试听暂时不可用：" + ((r && r.error) || "没有拿到音频") })
    } catch (e: any) {
      setNotice({ tone: "error", text: "试听暂时不可用：" + String((e && e.message) || e) })
    }
    setBusy("")
  }

  const card = (m: any, isWorking: boolean) => (
    <Panel key={m.path} style={{ display: "flex", flexDirection: "column", gap: 12, border: m.current ? "1px solid rgba(224,45,118,0.35)" : undefined }}>
      <Row style={{ justifyContent: "space-between", gap: 8 }}>
        <div style={{ display: "flex", gap: 12, alignItems: "center", minWidth: 0 }}>
          <span style={{ width: 40, height: 40, borderRadius: 13, background: "linear-gradient(150deg,#ffe5ef,#f4d6ea)", color: BRAND, display: "grid", placeItems: "center", fontSize: 18, flex: "0 0 auto" }}>♪</span>
          <div style={{ minWidth: 0 }}>
            <div style={{ fontSize: 13.5, fontWeight: 700, color: INK, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{m.exp_dir || "训练好的声音"}</div>
            <div style={{ fontSize: 11.5, color: INK_2 }}>{m.mtime || ""}{m.size_mb ? " · " + m.size_mb + " MB" : ""}</div>
          </div>
        </div>
        {m.current ? <Pill tone="success">正在使用</Pill> : isWorking ? <Pill tone="warning">还没训完</Pill> : null}
      </Row>
      <Row>
        {m.has_config ? null : <StatusBadge tone="warning">缺少配置文件，可能用不了</StatusBadge>}
        {(m.spks || []).length ? <Muted>歌者：{(m.spks || []).join(" / ")}</Muted> : null}
      </Row>
      {open === m.path ? (
        <div style={{ padding: 10, borderRadius: 12, background: "rgba(248,250,252,0.95)", border: "1px solid " + LINE }}>
          <Muted style={{ wordBreak: "break-all" }}>{m.path}</Muted>
        </div>
      ) : null}
      <Row style={{ marginTop: "auto" }}>
        {!isWorking ? (
          <Btn style={m.current ? undefined : PRIMARY_BTN} disabled={!!busy || m.current} onClick={() => setCurrent(m)}>
            {m.current ? "正在使用" : busy === "cur:" + m.path ? "切换中…" : "设为当前"}
          </Btn>
        ) : null}
        <Btn disabled={!!busy} onClick={() => preview(m)}>{busy === "pv:" + m.path ? "读取中…" : "试听"}</Btn>
        {!isWorking ? (
          <Btn disabled={!!busy} onClick={() => exportModel(m)}>{busy === "exp:" + m.path ? "导出中…" : "导出"}</Btn>
        ) : null}
        <Btn onClick={() => setOpen(open === m.path ? "" : m.path)}>{open === m.path ? "收起位置" : "在哪里"}</Btn>
        {!isWorking && !m.current ? (
          <Btn style={{ color: "#c93c3c", borderColor: "rgba(201,60,60,0.28)" }}
               disabled={!!busy} onClick={() => setDelTarget(m)}>删除</Btn>
        ) : null}
      </Row>
    </Panel>
  )

  return (
    <Col style={{ gap: 24 }}>
      {notice ? <Alert tone={notice.tone === "error" ? "warning" : "info"}>{notice.text}</Alert> : null}

      <Section title="训练好的声音" extra={<Muted>{ready.length} 个 · 当前：{currentName || "未选"}</Muted>}>
        {ready.length === 0 ? (
          <Empty>还没有训练好的声音。先去「准备素材」放几首歌，再到「训练声音」开始训练。</Empty>
        ) : (
          <Grid cols={3} min={280} gap={20}>
            {ready.map((m: any) => card(m, false))}
          </Grid>
        )}
        {audio ? (
          <div style={{ marginTop: 12, padding: 12, borderRadius: 14, background: "#fff", border: "1px solid " + LINE }}>
            <Row style={{ justifyContent: "space-between", marginBottom: 6 }}>
              <div style={{ fontSize: 12.5, color: INK, fontWeight: 600 }}>试听 · {audio.title}</div>
              <Btn onClick={() => setAudio(null)}>关掉</Btn>
            </Row>
            <audio key={audio.id} controls autoPlay src={audio.src} style={{ width: "100%" }} />
          </div>
        ) : null}
        {exported ? (
          <Row style={{ marginTop: 12, flexWrap: "wrap", gap: 10, alignItems: "center" }}>
            <Muted style={{ wordBreak: "break-all" }}>已导出到：{exported.path}</Muted>
            <Btn onClick={openExportDir}>打开导出文件夹</Btn>
          </Row>
        ) : null}
        <Row style={{ marginTop: 14 }}>
          <Btn style={PRIMARY_BTN} onClick={() => props.onGo("train")}>训练新的声音 →</Btn>
          <Btn onClick={() => props.onGo("sing")}>去让她唱歌</Btn>
        </Row>
      </Section>

      {working.length ? (
        <Section title="还在训练" extra={<Muted>训完后会自动出现在上面</Muted>}>
          <Grid cols={3} min={280} gap={20}>
            {working.map((m: any) => card(m, true))}
          </Grid>
          <div style={{ marginTop: 10 }}><Bar pct={0} tone="rgba(224,45,118,0.25)" /></div>
          <Muted>进度在「训练声音」页能看到。</Muted>
        </Section>
      ) : null}

      <Muted style={{ color: BRAND_DEEP }}>提示：不要的声音可以直接删；选择「连本地文件一起删除」时会移进回收站，能找回。</Muted>

      <DeletePrompt open={!!delTarget}
                    title="删除这个训练好的声音？"
                    message={(delTarget ? (delTarget.exp_dir || delTarget.name) : "") +
                             " —— 连本地文件一起删会把它的检查点和实验目录移进回收站。"}
                    recordLabel="只从列表移除（文件保留）"
                    onCancel={() => setDelTarget(null)}
                    onRecordOnly={() => deleteModel(false)}
                    onWithFiles={() => deleteModel(true)} />
    </Col>
  )
}
