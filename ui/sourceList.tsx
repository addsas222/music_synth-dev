// 为她而唱 · 素材文件列表（2026-11-04）
//
// 「准备素材」三张卡片下面那几行文件：可以收起 / 展开，每行能 试听 · 隐藏 · 删除。
// 隐藏只是把这一行从列表里收起来（不参与训练计时，也不删文件），点「显示已隐藏」还能放回来；
// 删除会把原文件和它的切片一起移到回收站，能找回。
import { useState } from "@neko/plugin-ui"
import { audioSrcOf, callEntry, dataUriOf } from "./api"
import { Bar, Btn, Col, Muted, Row, fmtTime, INK, LINE } from "./ui"
import { num } from "./trainModel"

const DANGER = { color: "#c93c3c", borderColor: "rgba(201,60,60,0.28)" }

/** 素材源有两种形状：mock 里是文件名字符串；后端是 {files:[{path,name,bytes}]}。 */
function innerOf(f: any): any {
  return f && Array.isArray(f.files) && f.files.length ? f.files[0] : f
}

function nameOf(f: any, i: number): string {
  const inner = innerOf(f)
  const raw = inner && (inner.name || inner.path)
  const base = raw ? String(raw).split(/[\\/]/).pop() || String(raw) : "文件 " + (i + 1)
  return base
}

function bytesOf(f: any): number {
  const inner = innerOf(f)
  return inner && inner.bytes ? num(inner.bytes) : 0
}

function durOf(f: any): number {
  const inner = innerOf(f)
  return inner ? num(inner.dur_s) : 0
}

/** 「9 MB · 3 分 40 秒」——时长是上传时 ffprobe 量好的。 */
function metaOf(f: any): string {
  if (f && f.status === "queued") return "排队中…"
  if (f && f.status === "processing") return "处理中…"
  if (f && f.status === "error") return "上次处理中断，可重新添加"
  const bits: string[] = []
  if (bytesOf(f)) bits.push(Math.max(1, Math.round(bytesOf(f) / 1048576)) + " MB")
  if (durOf(f)) bits.push(fmtTime(durOf(f)))
  return bits.length ? bits.join(" · ") : "已收到"
}

export function SourceFileList(props: {
  api: any
  slot: string
  files: any[]
  pending?: any[]
  jobs?: any[]
  validS?: number
  onChanged: () => void
}) {
  const [open, setOpen] = useState(true)
  const [showHidden, setShowHidden] = useState(false)
  const [preview, setPreview] = useState<any>(null)
  const [busy, setBusy] = useState("")
  const [errMsg, setErrMsg] = useState("")
  // 删除成功后先本地移除这一行：不用等刷新，也避免对着同一行连点
  const [removed, setRemoved] = useState<string[]>([])

  const all = Array.isArray(props.files) ? props.files : []
  const visible = all.filter((f: any) => !(f && f.hidden))
  const hidden = all.filter((f: any) => f && f.hidden)
  const rows = (showHidden ? visible.concat(hidden) : visible)
    .filter((f: any) => removed.indexOf(String((f && f.id) || "")) < 0)
  // 正在上传、还没入库的文件（选完就显示，带上传进度）
  const uploads = (props.pending || []).filter((p: any) => p && p.slot === props.slot)
  const jobs = props.jobs || []
  const jobOf = (src: any) => {
    const id = String((src && src.job_id) || "")
    return id ? (jobs.find((j: any) => String(j.job_id) === id) || null) : null
  }
  const isDone = (src: any) => Boolean(src) && (src.status === "done" || (!src.status && (src.clips || []).length > 0))
  const isProcessing = (src: any) => Boolean(src) && src.status === "processing"
  const isQueued = (src: any) => Boolean(src) && src.status === "queued"

  const act = async (sourceId: string, action: string) => {
    if (!sourceId || busy) return
    // 删除：先本地收起来，立刻有反馈，也不会被连点
    if (action === "delete") setRemoved((prev) => prev.concat([sourceId]))
    setBusy(sourceId + ":" + action)
    setErrMsg("")
    try {
      const r: any = await callEntry(props.api, "ms_corpus_source",
        { source_id: sourceId, action }, 120000)
      if (r && r.ok === false) {
        if (action === "delete") setRemoved((prev) => prev.filter((x) => x !== sourceId))
        setErrMsg(String((r && r.error) || "操作没成功"))
        return
      }
      if (action === "delete" && preview && preview.id === sourceId) setPreview(null)
      props.onChanged()
    } catch (e: any) {
      // 失败时把行放回去，保持原样
      if (action === "delete") setRemoved((prev) => prev.filter((x) => x !== sourceId))
      setErrMsg(String((e && e.message) || e))
    } finally {
      setBusy("")
    }
  }

  const openPreview = async (f: any) => {
    const id = String(f && f.id)
    if (!id) return
    setBusy(id + ":preview")
    try {
      const r: any = await callEntry(props.api, "ms_corpus_source",
        { source_id: id, action: "preview" }, 240000)
      const src = audioSrcOf(r)
      if (src) setPreview({ id, src })
    } catch (e) {
      /* 读不出来就没有预览 */
    } finally {
      setBusy("")
    }
  }

  const head = (
    <Row style={{ justifyContent: "space-between", gap: 8, flexWrap: "wrap" }}>
      <Muted>
        {visible.length + uploads.length} 个文件
        {uploads.length ? " · 上传中 " + uploads.length : ""}
        {props.validS !== undefined ? " · 可用 " + fmtTime(props.validS) : ""}
        {hidden.length ? " · 已隐藏 " + hidden.length : ""}
      </Muted>
      <Row style={{ gap: 6, flexWrap: "wrap", justifyContent: "flex-end" }}>
        {hidden.length ? (
          <Btn onClick={() => setShowHidden(!showHidden)}>
            {showHidden ? "收起已隐藏" : "显示已隐藏 " + hidden.length}
          </Btn>
        ) : null}
        {all.length ? <Btn onClick={() => setOpen(!open)}>{open ? "收起" : "展开"}</Btn> : null}
      </Row>
    </Row>
  )
  return (
    <Col style={{ gap: 6 }}>
      {head}
      {errMsg ? <Muted style={{ color: "#b23b3b" }}>{errMsg}</Muted> : null}
      {open && (rows.length || uploads.length) ? (
        // 固定最大高度 + 内部滚动：素材多了只滚动这块，不会把整张卡片越拉越长
        <div style={{ maxHeight: 268, overflowY: "auto", overflowX: "hidden", paddingRight: 4, display: "flex", flexDirection: "column", gap: 6 }}>
          {/* 正在上传的文件：选完立刻出现，带自己的上传进度 */}
          {uploads.map((p: any) => (
            <div key={p.id} style={{ padding: "8px 10px", borderRadius: 12, background: "rgba(248,244,248,0.9)", border: "1px solid " + LINE }}>
              <Row style={{ gap: 8, alignItems: "center", minWidth: 0 }}>
                <span style={{ width: 22, height: 22, borderRadius: 7, background: "rgba(224,45,118,0.10)", color: "#c2185b", display: "grid", placeItems: "center", fontSize: 11, flex: "0 0 auto" }}>⇧</span>
                <div title={p.name} style={{ minWidth: 0, flex: 1, fontSize: 12.5, color: INK, lineHeight: "18px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {p.name}
                </div>
                <span style={{ fontSize: 11, color: "#b06a00", background: "#fff4e2", borderRadius: 999, padding: "1px 8px", flex: "0 0 auto" }}>
                  {p.state === "upload" ? "上传中" : "排队中"}
                </span>
              </Row>
              <Row style={{ gap: 8, alignItems: "center", marginTop: 6 }}>
                <Bar pct={p.state === "upload" ? p.pct : 100} />
                <Muted>{p.state === "upload" ? p.pct + "%" : "等待处理"}</Muted>
              </Row>
            </div>
          ))}
          {rows.map((f: any, i: number) => {
            const id = String((f && f.id) || "")
            const isHidden = Boolean(f && f.hidden)
            const done = isDone(f) && !isHidden
            const proc = isProcessing(f) && !isHidden
            const queued = isQueued(f) && !isHidden
            const job: any = jobOf(f)
            // 优先用后端随列表带回来的实时进度（恢复/续跑的素材前端没有本地 job）
            const livePct = (f && typeof f.progress === "number") ? f.progress : ((job && job.pct) || 0)
            const pct = proc ? Math.max(0, Math.min(100, Number(livePct) || 0)) : 0
            const stage = proc ? String((f && f.stage) || "") : ""
            const tone = done ? "#1c8a58" : proc ? "#b06a00" : "#c2185b"
            return (
              <div key={id || "f" + i}>
                <div style={{
                  padding: "8px 10px", borderRadius: 12,
                  background: done ? "rgba(232,247,239,0.92)" : isHidden ? "rgba(244,244,247,0.9)" : "rgba(248,244,248,0.9)",
                  border: "1px solid " + (done ? "rgba(31,157,99,0.35)" : LINE),
                }}>
                  {/* 名字单独一行：卡片很窄，和三个按钮挤在一行会被截成「文…」 */}
                  <Row style={{ gap: 8, alignItems: "center", minWidth: 0 }}>
                    <span style={{ width: 22, height: 22, borderRadius: 7, background: done ? "rgba(31,157,99,0.14)" : "rgba(224,45,118,0.10)", color: tone, display: "grid", placeItems: "center", fontSize: 11, flex: "0 0 auto" }}>
                      {done ? "✓" : isHidden ? "○" : props.slot === "song" ? "♪" : props.slot === "dry_sing" ? "◍" : "⌨"}
                    </span>
                    <div title={nameOf(f, i)} style={{ minWidth: 0, flex: 1, fontSize: 12.5, color: INK, lineHeight: "18px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {nameOf(f, i)}
                    </div>
                    {/* 右上角状态：处理完 → 绿色「完成」 */}
                    {done ? (
                      <span style={{ fontSize: 11, color: "#1c8a58", background: "rgba(31,157,99,0.14)", borderRadius: 999, padding: "1px 9px", fontWeight: 600, flex: "0 0 auto" }}>完成</span>
                    ) : proc ? (
                      <span style={{ fontSize: 11, color: "#b06a00", background: "#fff4e2", borderRadius: 999, padding: "1px 9px", flex: "0 0 auto" }}>处理中</span>
                    ) : queued ? (
                      <span style={{ fontSize: 11, color: "#6b6b76", background: "rgba(40,37,50,0.06)", borderRadius: 999, padding: "1px 9px", flex: "0 0 auto" }}>排队中</span>
                    ) : null}
                  </Row>
                  {/* 每个文件自己的处理进度条 */}
                  {proc ? (
                    <Row style={{ gap: 8, alignItems: "center", marginTop: 6 }}>
                      <Bar pct={pct} />
                      <Muted>{pct}%{stage ? " · " + stage.slice(0, 28) : ""}</Muted>
                    </Row>
                  ) : null}
                  <Row style={{ gap: 6, justifyContent: "space-between", marginTop: 6, flexWrap: "wrap" }}>
                    <Muted>{isHidden ? "已隐藏" : metaOf(f)}</Muted>
                    <Row style={{ gap: 4, flexWrap: "wrap", justifyContent: "flex-end" }}>
                      {!isHidden ? (
                        <Btn disabled={Boolean(busy)} onClick={() => openPreview(f)}>
                          {busy === id + ":preview" ? "读取中…" : "试听"}
                        </Btn>
                      ) : null}
                      <Btn disabled={Boolean(busy)} onClick={() => act(id, isHidden ? "show" : "hide")}>
                        {isHidden ? "显示" : "隐藏"}
                      </Btn>
                      <Btn style={DANGER} disabled={Boolean(busy)} onClick={() => act(id, "delete")}>删除</Btn>
                    </Row>
                  </Row>
                </div>
                {preview && preview.id === id ? (
                  <div style={{ marginTop: 6 }}>
                    {/* 点「试听」就直接播：preload=metadata 才会去取数据，autoPlay 负责开头 */}
                    <audio key={preview.id} controls autoPlay preload="metadata" src={preview.src}
                           style={{ width: "100%", height: 30 }} />
                  </div>
                ) : null}
              </div>
            )
          })}
        </div>
      ) : null}
    </Col>
  )
}
