// 为她而唱 · 准备素材（2026-11-04 方案第 2 节）
//
// 三张卡片直接把「放什么、有什么用」讲清楚；字段名（song / dry_sing / speech）只在代码里出现。
// 顶部三块：已收到多少、够不够训一个音色、想让她唱哪种语言。
// 数据走 materials.ts 里的共享单例（ms_corpus_list / add / job / requirements），调用与原来一致。
import { Alert, StatusBadge, useEffect, useState } from "@neko/plugin-ui"
import { audioSrcOf, callEntry, dataUriOf, uploadModelChunked } from "./api"
import { MATERIALS, useCorpus } from "./materials"
import { BRAND_DEEP, INK, INK_2, LINE, Btn, Col, DropZone, Empty, FoldCard, Grid, Muted, Panel, Pill, PRIMARY_BTN, Ring, Row, Section, Soft, fmtTime } from "./ui"
import { SourceFileList } from "./sourceList"
import { STEPS_CHOICES, clipList, deficitText, effectiveSeconds, materialTitle, num, requirementOf, slotView } from "./trainModel"

const LANGS = ["中文", "日语", "英语", "其他语言"]

/** 投递来源有两种形状：mock 里是文件名字符串；后端是 {files:[{path,name}]} 的来源。都取最后一段文件名。 */
function fileTitle(f: any, i: number): string {
  if (typeof f === "string") return f.split(/[\\/]/).pop() || f
  const inner = f && Array.isArray(f.files) && f.files.length ? f.files[0] : f
  const name = inner && (inner.name || inner.path)
  return name ? String(name).split(/[\\/]/).pop() || String(name) : "文件 " + (i + 1)
}

export function PageMaterials(props: { api: any; onGo: (page: string) => void }) {
  const { api, onGo } = props
  const c = useCorpus(api)
  const [langs, setLangs] = useState<string[]>([])
  const [preview, setPreview] = useState<any>(null)
  const [previewBusy, setPreviewBusy] = useState("")
  const [req, setReq] = useState<any>(null)
  const [importOpen, setImportOpen] = useState(false)
  const [importBusy, setImportBusy] = useState("")
  const [importPct, setImportPct] = useState(0)
  const [importMsg, setImportMsg] = useState<any>(null)

  // 推荐时长按 8000 步的推荐值算（这个数字只在内部用来画进度环，不展示给用户）
  useEffect(() => {
    callEntry(api, "ms_corpus_requirements", { steps: STEPS_CHOICES[1] }, 60000)
      .then((r: any) => setReq(r)).catch(() => {})
  }, [])

  const list = c.list
  const need = requirementOf(req, STEPS_CHOICES[1])
  const have = effectiveSeconds(list, c.countSpeech)
  const minS = need.min
  const recS = need.rec > minS ? need.rec : minS + 60
  const pct = Math.min(100, Math.round((have / (recS || 1)) * 100))
  const clips = clipList(list)
  const files = MATERIALS.reduce((a: number, m: any) => a + slotView(list, m.slot).files, 0)
  const jobsN = c.jobs.length

  const openPreview = async (clip: any) => {
    setPreviewBusy(clip.id)
    try {
      const r: any = await callEntry(api, "ms_corpus_clip", { clip_id: clip.id, action: "preview" }, 120000)
      const src = audioSrcOf(r)
      if (src) setPreview({ id: clip.id, src })
      else c.setNotice({ tone: "error", text: "这段音频暂时读不出来" })
    } catch (e: any) {
      c.setNotice({ tone: "error", text: "这段音频暂时读不出来：" + String((e && e.message) || e) })
    }
    setPreviewBusy("")
  }

  const toggleLang = (t: string) => setLangs(langs.indexOf(t) >= 0 ? langs.filter((x: string) => x !== t) : langs.concat([t]))

  const importVoice = async (files: FileList | null) => {
    const arr: File[] = files ? Array.prototype.slice.call(files) : []
    const f = arr[0]
    if (!f) return
    if (!/\.zip$/i.test(f.name)) {
      setImportMsg({ tone: "error", text: "请选择 .zip 声音包（里面要同时有 model_*.pt 和 config.yaml）" })
      return
    }
    setImportBusy("import")
    setImportPct(0)
    setImportMsg(null)
    try {
      const r: any = await uploadModelChunked(api, f, 600000, setImportPct)
      if (r && r.ok) {
        setImportMsg({ tone: "info", text: "导入成功：「" + (r.name || f.name) + "」已经出现在「我的声音」里了" })
      } else {
        setImportMsg({ tone: "error", text: "导入失败：" + ((r && r.error) || "未知错误") })
      }
    } catch (e: any) {
      setImportMsg({ tone: "error", text: "导入失败：" + String((e && e.message) || e) })
    } finally {
      setImportBusy("")
    }
  }

  return (
    <Col style={{ gap: 24 }}>
      {c.notice ? <Soft style={{ background: c.notice.tone === "error" ? "#fff1f1" : "#f2f7ff", fontSize: 12.5, color: c.notice.tone === "error" ? "#b23b3b" : INK_2 }}>{c.notice.text}</Soft> : null}
      {c.error ? <Soft style={{ background: "#fff1f1", fontSize: 12.5, color: "#b23b3b" }}>素材列表暂时没读到：{c.error}</Soft> : null}

      {/* 总览 */}
      <Panel>
        <div style={{ display: "flex", gap: 24, flexWrap: "wrap", alignItems: "center" }}>
          <Ring pct={pct} size={116} star={have >= recS} label={pct + "%"} />
          <div style={{ flex: "1 1 260px", minWidth: 220, display: "flex", flexDirection: "column", gap: 8 }}>
            <div style={{ fontSize: 15.5, fontWeight: 700, color: INK }}>
              已收到 {files} 个文件{jobsN ? " · " + jobsN + " 个还在处理" : ""}
            </div>
            <div style={{ fontSize: 13, color: INK_2, lineHeight: 1.9 }}>
              已准备 <b style={{ color: BRAND_DEEP }}>{fmtTime(have)}</b> 可用人声
              {" · 至少需要 " + fmtTime(minS)}
              {" · 推荐 " + fmtTime(recS) + " 以上"}
            </div>
            <Row>
              {have >= minS
                ? <StatusBadge tone="success">{have >= recS ? "素材很充足，可以开始训练了" : "已经够训练，再多一点效果更好"}</StatusBadge>
                : <StatusBadge tone="warning">{deficitText(have, minS) + "就够训练"}</StatusBadge>}
            </Row>
          </div>
          <Col style={{ gap: 8, flex: "0 0 auto" }}>
            <Btn style={PRIMARY_BTN} onClick={() => onGo("train")}>去训练声音 →</Btn>
            <Muted>说话录音默认不计入，可在训练页打开。</Muted>
          </Col>
        </div>
      </Panel>

      {/* 语言说明 */}
      <Panel>
        <Col style={{ gap: 10 }}>
          <div style={{ fontSize: 15, fontWeight: 700, color: INK }}>想让她唱哪种语言？</div>
          <div style={{ fontSize: 12.5, color: INK_2, lineHeight: 1.8 }}>
            想让她会唱哪种语言，就投那种语言的素材。只投中文素材，唱出来就主要是中文的咬字和腔调；想要日语、英语，就各放一些对应语言的歌或清唱。
          </div>
          <Row>
            {LANGS.map((t: string) => {
              const on = langs.indexOf(t) >= 0
              return (
                <button key={t} type="button" onClick={() => toggleLang(t)} style={{
                  border: "1px solid " + (on ? "rgba(224,45,118,0.28)" : LINE), borderRadius: 999, padding: "6px 14px", fontSize: 12.5,
                  background: on ? "rgba(255,229,239,0.85)" : "rgba(255,255,255,0.9)", color: on ? BRAND_DEEP : INK_2, cursor: "pointer",
                  fontWeight: on ? 700 : 400, fontFamily: "inherit",
                }}>{on ? "✓ " : ""}{t}</button>
              )
            })}
          </Row>
          <Muted>这里只是帮你记一下打算投哪些语言，不影响处理结果。</Muted>
        </Col>
      </Panel>

      {/* 导入声音：不用训练，直接把现成的声音包放进来 */}
      <FoldCard open={importOpen} onToggle={() => setImportOpen(!importOpen)}
                title="已有现成的声音？直接导入"
                sub={<Muted>不用训练，导入 .zip 声音包</Muted>}>
        <Col style={{ gap: 10 }}>
          <div style={{ fontSize: 12.5, color: INK_2, lineHeight: 1.8 }}>
            把「我的声音 → 导出」生成的声音包，或者别人打包好的 zip 拖进来。包里要有
            <b> model_*.pt</b> 和 <b>config.yaml</b>；导入后直接出现在「我的声音」里，不用再训练。
          </div>
          <DropZone busy={!!importBusy} accept=".zip" multiple={false}
                    hint="点击或拖拽 .zip 声音包至此" note="只接受 .zip 声音包"
                    onFiles={importVoice} />
          {importBusy ? <Muted>正在导入… {importPct}%</Muted> : null}
          {importMsg ? <Alert tone={importMsg.tone === "error" ? "warning" : "info"}>{importMsg.text}</Alert> : null}
          {importMsg && importMsg.tone !== "error" ? (
            <Row><Btn style={PRIMARY_BTN} onClick={() => onGo("models")}>去「我的声音」看看</Btn></Row>
          ) : null}
        </Col>
      </FoldCard>

      {/* 三张素材卡 */}
      <Grid cols={3} min={280} gap={24}>
        {MATERIALS.map((m: any) => {
          const v = slotView(list, m.slot)
          // 进度按「服务端每一条素材的状态」算，不依赖本页的临时 job 记录
          // （切页面/插件重启后本地记录会丢，进度就会假死在 1%）
          const srcs = v.raw_files.filter((f: any) => !(f && f.hidden))
          const doneN = srcs.filter((f: any) => f.status === "done" || (!f.status && (f.clips || []).length > 0)).length
          const activeSrc = srcs.find((f: any) => f.status === "processing")
          const totalN = srcs.length
          const pctN = totalN
            ? Math.min(100, Math.round(((doneN + (activeSrc ? (Number(activeSrc.progress) || 0) / 100 : 0)) / totalN) * 100))
            : 0
          const busyN = Boolean(activeSrc)
          return (
            <Panel key={m.slot} style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <Row style={{ justifyContent: "space-between", gap: 8 }}>
                <div style={{ fontSize: 15, fontWeight: 700, color: INK }}>{m.title}</div>
                <Pill tone={m.slot === "dry_sing" ? "success" : m.slot === "speech" ? "warning" : "neutral"}>{m.ok}</Pill>
              </Row>
              <div style={{ fontSize: 12.5, color: INK_2, lineHeight: 1.8, minHeight: 44 }}>{m.lead}</div>
              <Muted>{m.how}</Muted>
              <DropZone busy={c.busy || busyN} onFiles={(f: any) => c.addFiles(m.slot, f)} hint="把文件拖到这里，或点一下选择" />
              <SourceFileList api={api} slot={m.slot} files={v.raw_files} validS={v.valid_s}
                              pending={c.pending} jobs={c.jobs}
                              onChanged={() => { c.refresh() }} />
              {num((list && list.orphans && list.orphans[m.slot]) || 0) > 0 ? (
                <Row style={{ justifyContent: "flex-end" }}>
                  <Btn onClick={async () => {
                    try {
                      const r: any = await callEntry(api, "ms_corpus_recover", { slot: m.slot }, 300000)
                      await c.refresh()
                      if (r && r.recovered !== undefined) {
                        c.setNotice({
                          tone: "info",
                          text: "已恢复 " + r.recovered + " 个文件"
                            + (r.duplicates ? "，跳过 " + r.duplicates + " 个重复（已移到回收站）" : ""),
                        })
                      }
                    } catch (e) { /* 下次刷新再试 */ }
                  }}>恢复全部 {num(list.orphans[m.slot])} 个未入库文件</Btn>
                </Row>
              ) : null}
              {busyN && totalN > 0 ? (
                <Row style={{ justifyContent: "flex-end" }}>
                  <StatusBadge tone="warning">
                    {"处理中 " + pctN + "% · " + doneN + "/" + totalN}
                  </StatusBadge>
                </Row>
              ) : null}
            </Panel>
          )
        })}
      </Grid>

      {/* 收到的片段 */}
      <Section title="收到的片段" extra={<Muted>一共 {clips.length} 段 · 不合格的会被自动剔除</Muted>}>
        {clips.length === 0 ? (
          <Empty>还没有片段。上面任意一张卡片放进音频后，处理好的片段会出现在这里。</Empty>
        ) : (
          // 片段多了不让整页无限拉长：固定最大高度 + 内部滚动
          <div style={{ maxHeight: 420, overflowY: "auto", overflowX: "hidden", paddingRight: 4,
                        display: "flex", flexDirection: "column", gap: 8 }}>
            {clips.slice().reverse().map((clip: any) => {
              const bad = clip.status === "rejected"
              return (
                <div key={clip.id} style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", padding: "9px 12px", borderRadius: 14, background: bad ? "rgba(255,241,241,0.9)" : "rgba(248,250,252,0.9)", border: "1px solid " + LINE }}>
                  <span style={{ fontSize: 12.5, color: INK, minWidth: 110 }}>{materialTitle(clip.slot)}</span>
                  <Muted>{fmtTime(clip.dur_s)}</Muted>
                  <StatusBadge tone={bad ? "error" : "success"}>{bad ? "已剔除" : "可用"}</StatusBadge>
                  {clip.reason ? <Muted>{clip.reason}</Muted> : null}
                  <div style={{ flex: 1 }} />
                  {!bad ? <Btn onClick={() => openPreview(clip)}>{previewBusy === clip.id ? "读取中…" : "试听"}</Btn> : null}
                  <Btn onClick={() => c.setClip(clip.id, bad ? "restore" : "reject")}>{bad ? "恢复" : "剔除"}</Btn>
                  <Btn style={{ color: "#c93c3c", borderColor: "rgba(201,60,60,0.28)" }}
                       onClick={() => c.setClip(clip.id, "delete")}>删除</Btn>
                </div>
              )
            })}
          </div>
        )}
        {preview ? (
          <div style={{ marginTop: 10, padding: 12, borderRadius: 14, background: "#fff", border: "1px solid " + LINE }}>
            <audio key={preview.id} controls autoPlay src={preview.src} style={{ width: "100%" }} />
          </div>
        ) : null}
      </Section>

      <Muted>提示：同一首歌里如果有合唱、和声，效果会差一些；没有歌声的纯音乐放进来会被直接丢掉，不会算进时长。</Muted>
    </Col>
  )
}
