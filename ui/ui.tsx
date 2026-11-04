// 为她而唱 · 基础控件（2026-11-03）
//
// 只用宿主实证过的组件（Button / StatusBadge / Alert / Modal / Card / Stack），
// 其余小东西用原生元素 + 内联样式，避免依赖未验证的 kit 导出。
//
// 2026-11-03 第二版：参考「青于蓝 / 灵境 demo52」的玻璃质感重新调色。
//   - 底色是淡粉雾（#F8F4F8 → #FDF6F9），不再是冷灰；
//   - 主色洋红 #E02D76，浅容器 #FFE5EF，深字 #9C204E；
//   - 卡片 = 白→极淡粉渐变 + 1px 近白描边 + 外阴影（0 7px 24px / 6%）+ 顶部一道细高光，
//     悬停时卡片微微抬起（-3px）阴影加深；按下时落回（和 demo52 的 LiftSurface 一致）；
//   - 输入框改成「填充、无边框」的一体式（和 mkw_theme 的 inputDecorationTheme 一致）；
//   - 所有标签/数值统一用 CSS 变量，方便以后整体换主色。
import { Alert, Button, StatusBadge, useEffect, useRef, useState } from "@neko/plugin-ui"

export type Tone = "success" | "warning" | "error" | "neutral"

// ---- 设计令牌（全站唯一来源） ------------------------------------------------
export const BRAND = "#e02d76"          // 主色洋红
export const BRAND_SOFT = "#ffe5ef"     // 主色浅容器
export const BRAND_DEEP = "#9c204e"     // 浅容器上的深字
export const INK = "#282532"            // 正文
export const INK_2 = "#74717e"          // 次要文字
export const INK_3 = "#9a97a4"          // 更弱
export const LINE = "rgba(40,37,50,0.06)"

// 2026-11-04 重做：底色按方案改成浅灰蓝 #F8FAFC，卡片圆角 26（窄屏 23），
// 其余（主色、阴影、悬停上浮、进度条 7px/圆角 5）沿用参考主题 FrostedSection。
export const PAGE_BG = "#f8fafc"
export const CARD_BG = "#ffffff"
export const CARD_R = 26
export const CARD_R_NARROW = 23
export const CARD_PAD = 24
export const CARD_GAP = 24
export const CARD_SHADOW = "0 7px 24px rgba(142,159,187,0.06)"
export const CARD_SHADOW_HOVER = "0 14px 34px rgba(133,153,189,0.16)"

/** 页面底色：浅蓝灰 + 两团极淡的洋红/湖蓝光晕，铺在 panel.tsx 的最外层。 */
export const PAGE_SKIN = {
  backgroundColor: PAGE_BG,
  backgroundImage: [
    "radial-gradient(900px 420px at 8% -6%, rgba(224,45,118,0.06), rgba(224,45,118,0) 70%)",
    "radial-gradient(760px 380px at 96% 0%, rgba(112,151,188,0.10), rgba(112,151,188,0) 72%)",
  ].join(","),
  backgroundAttachment: "fixed",
}

/** 卡片底色：白 → 极淡肤色，左上到右下。 */
export const CARD_FILL = {
  backgroundImage: "linear-gradient(150deg,#ffffff 0%,#fffdfe 46%,#fdf7fa 100%)",
}

export const FONT = "'Microsoft YaHei UI','Microsoft YaHei',PingFang SC,'Noto Sans SC',system-ui,sans-serif"
export const TRANS = "box-shadow .19s ease, transform .19s ease, background-color .19s ease"

/** 页眉：标题 + 一句说明。 */
export function Head(props: { title: string; sub: string }) {
  return (
    <div>
      <div style={{ fontSize: 17, fontWeight: 700, color: INK, letterSpacing: 0.2 }}>{props.title}</div>
      <div style={{ fontSize: 12, color: INK_2, marginTop: 3 }}>{props.sub}</div>
    </div>
  )
}

/** 自绘白卡片：大圆角 + 近白描边 + 顶部高光 + 双层阴影，不用表格元素、不用粗边框。
 *  hover=true 时抬起 -3px、阴影加深（点击类卡片才开）。 */
export function Panel(props: { title?: any; extra?: any; children?: any; style?: any; hover?: boolean; onClick?: any }) {
  const [hover, setHover] = useState(false)
  const [press, setPress] = useState(false)
  const lift = props.hover && hover
  return (
    <div
      onClick={props.onClick}
      onMouseDown={props.hover ? () => setPress(true) : undefined}
      onMouseUp={props.hover ? () => setPress(false) : undefined}
      onMouseEnter={props.hover ? () => setHover(true) : undefined}
      onMouseLeave={props.hover ? () => { setHover(false); setPress(false) } : undefined}
      style={{
        position: "relative",
        ...CARD_FILL,
        borderRadius: CARD_R, padding: CARD_PAD, minWidth: 0,
        border: "1px solid rgba(255,255,255,0.94)",
        boxShadow: lift ? CARD_SHADOW_HOVER : CARD_SHADOW, transition: TRANS,
        transform: press ? "translateY(1px)" : lift ? "translateY(-3px)" : "none",
        cursor: props.onClick ? "pointer" : "default",
        ...(props.style || {}),
      }}
    >
      {/* 顶部一道 1px 高光，模拟玻璃上沿（demo52 GlassReflectionPainter） */}
      <i aria-hidden style={{ position: "absolute", left: CARD_PAD, right: CARD_PAD, top: 0, height: 1, background: "rgba(255,255,255,0.95)", borderRadius: 1, pointerEvents: "none" }} />
      {props.title || props.extra ? (
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, flexWrap: "wrap", marginBottom: props.children ? 14 : 0 }}>
          <div style={{ fontSize: 14.5, fontWeight: 700, color: INK }}>{props.title}</div>
          {props.extra ? <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center" }}>{props.extra}</div> : null}
        </div>
      ) : null}
      {props.children}
    </div>
  )
}

/** 可折叠卡片：默认收起，只露标题；点击标题展开。 */
export function FoldCard(props: { title: string; sub?: any; open: boolean; onToggle: () => void; children?: any }) {
  const [hover, setHover] = useState(false)
  return (
    <div style={{ ...CARD_FILL, borderRadius: CARD_R, border: "1px solid rgba(255,255,255,0.94)", boxShadow: hover ? CARD_SHADOW_HOVER : CARD_SHADOW, transition: TRANS }}>
      <div
        onClick={props.onToggle}
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => { setHover(false) }}
        style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, padding: "18px 24px", cursor: "pointer" }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0 }}>
          <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: 3, background: BRAND, flex: "0 0 auto", boxShadow: "0 0 0 3px rgba(224,45,118,0.12)" }} />
          <span style={{ fontSize: 14.5, fontWeight: 700, color: INK }}>{props.title}</span>
          {props.sub ? <Muted>{props.sub}</Muted> : null}
        </div>
        <span style={{ fontSize: 12.5, color: BRAND, whiteSpace: "nowrap", fontWeight: 500 }}>{props.open ? "收起 ▲" : "查看 ▼"}</span>
      </div>
      {props.open ? <div style={{ padding: "0 24px 24px" }}>{props.children}</div> : null}
    </div>
  )
}

/** 投递用的虚线框（Dropzone）：点击或拖拽文件进来。 */
export function DropZone(props: { onFiles: (files: FileList | null) => void; hint?: string; busy?: boolean; accept?: string; note?: string; multiple?: boolean }) {
  const [over, setOver] = useState(false)
  const input = useRef<any>(null)
  const color = over ? BRAND : "rgba(224,45,118,0.30)"
  return (
    <div
      onClick={() => { if (input.current) input.current.click() }}
      onDragOver={(e: any) => { e.preventDefault(); setOver(true) }}
      onDragLeave={() => setOver(false)}
      onDrop={(e: any) => {
        e.preventDefault()
        setOver(false)
        const f = e.dataTransfer && e.dataTransfer.files
        if (f && f.length) props.onFiles(f)
      }}
      style={{
        border: "1.5px dashed " + color, borderRadius: 14,
        background: over ? "rgba(224,45,118,0.06)" : "rgba(255,229,239,0.32)",
        padding: "22px 16px", textAlign: "center", cursor: "pointer", transition: "border-color .15s, background .15s",
      }}
    >
      <div style={{ width: 40, height: 40, margin: "0 auto", borderRadius: 12, display: "grid", placeItems: "center", background: "#fff", boxShadow: "0 6px 16px rgba(224,45,118,0.14)", fontSize: 18, color: BRAND, lineHeight: 1 }}>☁</div>
      <div style={{ fontSize: 13, color: INK, marginTop: 8, fontWeight: 500 }}>{props.busy ? "正在读取文件…" : (props.hint || "点击或拖拽音频文件至此")}</div>
      <div style={{ fontSize: 11.5, color: INK_3, marginTop: 3 }}>{props.note || "支持 mp3 / wav / flac，可多选"}</div>
      <input ref={input} type="file" accept={props.accept || "audio/*"}
             multiple={props.multiple !== false} style={{ display: "none" }}
             onChange={(e: any) => {
               // FileList 是活对象：先把文件复制成数组，再清空 input，
               // 否则 e.target.value = "" 会把 FileList 一起清掉，onFiles 收到空列表（多选后加不进去）。
               const picked = e.target.files ? Array.prototype.slice.call(e.target.files) : []
               e.target.value = ""
               if (picked.length) props.onFiles(picked)
             }} />
    </div>
  )
}

/** 文件列表的一行：图标 + 名称 + 时长/说明 + 右侧操作。 */
export function FileRow(props: { icon?: string; title: any; meta?: any; right?: any }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 10px", background: "rgba(248,244,248,0.9)", border: "1px solid " + LINE, borderRadius: 12 }}>
      <span style={{ width: 26, height: 26, borderRadius: 8, background: BRAND_SOFT, color: BRAND_DEEP, display: "grid", placeItems: "center", fontSize: 12, flex: "0 0 auto" }}>
        {props.icon || "♪"}
      </span>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 12.5, color: INK, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{props.title}</div>
        {props.meta ? <div style={{ fontSize: 11.5, color: INK_3 }}>{props.meta}</div> : null}
      </div>
      {props.right ? <div style={{ display: "flex", gap: 4, alignItems: "center", flex: "0 0 auto" }}>{props.right}</div> : null}
    </div>
  )
}

/** 自适应网格：窄屏自动变一列；cols 是宽屏时的列数上限。 */
export function Grid(props: { cols?: number; min?: number; gap?: number; children?: any; style?: any }) {
  const cols = Math.max(1, props.cols || 2)
  return (
    <div style={{ display: "grid", gap: props.gap || 20, gridTemplateColumns: "repeat(auto-fit,minmax(" + (props.min || 240) + "px,1fr))", ...(props.style || {}) }}>
      {props.children}
    </div>
  )
}

export function Section(props: { title: string; extra?: any; children?: any; style?: any }) {
  return <Panel title={props.title} extra={props.extra} style={props.style}>{props.children}</Panel>
}

/** 卡片内部的浅色内衬：比白卡片低一档，像嵌进去的一块软垫。 */
export function Soft(props: { children?: any; style?: any }) {
  return (
    <div style={{ padding: 16, borderRadius: 14, background: "rgba(255,246,250,0.86)", border: "1px solid rgba(224,45,118,0.07)", ...(props.style || {}) }}>
      {props.children}
    </div>
  )
}

export function Row(props: { children?: any; style?: any }) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center", ...(props.style || {}) }}>
      {props.children}
    </div>
  )
}

export function Col(props: { children?: any; style?: any }) {
  return <div style={{ display: "flex", flexDirection: "column", gap: 6, ...(props.style || {}) }}>{props.children}</div>
}

export function Muted(props: { children?: any; style?: any }) {
  return <span style={{ fontSize: 12, color: INK_2, ...(props.style || {}) }}>{props.children}</span>
}

export function Err(props: { children?: any }) {
  return <div style={{ fontSize: 12, color: "#d64545", wordBreak: "break-all" }}>{props.children}</div>
}

/** 语义状态点：统一用 StatusBadge，语气由 tone 决定。 */
export function Tag(props: { tone: Tone; children?: any }) {
  return <StatusBadge tone={props.tone}>{props.children}</StatusBadge>
}

export function Pill(props: { children?: any; tone?: Tone }) {
  const tone = props.tone || "neutral"
  const map: any = {
    success: ["#e8f7ef", "#1c8a58"],
    warning: ["#fff3e0", "#b9750f"],
    error: ["#fdecec", "#c93c3c"],
    neutral: [BRAND_SOFT, BRAND_DEEP],
  }
  const c = map[tone] || map.neutral
  return (
    <span style={{ padding: "3px 9px", borderRadius: 999, fontSize: 12, background: c[0], color: c[1], whiteSpace: "nowrap", fontWeight: 500 }}>
      {props.children}
    </span>
  )
}

export function Bar(props: { pct: number; tone?: string }) {
  const pct = Math.max(0, Math.min(100, Number(props.pct) || 0))
  return (
    <div style={{ height: 7, borderRadius: 5, background: "rgba(224,45,118,0.10)", overflow: "hidden", minWidth: 60, flex: 1 }}>
      <i style={{ display: "block", height: "100%", width: pct + "%", background: props.tone || "linear-gradient(90deg,#ff7fb0,#e02d76)", borderRadius: 5, transition: "width .3s" }} />
    </div>
  )
}

/** 进度环：0～100。不足时为橙色，达标转绿，超过推荐值加一颗星。 */
export function Ring(props: { pct: number; size?: number; tone?: string; star?: boolean; label?: any }) {
  const size = props.size || 96
  const stroke = Math.max(6, Math.round(size / 13))
  const pct = Math.max(0, Math.min(100, Number(props.pct) || 0))
  const r = (size - stroke) / 2
  const c = 2 * Math.PI * r
  const color = props.tone || (pct >= 100 ? "#1c8a58" : "#e8912a")
  return (
    <div style={{ position: "relative", width: size, height: size, flex: "0 0 auto" }}>
      <svg width={size} height={size} viewBox={"0 0 " + size + " " + size} style={{ display: "block" }}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(224,45,118,0.10)" strokeWidth={stroke} />
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth={stroke}
                strokeLinecap="round" strokeDasharray={c + " " + c}
                strokeDashoffset={c * (1 - pct / 100)}
                transform={"rotate(-90 " + size / 2 + " " + size / 2 + ")"}
                style={{ transition: "stroke-dashoffset .45s ease, stroke .3s" }} />
      </svg>
      <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", textAlign: "center", lineHeight: 1.2 }}>
        <div>
          <div style={{ fontSize: Math.round(size / 5.2), fontWeight: 700, color: INK }}>
            {Math.round(pct) + "%"}{props.star ? <span style={{ color: "#e8912a", fontSize: Math.round(size / 6) }}> ★</span> : null}
          </div>
          {props.label ? <div style={{ fontSize: 10.5, color: INK_3, marginTop: 2 }}>{props.label}</div> : null}
        </div>
      </div>
    </div>
  )
}

/** 人话电量式开关：纯 CSS，不用字符图标（以前那个「?」就是字符渲染不出来）。 */
export function Toggle(props: { on: boolean; onChange: (v: boolean) => void; disabled?: boolean; label?: any }) {
  const on = !!props.on
  return (
    <button type="button" role="switch" aria-checked={on} disabled={props.disabled}
      onClick={() => { if (!props.disabled) props.onChange(!on) }}
      style={{
        display: "inline-flex", alignItems: "center", gap: 10, border: 0, background: "none",
        padding: 0, cursor: props.disabled ? "not-allowed" : "pointer", fontFamily: FONT,
        fontSize: 12.5, color: on ? BRAND_DEEP : INK_2, opacity: props.disabled ? 0.6 : 1, textAlign: "left",
      }}>
      <span aria-hidden style={{
        position: "relative", width: 36, height: 20, borderRadius: 999, flex: "0 0 auto",
        background: on ? BRAND : "rgba(40,37,50,0.18)", transition: "background-color .19s ease",
        boxShadow: on ? "0 4px 10px rgba(224,45,118,0.25)" : "none",
      }}>
        <span style={{
          position: "absolute", top: 2, left: on ? 18 : 2, width: 16, height: 16, borderRadius: 999,
          background: "#fff", transition: "left .19s ease", boxShadow: "0 1px 3px rgba(40,37,50,0.25)",
        }} />
      </span>
      {props.label}
    </button>
  )
}

export function KV(props: { items: any[][]; cols?: string }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: (props.cols || "104px") + " 1fr", gap: "6px 10px", fontSize: 12.5 }}>
      {props.items.map((it: any[], i: number) => [
        <b key={"k" + i} style={{ fontWeight: 400, color: INK_2 }}>{it[0]}</b>,
        <div key={"v" + i} style={{ wordBreak: "break-all", color: INK }}>{it[1]}</div>,
      ])}
    </div>
  )
}

export function Field(props: { label?: string; children?: any; style?: any }) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: 12, color: INK_2, ...(props.style || {}) }}>
      {props.label ? <span>{props.label}</span> : null}
      {props.children}
    </label>
  )
}

const FIELD_STYLE = {
  border: "1px solid transparent", borderRadius: 12, padding: "9px 14px", fontSize: 13,
  background: "rgba(248,244,248,0.95)", color: INK, minWidth: 0, outline: "none",
  boxShadow: "inset 0 1px 2px rgba(40,37,50,0.04)", fontFamily: FONT,
}

export function Input(props: any) {
  const [focus, setFocus] = useState(false)
  return (
    <input {...props}
      onFocus={(e: any) => { setFocus(true); if (props.onFocus) props.onFocus(e) }}
      onBlur={(e: any) => { setFocus(false); if (props.onBlur) props.onBlur(e) }}
      style={{
        ...FIELD_STYLE, background: focus ? "#fff" : FIELD_STYLE.background,
        boxShadow: focus ? "0 0 0 3px rgba(224,45,118,0.16)" : FIELD_STYLE.boxShadow,
        transition: TRANS,
        ...(props.style || {}),
      }} />
  )
}

export function Select(props: any) {
  return (
    <select {...props} style={{ ...FIELD_STYLE, cursor: "pointer", ...(props.style || {}) }}>
      {props.children}
    </select>
  )
}

export function Btn(props: any) {
  return <Button {...props} />
}

/** 空状态：一句话说明为什么这里是空的、下一步点哪里。 */
export function Empty(props: { children?: any }) {
  return (
    <div style={{ padding: "16px 12px", border: "1px dashed rgba(224,45,118,0.22)", borderRadius: 14, background: "rgba(255,246,250,0.7)", fontSize: 12.5, color: INK_2, textAlign: "center" }}>
      {props.children}
    </div>
  )
}

export function MockNote() {
  return <Alert tone="info">当前显示的是模拟数据（后端 §5 接口未就绪），只用于看界面流程，不代表真实训练结果。</Alert>
}

// ---- 弹窗队列（DialogLayer）-------------------------------------------------
// 方案第 4 节：全局同时只允许一个弹窗。每个弹窗用 `push()` 拿到一个 id，只往队列里追加；
// 队列里只有「队首」那个真的渲染出来，关掉（pop）后下一个才会出现。
// 以前「试听节点」和「是否过拟合」会同时冒出来，就是两处各自直接渲染 Modal 造成的。
let DL_SET: any = null
let DL_SEQ = 0

export function dialogOpen(id: string) {
  if (!DL_SET) return
  DL_SET((list: string[]) => (list.indexOf(id) >= 0 ? list : list.concat([id])))
}

export function dialogClose(id: string) {
  if (!DL_SET) return
  DL_SET((list: string[]) => list.filter((x: string) => x !== id))
}

export function DialogLayer(props: { children?: (id: string) => any }) {
  const [list, setList] = useState<string[]>([])
  useEffect(() => {
    DL_SET = setList
    return () => { if (DL_SET === setList) DL_SET = null }
  }, [])
  const head = list[0]
  if (!head) return null
  return <>{props.children ? props.children(head) : null}</>
}

/** 一层半透明遮罩 + 居中卡片；内容之外的点点击不穿透。 */
function Sheet(props: { title: string; onClose: () => void; children?: any; width?: number }) {
  useEffect(() => {
    const prev = document.body ? document.body.style.overflow : ""
    if (document.body) document.body.style.overflow = "hidden"
    return () => { if (document.body) document.body.style.overflow = prev }
  }, [])
  return (
    <div
      onMouseDown={(e: any) => { if (e.target === e.currentTarget) props.onClose() }}
      style={{
        position: "fixed", inset: 0, zIndex: 1200, display: "flex", alignItems: "center", justifyContent: "center",
        padding: 24, background: "rgba(40,37,50,0.38)", backdropFilter: "blur(3px)", WebkitBackdropFilter: "blur(3px)",
        fontFamily: FONT,
      }}
    >
      <div style={{
        width: "100%", maxWidth: props.width || 460, maxHeight: "88vh", overflow: "auto",
        ...CARD_FILL, borderRadius: 24, border: "1px solid rgba(255,255,255,0.94)",
        boxShadow: "0 24px 60px rgba(40,37,50,0.24)", padding: "22px 24px 24px",
      }}>
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 12, marginBottom: 14 }}>
          <div style={{ fontSize: 15.5, fontWeight: 700, color: INK, lineHeight: 1.4 }}>{props.title}</div>
          <button type="button" aria-label="关闭" onClick={props.onClose} style={{
            border: 0, background: "rgba(40,37,50,0.05)", color: INK_2, width: 26, height: 26, borderRadius: 999,
            cursor: "pointer", fontSize: 14, lineHeight: "26px", flex: "0 0 auto", padding: 0,
          }}>×</button>
        </div>
        {props.children}
      </div>
    </div>
  )
}

/** 队列化弹窗：open 变 true 时排队，同一时刻只有最前面的那个显示。 */
export function QueueDialog(props: {
  open: boolean
  title: string
  children?: any
  okText?: string
  cancelText?: string
  onOk?: () => void
  onCancel?: () => void
  width?: number
}) {
  const idRef = useRef<string>("")
  if (!idRef.current) { DL_SEQ += 1; idRef.current = "dl-" + DL_SEQ }
  const id = idRef.current
  useEffect(() => {
    if (props.open) dialogOpen(id)
    else dialogClose(id)
  }, [props.open])
  useEffect(() => () => dialogClose(id), [])
  return (
    <DialogLayer>
      {(head: string) => head !== id ? null : (
        <Sheet title={props.title} width={props.width} onClose={() => { if (props.onCancel) props.onCancel(); else dialogClose(id) }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <div style={{ fontSize: 13, lineHeight: 1.7, color: INK }}>{props.children}</div>
            <Row style={{ justifyContent: "flex-end" }}>
              {props.onCancel ? <Button onClick={props.onCancel}>{props.cancelText || "取消"}</Button> : null}
              {props.onOk ? <Button style={PRIMARY_BTN} onClick={props.onOk}>{props.okText || "确定"}</Button> : null}
            </Row>
          </div>
        </Sheet>
      )}
    </DialogLayer>
  )
}

export const PRIMARY_BTN = { border: 0, background: BRAND, color: "#fff", borderRadius: 12, boxShadow: "0 6px 16px rgba(224,45,118,0.22)" }

/** 确认弹窗（过拟合询问、危险操作）——走 QueueDialog，永远排队出现。 */
export function Confirm(props: {
  open: boolean
  title: string
  children?: any
  okText?: string
  cancelText?: string
  onOk: () => void
  onCancel: () => void
}) {
  return (
    <QueueDialog open={props.open} title={props.title} okText={props.okText} cancelText={props.cancelText}
                 onOk={props.onOk} onCancel={props.onCancel}>
      {props.children}
    </QueueDialog>
  )
}

/**
 * 面板内试听：<audio preload=none>，数据按需拉取，不拉起任何系统播放器。
 * 2026-11-03：反复打开时音频没声音，是因为 src 换成新地址后浏览器沿用同一 <audio>，
 * 没有重新 load()，播放进度停在旧音频的结尾。这里在 src 变化后显式 load() 一次。
 */
export function AudioBox(props: { src: string; keyId?: string }) {
  const [url, setUrl] = useState(props.src || "")
  const ref = useRef<any>(null)
  useEffect(() => { if (props.src && props.src !== url) setUrl(props.src) }, [props.src])
  if (!url) return <Muted>（这条没有音频地址）</Muted>
  return (
    <audio
      key={props.keyId || url}
      ref={ref}
      controls
      preload="metadata"
      src={url}
      onLoadedMetadata={() => { try { if (ref.current) { ref.current.currentTime = 0; ref.current.play() } } catch (e) {} }}
      style={{ width: "100%", height: 30 }}
    />
  )
}

/**
 * 删除确认：让用户选「只删记录」还是「连本地文件一起删」。
 * 生成物（成品曲 / 训练记录 / 模型 / 切片）统一走这个提示。
 */
export function DeletePrompt(props: {
  open: boolean
  title: string
  message?: any
  recordLabel?: string
  onCancel: () => void
  onRecordOnly: () => void
  onWithFiles: () => void
}) {
  if (!props.open) return null
  return (
    <div
      onMouseDown={(e: any) => { if (e.target === e.currentTarget) props.onCancel() }}
      style={{ position: "fixed", inset: 0, zIndex: 2000, display: "flex", alignItems: "center",
               justifyContent: "center", background: "rgba(40,37,50,0.38)", padding: 24 }}
    >
      <div style={{ width: "100%", maxWidth: 430, background: "#fff", borderRadius: 20,
                    padding: "22px 24px", boxShadow: "0 24px 60px rgba(40,37,50,0.24)" }}>
        <div style={{ fontSize: 15, fontWeight: 700, color: INK }}>{props.title}</div>
        {props.message ? (
          <div style={{ marginTop: 8, fontSize: 12.5, lineHeight: 1.7, color: INK_2 }}>{props.message}</div>
        ) : null}
        <div style={{ marginTop: 16, display: "flex", flexDirection: "column", gap: 8 }}>
          <Btn style={{ background: "#c93c3c", color: "#fff", borderColor: "#c93c3c" }}
               onClick={props.onWithFiles}>连本地文件一起删除</Btn>
          <Btn onClick={props.onRecordOnly}>{props.recordLabel || "只删这条记录（文件保留）"}</Btn>
          <Btn onClick={props.onCancel}>取消</Btn>
        </div>
        <Muted style={{ display: "block", marginTop: 10, fontSize: 11.5 }}>
          删掉的文件会进回收站，能找回来。
        </Muted>
      </div>
    </div>
  )
}

export function fmtTime(seconds: any): string {
  const n = Number(seconds)
  if (!isFinite(n) || n < 0) return "—"
  const total = Math.round(n)
  const m = Math.floor(total / 60)
  const s = total % 60
  return m > 0 ? m + " 分 " + (s < 10 ? "0" : "") + s + " 秒" : s + " 秒"
}

export function fmtEta(seconds: any): string {
  const n = Number(seconds)
  if (!isFinite(n) || n < 0) return "—"
  if (n < 60) return Math.max(1, Math.round(n)) + " 秒"
  if (n < 3600) return Math.round(n / 60) + " 分钟"
  return (n / 3600).toFixed(1) + " 小时"
}
