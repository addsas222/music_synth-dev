// 运行时类型别名（2026-11-03）：预览/去 TS 环境里 type 声明会被删掉，
// 但代码里仍用 `c: CLIP_TYPE` 这类标注，所以这里用 const 垫一层，保证名字在运行时存在。
// 只给本插件 ui/*.tsx 共用，不导出到宿主。
// 注意：宿主编译器（hosted-ui）不接受“一个 export const 多个声明符”，所以一条一行。
export const CLIP_TYPE = null
export const SLOT_TYPE = null
export const TONE_OF = { song: "neutral", dry_sing: "success", speech: "warning" }
