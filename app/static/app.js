const $ = (sel) => document.querySelector(sel);

const EXAMPLES = [
  "自然人客户现金交易达到多少需要报告大额交易？",
  "和关联自然人的交易超过多少钱要上董事会？",
  "发生个人信息泄露后，多长时间内必须向监管部门报告？",
  "2025 年公司营业收入是多少，同比增长多少？",
  "去上海出差住酒店一晚的报销上限是多少？",
];
const OUT_OF_SCOPE = "公司的员工年假有多少天？";

const NODE_NAMES = {
  retrieve: "retrieve · 混合检索",
  grade_documents: "grade_documents · 相关性评估",
  transform_query: "transform_query · 查询改写",
  generate: "generate · 生成答案",
  grade_generation: "grade_generation · 幻觉检测",
  fallback: "fallback · 拒答兜底",
};

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ---------- status & knowledge base ---------- */
async function loadStatus() {
  try {
    const h = await (await fetch("/api/v1/health")).json();
    const pills = [];
    pills.push(h.llm_enabled
      ? `<span class="pill ok"><span class="dot"></span>LLM · ${esc(h.llm_model)}</span>`
      : `<span class="pill warn" title="未配置 LLM_API_KEY：答案由原文抽取生成"><span class="dot"></span>离线抽取模式</span>`);
    pills.push(`<span class="pill">Embedding · ${esc(h.embedding_model)}</span>`);
    pills.push(`<span class="pill">Reranker · ${esc(h.reranker_model || "off")}</span>`);
    pills.push(`<span class="pill">${h.chunks} chunks</span>`);
    $("#status").innerHTML = pills.join("");
  } catch {
    $("#status").innerHTML = `<span class="pill bad"><span class="dot"></span>后端不可用</span>`;
  }
}

async function loadDocs() {
  const docs = await (await fetch("/api/v1/documents")).json();
  $("#kb").innerHTML = docs.map((d) =>
    `<li><span class="kb-title" title="${esc(d.source)}">${esc(d.title)}</span><span class="kb-n">${d.chunks} 段</span></li>`
  ).join("");
}

$("#upload").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const msg = $("#upload-msg");
  msg.textContent = `正在上传并重建索引：${file.name} …`;
  const fd = new FormData();
  fd.append("file", file);
  try {
    const r = await fetch("/api/v1/documents", { method: "POST", body: fd });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    msg.textContent = `已加入 ${data.source}：${data.chunks} 个片段（总计 ${data.total_chunks}）。`;
    await Promise.all([loadDocs(), loadStatus()]);
  } catch (err) {
    msg.textContent = `上传失败：${err.message}`;
  }
  e.target.value = "";
});

/* ---------- examples ---------- */
$("#examples").innerHTML =
  EXAMPLES.map((q) => `<button type="button" class="chip">${esc(q)}</button>`).join("") +
  `<button type="button" class="chip oos" title="知识库里没有答案，用来演示拒答">${esc(OUT_OF_SCOPE)}</button>`;
$("#examples").addEventListener("click", (e) => {
  if (e.target.classList.contains("chip")) {
    $("#question").value = e.target.textContent;
    ask();
  }
});

/* ---------- graph highlighting ---------- */
let path = [];
function resetGraph() {
  path = ["START"];
  document.querySelectorAll("#graph .node").forEach((n) => n.classList.remove("visited", "active", "warn"));
  document.querySelectorAll("#graph .edges path").forEach((p) => p.classList.remove("on"));
}
function markNode(name) {
  const prev = path[path.length - 1];
  path.push(name);
  document.querySelectorAll("#graph .node.active").forEach((n) => n.classList.replace("active", "visited"));
  const el = document.querySelector(`#graph .node[data-node="${name}"]`);
  if (el) el.classList.add(name === "fallback" ? "warn" : "active");
  const edge = document.querySelector(`#graph path[data-edge="${prev}>${name}"]`);
  if (edge) edge.classList.add("on");
}
function finishGraph() {
  const prev = path[path.length - 1];
  const edge = document.querySelector(`#graph path[data-edge="${prev}>END"]`);
  if (edge) edge.classList.add("on");
  document.querySelectorAll("#graph .node.active").forEach((n) => n.classList.replace("active", "visited"));
}

/* ---------- timeline ---------- */
function addStep(step) {
  const tl = $("#timeline");
  tl.querySelector(".placeholder")?.remove();
  const li = document.createElement("li");
  if (step.node === "fallback") li.className = "fallback";
  li.innerHTML = `
    <div class="step-head"><span class="step-name">${esc(NODE_NAMES[step.node] || step.node)}</span>
    <span class="step-ms">${step.ms.toFixed(0)} ms</span></div>
    <div class="step-detail">${esc(step.detail)}</div>`;
  tl.appendChild(li);
}

/* ---------- answer rendering ---------- */
function renderAnswer(text) {
  const inline = (s) =>
    esc(s)
      .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
      .replace(/\[(\d+)\]/g, '<button type="button" class="cite" data-cite="$1">$1</button>');
  const out = [];
  let list = [];
  const flush = () => { if (list.length) { out.push(`<ul>${list.join("")}</ul>`); list = []; } };
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line) { flush(); continue; }
    const m = line.match(/^(?:[-*•]|\d+[.、)])\s*(.*)$/);
    if (m) list.push(`<li>${inline(m[1])}</li>`);
    else { flush(); out.push(`<p>${inline(line.replace(/^#+\s*/, ""))}</p>`); }
  }
  flush();
  return out.join("");
}

function scoreBar(label, value, max) {
  if (value === null || value === undefined) return `<span class="score">${label} –</span>`;
  const pct = Math.max(4, Math.min(100, (value / max) * 100));
  return `<span class="score">${label} <span class="bar"><span style="width:${pct}%"></span></span> ${value}</span>`;
}

function renderSource(s, showRelevance) {
  const sc = s.scores || {};
  const rel = showRelevance
    ? (s.relevant ? '<span class="tag yes">相关</span>' : '<span class="tag no">已过滤</span>')
    : "";
  return `
    <div class="source" id="${showRelevance ? "cand" : "src"}-${s.index}">
      <div class="source-head">
        <span class="source-idx">${showRelevance ? "#" : ""}${s.index}</span>
        <span class="source-title">${esc(s.title)}</span>
        <span class="source-sec">${esc(s.section || "")}</span>
        ${rel}
      </div>
      <p class="source-body">${esc(s.content)}</p>
      <div class="scores">
        <span class="score">向量排名 ${sc.vector_rank ?? "–"}</span>
        <span class="score">BM25 排名 ${sc.bm25_rank ?? "–"}</span>
        ${scoreBar("RRF", sc.rrf_score, 0.034)}
        ${scoreBar("Rerank", sc.rerank_score, 1)}
      </div>
    </div>`;
}

function renderResult(r) {
  finishGraph();
  $("#empty-state").hidden = true;
  $("#answer-card").hidden = false;
  const body = $("#answer");
  body.innerHTML = renderAnswer(r.answer);
  if (r.mode === "fallback") body.classList.add("fallback-note");
  else body.classList.remove("fallback-note");

  const m = [];
  m.push(`<span class="pill">${(r.latency_ms / 1000).toFixed(2)} s</span>`);
  if (r.mode === "llm") m.push(`<span class="pill ok">LLM 生成</span>`);
  if (r.mode === "offline") m.push(`<span class="pill warn">抽取式</span>`);
  if (r.mode === "fallback") m.push(`<span class="pill warn">已拒答</span>`);
  if (r.groundedness !== null && r.groundedness !== undefined) {
    const cls = r.grounded ? "ok" : "bad";
    m.push(`<span class="pill ${cls}" title="答案句子被来源支撑的比例">Groundedness ${r.groundedness.toFixed(2)}</span>`);
  }
  $("#metrics").innerHTML = m.join("");

  const rw = $("#rewrite");
  rw.hidden = !r.rewritten_query;
  if (r.rewritten_query) rw.textContent = `首轮检索不相关，Agent 将查询改写为：「${r.rewritten_query}」`;

  const relevantKeys = new Set(r.sources.map((s) => s.content));
  const cands = r.candidates.map((c) => ({ ...c, relevant: relevantKeys.has(c.content) }));
  $("#sources-card").hidden = false;
  $("#cited-count").textContent = r.sources.length;
  $("#cand-count").textContent = cands.length;
  $("#cited").innerHTML = r.sources.length
    ? r.sources.map((s) => renderSource(s, false)).join("")
    : `<p class="hint">没有可引用的来源。</p>`;
  $("#candidates").innerHTML = cands.map((s) => renderSource(s, true)).join("");
}

$("#answer").addEventListener("click", (e) => {
  const n = e.target.dataset?.cite;
  if (!n) return;
  switchTab("cited");
  const el = document.getElementById(`src-${n}`);
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.add("flash");
  setTimeout(() => el.classList.remove("flash"), 1400);
});

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  $("#cited").hidden = name !== "cited";
  $("#candidates").hidden = name !== "candidates";
}
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));

/* ---------- ask (SSE stream) ---------- */
let busy = false;
async function ask() {
  const question = $("#question").value.trim();
  if (!question || busy) return;
  busy = true;
  history.replaceState(null, "", `?q=${encodeURIComponent(question)}`);
  const btn = $("#ask-btn");
  btn.disabled = true;
  btn.classList.add("loading");
  $("#timeline").innerHTML = "";
  resetGraph();

  try {
    const resp = await fetch("/api/v1/query/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        if (!chunk.startsWith("data: ")) continue;
        const evt = JSON.parse(chunk.slice(6));
        if (evt.type === "step") { markNode(evt.step.node); addStep(evt.step); }
        else if (evt.type === "result") renderResult(evt.result);
        else if (evt.type === "error") throw new Error(evt.error);
      }
    }
  } catch (err) {
    $("#empty-state").hidden = true;
    $("#answer-card").hidden = false;
    $("#answer").innerHTML = `<p class="fallback-note">请求失败：${esc(err.message)}</p>`;
    $("#metrics").innerHTML = "";
  } finally {
    busy = false;
    btn.disabled = false;
    btn.classList.remove("loading");
  }
}

$("#ask-form").addEventListener("submit", (e) => { e.preventDefault(); ask(); });
$("#question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); ask(); }
});

loadStatus();
loadDocs();

// Shareable links: /?q=问题 runs the question on load.
const initialQ = new URLSearchParams(location.search).get("q");
if (initialQ) {
  $("#question").value = initialQ;
  ask();
}
