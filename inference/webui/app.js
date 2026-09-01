/* ============ CNN 推理界面 - 前端逻辑 ============ */
"use strict";

const $ = (id) => document.getElementById(id);

const el = {
  statusChip: $("statusChip"), statusText: $("statusText"),
  modelSelect: $("modelSelect"), modelHint: $("modelHint"),
  dropZone: $("dropZone"), fileInput: $("fileInput"),
  fileList: $("fileList"), predictBtn: $("predictBtn"), clearBtn: $("clearBtn"),
  uploadStatus: $("uploadStatus"),
  summaryRow: $("summaryRow"), sumTotal: $("sumTotal"), sumLty: $("sumLty"),
  sumMiku: $("sumMiku"), sumUnknown: $("sumUnknown"), sumFail: $("sumFail"),
  resultsGrid: $("resultsGrid"), emptyState: $("emptyState"),
  evalFolder: $("evalFolder"), evalModel: $("evalModel"),
  evalBtn: $("evalBtn"), evalStatus: $("evalStatus"),
  evalResult: $("evalResult"), evalEmpty: $("evalEmpty"),
  evalAcc: $("evalAcc"), evalN: $("evalN"), evalClassMetrics: $("evalClassMetrics"),
  evalCm: $("evalCm"), wrongGrid: $("wrongGrid"), toast: $("toast"),
};

let state = {
  models: [],          // 可用模型名列表
  classCn: {},         // 类别中文名
  files: [],           // 待推理文件 [{file, url}]
  predicting: false,
  health: null,
};

/* ---------------- 工具 ---------------- */

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function fmtSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1024 / 1024).toFixed(2) + " MB";
}

function toast(msg, ok = false) {
  el.toast.textContent = msg;
  el.toast.className = "toast" + (ok ? " ok" : "");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.toast.classList.add("hidden"), 3600);
}

/* ---------------- 服务状态 ---------------- */

async function refreshHealth() {
  try {
    const r = await fetch("/api/health");
    if (!r.ok) throw new Error("HTTP " + r.status);
    state.health = await r.json();
    state.models = state.health.models || [];
    state.classCn = state.health.class_cn || {};
    setStatus("online", `服务正常 · ${state.models.length} 个模型 · ORT ${state.health.onnxruntime || "?"}`);
    fillModelSelects();
  } catch (e) {
    setStatus("offline", "无法连接推理服务");
  }
}

function setStatus(cls, text) {
  el.statusChip.className = "status-chip " + cls;
  el.statusText.textContent = text;
}

function fillModelSelects() {
  const opts = state.models.map((m) => {
    const label = m.replace(/^best_/, "").replace(/_/g, " ");
    return `<option value="${esc(m)}">${esc(label)}（${esc(m)}.onnx）</option>`;
  }).join("");
  el.modelSelect.innerHTML = opts;
  el.evalModel.innerHTML = opts;
  // 默认选中 chuyi（当前唯一模型；找不到时回退到第一个）
  const prefer = ["chuyi"].find((m) => state.models.includes(m));
  if (prefer) {
    el.modelSelect.value = prefer;
    el.evalModel.value = prefer;
  }
  updateModelHint();
}

function updateModelHint() {
  const m = el.modelSelect.value;
  if (!m) return;
  const ood = m.includes("amp");
  el.modelHint.innerHTML = ood
    ? "🛡️ 启用三信号 OOD 拒识（能量 / 置信度 / 特征距离）"
    : "推理模式：仅置信度阈值（&lt; 0.60 判未知）";
}

/* ---------------- 文件选择 ---------------- */

function addFiles(fileList) {
  const imgs = [...fileList].filter((f) => f.type.startsWith("image/") ||
    /\.(jpe?g|png|webp|bmp|gif)$/i.test(f.name));
  for (const f of imgs) {
    if (state.files.some((x) => x.file === f)) continue;
    state.files.push({ file: f, url: URL.createObjectURL(f) });
  }
  renderFileList();
}

function renderFileList() {
  el.fileList.innerHTML = state.files.map((f, i) => `
    <div class="file-item">
      <img src="${f.url}" alt="">
      <span class="fname" title="${esc(f.file.name)}">${esc(f.file.name)}</span>
      <span class="fsize">${fmtSize(f.file.size)}</span>
      <span class="fdel" data-i="${i}" title="移除">✕</span>
    </div>`).join("");
  el.fileList.querySelectorAll(".fdel").forEach((d) =>
    d.addEventListener("click", () => removeFile(+d.dataset.i)));
  const n = state.files.length;
  el.predictBtn.disabled = n === 0 || state.predicting;
  el.clearBtn.disabled = n === 0;
  el.uploadStatus.textContent = n ? `已选择 ${n} 张图片，共 ${fmtSize(state.files.reduce((s, x) => s + x.file.size, 0))}` : "";
  el.uploadStatus.classList.remove("err");
  if (n === 0) {
    el.emptyState.classList.remove("hidden");
    el.summaryRow.classList.add("hidden");
  }
}

function removeFile(i) {
  URL.revokeObjectURL(state.files[i].url);
  state.files.splice(i, 1);
  renderFileList();
}

function clearAll() {
  state.files.forEach((f) => URL.revokeObjectURL(f.url));
  state.files = [];
  el.resultsGrid.innerHTML = "";
  el.fileInput.value = "";
  renderFileList();
  el.summaryRow.classList.add("hidden");
  el.emptyState.classList.remove("hidden");
}

/* ---------------- 推理 ---------------- */

async function runPredict() {
  if (!state.files.length || state.predicting) return;
  const model = el.modelSelect.value;
  if (!model) return toast("请先在服务中选择可用模型");

  state.predicting = true;
  el.predictBtn.disabled = true;
  el.predictBtn.classList.add("loading");
  el.predictBtn.querySelector(".btn-icon").textContent = "⟳";
  el.uploadStatus.textContent = `正在推理 ${state.files.length} 张图片…（首次加载模型约需数秒）`;
  el.uploadStatus.classList.remove("err");

  const fd = new FormData();
  state.files.forEach((f) => fd.append("files", f.file));
  fd.append("model", model);

  try {
    const r = await fetch("/api/predict", { method: "POST", body: fd });
    const data = await r.json();
    if (!r.ok || !data.ok) throw new Error(data.error || "推理失败");

    renderResults(data.results);
    renderSummary(data.summary);
    el.uploadStatus.textContent = `完成：${data.summary.total} 张（模型 ${data.model}${data.ood ? "，OOD 拒识已启用" : ""}）`;
    toast("推理完成 ✓", true);
  } catch (e) {
    el.uploadStatus.textContent = "推理失败：" + e.message;
    el.uploadStatus.classList.add("err");
    toast("推理失败：" + e.message);
  } finally {
    state.predicting = false;
    el.predictBtn.classList.remove("loading");
    el.predictBtn.querySelector(".btn-icon").textContent = "▶";
    el.predictBtn.disabled = state.files.length === 0;
  }
}

function renderSummary(s) {
  el.sumTotal.textContent = s.total;
  el.sumLty.textContent = s.lty;
  el.sumMiku.textContent = s.miku;
  el.sumUnknown.textContent = s.unknown;
  el.sumFail.textContent = s.failed;
  el.summaryRow.classList.remove("hidden");
}

function renderResults(results) {
  el.emptyState.classList.add("hidden");
  el.resultsGrid.innerHTML = results.map((r, i) => cardHtml(r, i)).join("");
}

function cardHtml(r, i) {
  const f = state.files[i];
  const url = f ? f.url : "";
  const name = esc(r.name || "(未知)");
  if (!r.ok) {
    return `
    <div class="result-card">
      <div class="rc-img-wrap"><img src="${url}" alt=""></div>
      <div class="rc-body">
        <div class="rc-name" title="${name}">${name}</div>
        <div class="rc-verdict">
          <span class="badge fail">✕ 解析失败</span>
        </div>
        <div class="rc-reason">${esc(r.error || "")}</div>
      </div>
    </div>`;
  }
  const cls = r.rejected ? "unknown" : r.pred;
  const p = r.probs || {};
  const pLty = (p.lty || 0) * 100, pMiku = (p.miku || 0) * 100;
  const bars = `
    <div class="bar-row">
      <div class="bar-label"><span>洛天依 lty</span><span>${pLty.toFixed(1)}%</span></div>
      <div class="bar-track"><div class="bar-fill lty" style="width:${pLty}%"></div></div>
    </div>
    <div class="bar-row">
      <div class="bar-label"><span>初音未来 miku</span><span>${pMiku.toFixed(1)}%</span></div>
      <div class="bar-track"><div class="bar-fill miku" style="width:${pMiku}%"></div></div>
    </div>`;
  return `
  <div class="result-card">
    <div class="rc-img-wrap"><img src="${url}" alt="" loading="lazy"></div>
    <div class="rc-body">
      <div class="rc-name" title="${name}">${name}</div>
      <div class="rc-verdict">
        <span class="badge ${cls}">${r.rejected ? "❓" : r.pred === "lty" ? "🎤" : "🎶"} ${esc(r.pred_cn)}</span>
        <span class="conf-num ${cls}">${(r.conf * 100).toFixed(1)}%</span>
      </div>
      ${r.rejected ? `<div class="rc-reason">⚠️ ${esc(r.reason || "OOD 拒识")}（${esc(r.pred_cn)}）</div>` : ""}
      ${bars}
    </div>
  </div>`;
}

/* ---------------- 准确率验证 ---------------- */

async function runEvaluate() {
  const folder = el.evalFolder.value.trim().replace(/^"|"$/g, "");
  const model = el.evalModel.value;
  if (!folder) return toast("请填写数据集目录");
  if (!model) return toast("请选择模型");

  el.evalBtn.disabled = true;
  el.evalBtn.classList.add("loading");
  el.evalStatus.textContent = "正在批量推理…（数据集越大耗时越长）";
  el.evalStatus.classList.remove("err");

  const fd = new FormData();
  fd.append("folder", folder);
  fd.append("model", model);

  try {
    const r = await fetch("/api/evaluate", { method: "POST", body: fd });
    const data = await r.json();
    if (!r.ok || !data.ok) throw new Error(data.error || "验证失败");

    el.evalEmpty.classList.add("hidden");
    el.evalResult.classList.remove("hidden");
    el.evalAcc.textContent = (data.accuracy * 100).toFixed(2) + "%";
    el.evalN.textContent = data.n;
    el.evalCm.src = data.confusion;
    el.evalStatus.textContent = `完成：${data.folder} · 共 ${data.n} 张（模型 ${model}）`;
    el.evalStatus.classList.remove("err");

    el.evalClassMetrics.innerHTML = Object.entries(data.classes || {}).map(([c, d]) => {
      const cn = state.classCn[c] || c;
      const color = c === "lty" ? "var(--lty)" : "var(--miku)";
      return `<div class="metric-card">
        <div class="metric-label" style="color:${color}">${esc(cn)} (${esc(c)})</div>
        <div class="metric-value" style="color:${color}">P ${(d.precision * 100).toFixed(1)}%</div>
        <div class="metric-sub">recall ${(d.recall * 100).toFixed(1)}% · F1 ${(d.f1 * 100).toFixed(1)}% · support ${d.support}</div>
      </div>`;
    }).join("");

    el.wrongGrid.innerHTML = (data.wrong || []).map((w) => `
      <div class="wrong-item">
        <img src="${w.img}" alt="" loading="lazy">
        <div class="w-info">
          <span class="wt">真: ${esc(state.classCn[w.true] || w.true)}</span>
          <span class="wp">预测: ${esc(state.classCn[w.pred] || w.pred)} · 置信度 ${(w.conf * 100).toFixed(1)}%</span>
        </div>
      </div>`).join("") || `<p style="color:var(--text-dim);font-size:13px;padding:10px 0">🎉 没有错误样本！</p>`;
    toast("验证完成 ✓", true);
  } catch (e) {
    el.evalStatus.textContent = "验证失败：" + e.message;
    el.evalStatus.classList.add("err");
    toast("验证失败：" + e.message);
  } finally {
    el.evalBtn.disabled = false;
    el.evalBtn.classList.remove("loading");
  }
}

/* ---------------- 事件绑定 ---------------- */

el.dropZone.addEventListener("click", () => el.fileInput.click());
el.dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.fileInput.click(); }
});
el.fileInput.addEventListener("change", () => { addFiles(el.fileInput.files); el.fileInput.value = ""; });

["dragenter", "dragover"].forEach((ev) =>
  el.dropZone.addEventListener(ev, (e) => {
    e.preventDefault();
    el.dropZone.classList.add("dragover");
  }));
["dragleave", "drop"].forEach((ev) =>
  el.dropZone.addEventListener(ev, (e) => {
    e.preventDefault();
    el.dropZone.classList.remove("dragover");
  }));
el.dropZone.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));

el.predictBtn.addEventListener("click", runPredict);
el.clearBtn.addEventListener("click", clearAll);
el.modelSelect.addEventListener("change", updateModelHint);
el.evalBtn.addEventListener("click", runEvaluate);

document.querySelectorAll(".tab").forEach((t) =>
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".tab-body").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    $(`tab-${t.dataset.tab}`).classList.add("active");
  }));

/* 防止浏览器直接打开被拖入的图片 */
["dragover", "drop"].forEach((ev) =>
  window.addEventListener(ev, (e) => e.preventDefault()));

/* 默认数据集目录（服务器同级的 dataset/test） */
refreshHealth().then(() => {
  if (state.health && state.health.default_folder) {
    el.evalFolder.value = state.health.default_folder;
  }
});
