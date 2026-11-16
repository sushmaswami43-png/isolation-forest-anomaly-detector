"use strict";

const $ = (id) => document.getElementById(id);
const BLUE = "#2a6fdb", RED = "#d62728";
const DEBOUNCE_MS = 300;

const state = { file: null, result: null, features: null, controller: null, timer: null };

// ---------- helpers ----------
function setStatus(text, kind = "") {
  const el = $("status");
  el.textContent = text;
  el.className = "status " + kind;
}

function cell(tag, text) {          // textContent only: CSV headers/values are untrusted
  const el = document.createElement(tag);
  el.textContent = text;
  return el;
}

function csvEscape(value) {
  const s = String(value);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

// ---------- server communication ----------
async function runDetection() {
  if (!state.file) return;
  if (state.controller) state.controller.abort();      // drop stale in-flight request
  state.controller = new AbortController();

  const form = new FormData();
  form.append("file", state.file);
  form.append("contamination", $("contamination").value);
  form.append("n_estimators", $("n-estimators").value);
  if (state.features) form.append("features", state.features.join(","));

  setStatus("Running Isolation Forest…", "busy");
  try {
    const resp = await fetch("/api/detect", { method: "POST", body: form, signal: state.controller.signal });
    const body = await resp.json();
    if (!resp.ok) throw new Error(body.error || "Request failed.");
    state.result = body;
    renderAll();
  } catch (err) {
    if (err.name === "AbortError") return;
    setStatus(err.message, "error");
  }
}

function scheduleDetection() {
  clearTimeout(state.timer);
  state.timer = setTimeout(runDetection, DEBOUNCE_MS);
}

async function loadExample(id) {
  const resp = await fetch(`/api/examples/${id}`);
  if (!resp.ok) return setStatus("Could not load the example.", "error");
  const blob = await resp.blob();
  setFile(new File([blob], `${id}.csv`, { type: "text/csv" }));
}

function setFile(file) {
  state.file = file;
  state.features = null;          // reset to "all numeric columns" for a new dataset
  state.result = null;
  runDetection();
}

// ---------- rendering ----------
function renderAll() {
  const r = state.result;
  renderControls(r);
  renderSummary(r);
  renderScatter();
  renderHistogram(r);
  renderTable(r);
  const notes = [];
  if (r.dropped_rows) notes.push(`${r.dropped_rows} rows with missing values were dropped`);
  if (r.ignored_columns.length) notes.push(`ignored non-numeric: ${r.ignored_columns.join(", ")}`);
  setStatus(`Done. ${notes.join("; ")}`.trim());
}

function renderControls(r) {
  $("feature-section").hidden = false;
  const list = $("feature-list");
  if (list.dataset.source !== r.columns.join("|")) {    // rebuild only when the dataset changed
    list.dataset.source = r.columns.join("|");
    list.replaceChildren(...r.columns.map((col) => {
      const label = document.createElement("label");
      const box = document.createElement("input");
      box.type = "checkbox"; box.value = col; box.checked = true;
      label.append(box, document.createTextNode(col));
      return label;
    }));
    for (const id of ["x-axis", "y-axis"]) {
      $(id).replaceChildren(...r.columns.map((c) => new Option(c, c)));
    }
    $("x-axis").value = r.columns[0];
    $("y-axis").value = r.columns[1];
  }
}

function renderSummary(r) {
  const stats = [
    [r.n_rows, "rows"], [r.n_anomalies, "anomalies"],
    [(100 * r.n_anomalies / r.n_rows).toFixed(1) + "%", "flagged"],
    [r.threshold.toFixed(3), "score threshold"],
  ];
  $("summary").replaceChildren(...stats.map(([v, l]) => {
    const div = document.createElement("div");
    div.className = "stat";
    div.append(cell("b", v), cell("span", l));
    return div;
  }));
  $("summary").hidden = false;
}

function pick(arr, mask, wanted) {
  return arr.filter((_, i) => mask[i] === wanted);
}

function renderScatter() {
  const r = state.result;
  const xName = $("x-axis").value, yName = $("y-axis").value;
  const x = r.data[xName], y = r.data[yName];
  const idx = x.map((_, i) => i);
  const trace = (wanted, name, color, symbol, size) => ({
    type: x.length > 5000 ? "scattergl" : "scatter", mode: "markers", name,
    x: pick(x, r.labels, wanted), y: pick(y, r.labels, wanted),
    text: pick(idx, r.labels, wanted).map((i) => `row ${i}<br>score ${r.scores[i].toFixed(3)}`),
    hovertemplate: "%{text}<br>%{x}, %{y}<extra></extra>",
    marker: { color, symbol, size, opacity: wanted ? 0.95 : 0.55 },
  });
  Plotly.react("scatter", [trace(0, "Normal", BLUE, "circle", 6), trace(1, "Anomaly", RED, "diamond", 9)], {
    title: "Detected anomalies", xaxis: { title: xName }, yaxis: { title: yName },
    margin: { t: 40, r: 10 }, legend: { orientation: "h", y: -0.2 },
  }, { responsive: true });
}

function renderHistogram(r) {
  Plotly.react("histogram", [{
    type: "histogram", x: r.scores, nbinsx: 40, marker: { color: BLUE }, name: "Rows",
  }], {
    title: "Anomaly score distribution", xaxis: { title: "Anomaly score (higher = more anomalous)" },
    yaxis: { title: "Rows" }, margin: { t: 40, r: 10 }, showlegend: false,
    shapes: [{ type: "line", x0: r.threshold, x1: r.threshold, yref: "paper", y0: 0, y1: 1,
               line: { color: RED, dash: "dash", width: 2 } }],
    annotations: [{ x: r.threshold, yref: "paper", y: 1, text: "threshold", showarrow: false,
                    xanchor: "left", font: { color: RED } }],
  }, { responsive: true });
}

function renderTable(r) {
  const top = r.scores.map((s, i) => [s, i]).filter(([, i]) => r.labels[i])
    .sort((a, b) => b[0] - a[0]).slice(0, 10);
  const table = $("anomaly-table");
  const head = document.createElement("tr");
  ["row", "score", ...r.columns].forEach((h) => head.append(cell("th", h)));
  table.replaceChildren(head, ...top.map(([score, i]) => {
    const tr = document.createElement("tr");
    [i, score.toFixed(3), ...r.columns.map((c) => r.data[c][i])].forEach((v) => tr.append(cell("td", v)));
    return tr;
  }));
  $("table-section").hidden = false;
}

function downloadResults() {
  const r = state.result;
  const lines = [[...r.columns, "anomaly_score", "is_anomaly"].map(csvEscape).join(",")];
  r.scores.forEach((s, i) => lines.push([...r.columns.map((c) => r.data[c][i]), s, r.labels[i]].map(csvEscape).join(",")));
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
  link.download = "anomaly_results.csv";
  link.click();
  URL.revokeObjectURL(link.href);
}

// ---------- wiring ----------
function bindSlider(id, labelId, format) {
  $(id).addEventListener("input", () => {
    $(labelId).textContent = format($(id).value);
    scheduleDetection();            // updates the plots without a page refresh
  });
}

async function init() {
  bindSlider("contamination", "contamination-value", (v) => Number(v).toFixed(3).replace(/0$/, ""));
  bindSlider("n-estimators", "n-estimators-value", (v) => v);

  $("file-input").addEventListener("change", (e) => e.target.files[0] && setFile(e.target.files[0]));
  $("feature-list").addEventListener("change", () => {
    const chosen = [...document.querySelectorAll("#feature-list input:checked")].map((b) => b.value);
    if (chosen.length < 1) return setStatus("Select at least one feature.", "error");
    state.features = chosen;
    scheduleDetection();
  });
  $("x-axis").addEventListener("change", () => state.result && renderScatter());
  $("y-axis").addEventListener("change", () => state.result && renderScatter());
  $("download-btn").addEventListener("click", downloadResults);

  const select = $("example-select");
  const examples = await (await fetch("/api/examples")).json();
  examples.forEach((ex) => select.append(new Option(ex.id, ex.id)));
  select.addEventListener("change", () => {
    const ex = examples.find((e) => e.id === select.value);
    $("example-desc").textContent = ex ? ex.description : "";
    if (ex) loadExample(ex.id);
  });
  select.value = examples[0].id;          // start with something visible
  select.dispatchEvent(new Event("change"));
}

init();
