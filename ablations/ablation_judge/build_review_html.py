"""Build a self-contained HTML reviewer for the judge-accuracy ablation.

Reads the same sample manifest as ``build_review_csv.py`` but emits a single
``review/<dataset>_review.html`` that embeds all rows and a minimal UI:

- One card per item with problem, gold, original answer, repaired answer,
  original/repaired step text, and downstream context.
- Keyboard shortcuts: 1 = correct, 2 = incorrect, 3 = unclear,
  n / → = next, p / ← = prev.
- Labels auto-persist to ``localStorage`` under a per-dataset key.
- "Download labeled CSV" button emits ``<dataset>_labeled.csv`` in the schema
  that ``analyze.py`` expects.

Usage:
    python -m ablations.ablation_judge.build_review_html --dataset sealqa
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pymongo import MongoClient

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from ablations.ablation_judge.build_review_csv import (
    FIELDNAMES,
    build_rows,
)


REVIEW_DIR = Path(__file__).parent / "review"


def get_db(db_name: str) -> Any:
    load_dotenv()
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI not set in .env")
    client: MongoClient = MongoClient(uri, serverSelectionTimeoutMS=10000)
    return client[db_name]


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Judge review — __DATASET__</title>
<style>
  :root {
    --bg: #0e1116;
    --panel: #151a21;
    --panel2: #1c232c;
    --border: #2b3440;
    --fg: #d6dde5;
    --muted: #8b97a7;
    --accent: #6ea8ff;
    --ok: #2ea043;
    --bad: #d1493f;
    --warn: #d4a72c;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.5 -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif;
         background: var(--bg); color: var(--fg); }
  header { padding: 10px 16px; border-bottom: 1px solid var(--border);
           display: flex; gap: 16px; align-items: center; position: sticky;
           top: 0; background: var(--bg); z-index: 10; }
  header h1 { font-size: 14px; margin: 0; font-weight: 600; }
  header .progress { color: var(--muted); }
  header .spacer { flex: 1; }
  button { background: var(--panel2); color: var(--fg); border: 1px solid var(--border);
           padding: 6px 12px; border-radius: 6px; cursor: pointer; font: inherit; }
  button:hover { border-color: var(--accent); }
  button.ok { background: rgba(46, 160, 67, 0.15); border-color: var(--ok); }
  button.bad { background: rgba(209, 73, 63, 0.15); border-color: var(--bad); }
  button.warn { background: rgba(212, 167, 44, 0.15); border-color: var(--warn); }
  .container { max-width: 1100px; margin: 0 auto; padding: 16px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
          padding: 16px; margin-bottom: 16px; }
  .row { display: grid; grid-template-columns: 160px 1fr; gap: 8px 16px; margin-bottom: 8px; }
  .label { color: var(--muted); text-transform: uppercase; font-size: 11px; letter-spacing: 0.05em;
           padding-top: 2px; }
  .value { word-break: break-word; }
  .value.mono { white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                font-size: 12px; background: var(--panel2); padding: 8px 10px; border-radius: 6px;
                border: 1px solid var(--border); }
  .value.gold { color: #7ee787; font-weight: 600; }
  .value.bad  { color: #ff7b72; }
  .value.good { color: #7ee787; }
  .verdict { padding: 2px 8px; border-radius: 4px; font-size: 12px;
             background: rgba(110, 168, 255, 0.15); border: 1px solid var(--accent); }
  .controls { display: flex; gap: 8px; align-items: center; margin-top: 12px;
              flex-wrap: wrap; }
  .controls .state { margin-left: 8px; color: var(--muted); font-size: 12px; }
  .state.correct { color: var(--ok); }
  .state.incorrect { color: var(--bad); }
  .state.unclear { color: var(--warn); }
  textarea { width: 100%; min-height: 60px; background: var(--panel2);
             color: var(--fg); border: 1px solid var(--border); border-radius: 6px;
             padding: 6px 8px; font: inherit; margin-top: 8px; }
  .hint { color: var(--muted); font-size: 12px; }
  .jump { background: var(--panel2); border: 1px solid var(--border); border-radius: 6px;
          padding: 4px 8px; color: var(--fg); }
  details { margin-top: 4px; }
  details summary { cursor: pointer; color: var(--muted); font-size: 12px; }
</style>
</head>
<body>
<header>
  <h1>Judge review — __DATASET__</h1>
  <span class="progress" id="progress"></span>
  <span class="spacer"></span>
  <button id="btn-prev">← Prev (p)</button>
  <input class="jump" id="jump" type="number" min="1" value="1" style="width: 60px">
  <button id="btn-next">Next (n) →</button>
  <button id="btn-download">Download labeled CSV</button>
  <button id="btn-reset" title="Clear all labels from this browser">Reset</button>
</header>
<div class="container" id="app"></div>
<script>
const DATA = __DATA_JSON__;
const DATASET = __DATASET_JSON__;
const FIELDNAMES = __FIELDNAMES_JSON__;
const STORAGE_KEY = "ablation_judge_labels::" + DATASET;

let idx = 0;

function loadLabels() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); }
  catch (e) { return {}; }
}
function saveLabels(obj) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(obj));
}
function rowKey(i) {
  const r = DATA[i];
  return [r.problem_id, r.step_id, r.proposal_idx || ""].join("|");
}

function esc(s) {
  if (s === undefined || s === null) return "";
  return String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
}

function render() {
  const labels = loadLabels();
  const r = DATA[idx];
  const k = rowKey(idx);
  const current = labels[k] || {};
  document.getElementById("progress").textContent =
    `${idx + 1} / ${DATA.length}  ·  labeled ${Object.values(labels).filter(v => v.label).length}`;
  document.getElementById("jump").value = idx + 1;

  const verdictBadge = `<span class="verdict">judge: ${esc(r.judge_verdict)}</span>`;
  const repairedDisplay = r.repaired_final_answer
    ? `<div class="value good">${esc(r.repaired_final_answer)}</div>`
    : `<div class="value" style="color: var(--muted); font-style: italic;">
         (not stored — judge is predict-outcome; compare repaired step vs. gold)
       </div>`;

  const app = document.getElementById("app");
  app.innerHTML = `
    <div class="card">
      <div class="row"><div class="label">Dataset</div>
        <div class="value">${esc(r.dataset)} · ${verdictBadge} · problem ${esc(r.problem_id)} · step ${esc(r.step_id)}${r.proposal_idx !== "" ? " · proposal " + esc(r.proposal_idx) : ""}</div></div>

      <div class="row"><div class="label">Problem</div>
        <div class="value">${esc(r.problem_statement)}</div></div>

      <div class="row"><div class="label">Gold answer</div>
        <div class="value gold">${esc(r.gold_answer)}</div></div>

      <div class="row"><div class="label">Original (failed)</div>
        <div class="value bad">${esc(r.original_final_answer)}</div></div>

      <div class="row"><div class="label">Repaired answer</div>
        ${repairedDisplay}</div>

      <div class="row"><div class="label">Original step</div>
        <div class="value mono">${esc(r.original_step_text)}</div></div>

      <div class="row"><div class="label">Repaired step</div>
        <div class="value mono">${esc(r.repaired_step_text)}</div></div>

      ${r.downstream_context ? `
      <div class="row"><div class="label">Downstream</div>
        <div class="value"><details><summary>show</summary><div class="mono" style="margin-top: 6px;">${esc(r.downstream_context)}</div></details></div></div>
      ` : ""}

      <div class="controls">
        <button class="ok"   data-lbl="correct">Correct (1)</button>
        <button class="bad"  data-lbl="incorrect">Incorrect (2)</button>
        <button class="warn" data-lbl="unclear">Unclear (3)</button>
        <span class="state ${current.label || ''}" id="state">${current.label ? "labeled: " + current.label : "unlabeled"}</span>
      </div>
      <textarea id="notes" placeholder="notes (optional)">${esc(current.notes || "")}</textarea>
      <div class="hint">Shortcuts: 1 correct · 2 incorrect · 3 unclear · n/→ next · p/← prev</div>
    </div>
  `;

  app.querySelectorAll("button[data-lbl]").forEach(b => {
    b.addEventListener("click", () => setLabel(b.getAttribute("data-lbl")));
  });
  document.getElementById("notes").addEventListener("input", (e) => {
    const labels = loadLabels();
    labels[k] = Object.assign({}, labels[k], { notes: e.target.value });
    saveLabels(labels);
  });
}

function setLabel(lbl) {
  const labels = loadLabels();
  const k = rowKey(idx);
  labels[k] = Object.assign({}, labels[k], { label: lbl });
  saveLabels(labels);
  // auto-advance
  if (idx < DATA.length - 1) { idx += 1; }
  render();
}

function go(delta) {
  idx = Math.max(0, Math.min(DATA.length - 1, idx + delta));
  render();
}

document.getElementById("btn-prev").addEventListener("click", () => go(-1));
document.getElementById("btn-next").addEventListener("click", () => go(+1));
document.getElementById("jump").addEventListener("change", (e) => {
  const v = parseInt(e.target.value, 10);
  if (!isNaN(v)) { idx = Math.max(0, Math.min(DATA.length - 1, v - 1)); render(); }
});
document.getElementById("btn-reset").addEventListener("click", () => {
  if (confirm("Clear all labels for " + DATASET + "?")) {
    localStorage.removeItem(STORAGE_KEY);
    render();
  }
});
document.getElementById("btn-download").addEventListener("click", () => {
  const labels = loadLabels();
  const rows = DATA.map((r, i) => {
    const k = rowKey(i);
    const l = labels[k] || {};
    return Object.assign({}, r, {
      human_label: l.label || "",
      human_notes: l.notes || "",
    });
  });
  const quote = (v) => {
    const s = v === undefined || v === null ? "" : String(v);
    if (/[",\n\r]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
    return s;
  };
  const header = FIELDNAMES.join(",");
  const body = rows.map(r => FIELDNAMES.map(f => quote(r[f])).join(",")).join("\n");
  const blob = new Blob([header + "\n" + body + "\n"], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = DATASET + "_labeled.csv";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
});

document.addEventListener("keydown", (e) => {
  // don't intercept typing in textarea or number input
  const tag = (e.target.tagName || "").toLowerCase();
  if (tag === "textarea" || tag === "input") return;
  if (e.key === "1") setLabel("correct");
  else if (e.key === "2") setLabel("incorrect");
  else if (e.key === "3") setLabel("unclear");
  else if (e.key === "n" || e.key === "ArrowRight") go(+1);
  else if (e.key === "p" || e.key === "ArrowLeft") go(-1);
});

render();
</script>
</body>
</html>
"""


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["sealqa", "medbrowse"])
    p.add_argument(
        "--db-name",
        default=os.getenv("MONGODB_NAME", "causal_flow_dups"),
    )
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    manifest_path = args.manifest or (REVIEW_DIR / f"{args.dataset}_sample.json")
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text())

    db = get_db(args.db_name)
    run_doc = db.runs.find_one({"run_id": manifest["run_id"]})
    if not run_doc:
        print(f"ERROR: run not found: {manifest['run_id']}", file=sys.stderr)
        return 2

    rows = build_rows(run_doc, manifest)

    html = (
        HTML_TEMPLATE
        .replace("__DATASET__", args.dataset)
        .replace("__DATASET_JSON__", json.dumps(args.dataset))
        .replace("__DATA_JSON__", json.dumps(rows))
        .replace("__FIELDNAMES_JSON__", json.dumps(FIELDNAMES))
    )

    out_path = args.out or (REVIEW_DIR / f"{args.dataset}_review.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    print(f"Wrote {out_path}  ({len(rows)} rows)")
    print("Open in a browser, label with 1/2/3 + n/p, then click 'Download labeled CSV'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
