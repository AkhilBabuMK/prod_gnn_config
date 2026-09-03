# -*- coding: utf-8 -*-
"""Parse rollout journey tables into a single reviewable HTML page."""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

SRC = Path(sys.argv[1])
DST = Path(sys.argv[2])

HEAD = re.compile(r"^\s*\[(\w+)\]\s+Train (\S+)\s+\((\w+), (\w+)\)\s+(\S+)")
ISSUED = re.compile(r"Forecast issued (\S+)\s+delay at that moment ([+-]?\d+) min")
ENTRY = re.compile(r"Entered corridor ([+-]?\d+) min late from upstream \((\S+) km")
SUMM = re.compile(r"(\d+) forecast stops\s+MAE ([\d.]+) min\s+persistence ([\d.]+) min\s+gain ([+-][\d.]+)")
# station sched actual delay pred err  note
ROW = re.compile(
    r"^\s{2}(\S+)\s+(\d\d:\d\d)\s+(\d\d:\d\d|\s*-)\s+([+-]?\d+|\?)"
    r"(?:\s+([+-]?\d+)\s+([+-]?\d+))?\s*(.*)$")

journeys = []
cur = None
for raw in SRC.read_text(encoding="utf-8", errors="replace").splitlines():
    line = raw.rstrip()
    m = HEAD.match(line)
    if m:
        if cur:
            journeys.append(cur)
        cur = {"scenario": m.group(1), "train": m.group(2), "klass": m.group(3),
               "dir": m.group(4), "date": m.group(5), "rows": [],
               "issued": "", "now_delay": None, "entry": None, "km": None}
        continue
    if cur is None:
        continue
    m = ISSUED.search(line)
    if m:
        cur["issued"] = m.group(1)
        cur["now_delay"] = int(m.group(2))
        continue
    m = ENTRY.search(line)
    if m:
        cur["entry"] = int(m.group(1))
        cur["km"] = m.group(2)
        continue
    m = SUMM.search(line)
    if m:
        cur["n"] = int(m.group(1))
        cur["mae"] = float(m.group(2))
        cur["persist"] = float(m.group(3))
        cur["gain"] = float(m.group(4))
        continue
    if "<== NOW" in line:
        pass
    m = ROW.match(line)
    if m and re.match(r"^[A-Z]{2,6}$", m.group(1)):
        stn, sched, actual, delay, pred, err, note = m.groups()
        cur["rows"].append({
            "stn": stn, "sched": sched, "actual": actual.strip(),
            "delay": delay, "pred": pred, "err": err,
            "note": (note or "").strip(),
            "now": "<== NOW" in line,
        })
if cur:
    journeys.append(cur)

journeys = [j for j in journeys if j.get("rows") and "mae" in j]
for j in journeys:
    hop = re.compile(r"h(\d+)\s")
    for r in j["rows"]:
        mm = hop.search(r["note"])
        r["hop"] = mm.group(1) if mm else ""
        n = r["note"]
        r["flag"] = ("miss" if "SPIKE MISSED" in n else
                     "spike" if "SPIKE" in n else
                     "recov" if "recovery" in n else "")
        r["note"] = re.sub(r"h\d+\s+\+\d+m\s*", "", n).strip()

data = json.dumps(journeys, separators=(",", ":"))
n_sp = sum(1 for j in journeys if j["scenario"] == "SPIKE")
tot_mae = sum(j["mae"] for j in journeys) / max(len(journeys), 1)
tot_per = sum(j["persist"] for j in journeys) / max(len(journeys), 1)
beat = sum(1 for j in journeys if j["mae"] < j["persist"])

DST.write_text(f"""<title>JBP–STA rollout review — v2</title>
<style>
:root {{
  --ground:#f4f5f7; --panel:#ffffff; --line:#d8dce2; --line-soft:#e8ebef;
  --ink:#161a1f; --ink-2:#5a6472; --ink-3:#8a93a1;
  --route:#2f5d8a; --route-soft:#e4edf6;
  --ok:#2f7d4f; --warn:#b5761a; --bad:#bd3f38;
  --ok-bg:#e6f2ea; --warn-bg:#faf0dd; --bad-bg:#fbe8e6;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --ground:#0e1116; --panel:#161b22; --line:#2a323d; --line-soft:#1e242c;
    --ink:#e6e9ee; --ink-2:#9aa4b2; --ink-3:#6b7482;
    --route:#6fa8dc; --route-soft:#18293a;
    --ok:#5fbe86; --warn:#e0a94b; --bad:#e97066;
    --ok-bg:#14291e; --warn-bg:#2e2413; --bad-bg:#2f1a18;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#0e1116; --panel:#161b22; --line:#2a323d; --line-soft:#1e242c;
  --ink:#e6e9ee; --ink-2:#9aa4b2; --ink-3:#6b7482;
  --route:#6fa8dc; --route-soft:#18293a;
  --ok:#5fbe86; --warn:#e0a94b; --bad:#e97066;
  --ok-bg:#14291e; --warn-bg:#2e2413; --bad-bg:#2f1a18;
}}
:root[data-theme="light"] {{
  --ground:#f4f5f7; --panel:#ffffff; --line:#d8dce2; --line-soft:#e8ebef;
  --ink:#161a1f; --ink-2:#5a6472; --ink-3:#8a93a1;
  --route:#2f5d8a; --route-soft:#e4edf6;
  --ok:#2f7d4f; --warn:#b5761a; --bad:#bd3f38;
  --ok-bg:#e6f2ea; --warn-bg:#faf0dd; --bad-bg:#fbe8e6;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--ground); color:var(--ink);
  font:14px/1.5 ui-sans-serif,system-ui,"Segoe UI",sans-serif;
  font-variant-numeric:tabular-nums;
}}
.wrap {{ max-width:1180px; margin:0 auto; padding:0 20px 80px; }}
header {{
  border-bottom:2px solid var(--ink); padding:28px 0 16px; margin-bottom:0;
}}
h1 {{
  font:600 15px/1.2 ui-sans-serif,system-ui,sans-serif;
  letter-spacing:.16em; text-transform:uppercase; margin:0 0 6px;
}}
.sub {{ color:var(--ink-2); font-size:13px; margin:0; }}
.strip {{
  display:flex; flex-wrap:wrap; gap:0; border-bottom:1px solid var(--line);
  background:var(--panel); position:sticky; top:0; z-index:5;
}}
.stat {{ padding:12px 20px 12px 0; margin-right:20px; border-right:1px solid var(--line-soft); }}
.stat:last-child {{ border-right:0; }}
.stat b {{ display:block; font:600 20px/1.1 ui-monospace,Menlo,Consolas,monospace; }}
.stat span {{ font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--ink-3); }}
.filters {{ display:flex; flex-wrap:wrap; gap:8px; padding:14px 0; }}
button.chip {{
  font:500 12px/1 ui-sans-serif,system-ui,sans-serif; letter-spacing:.04em;
  padding:7px 12px; border:1px solid var(--line); background:var(--panel);
  color:var(--ink-2); border-radius:2px; cursor:pointer;
}}
button.chip[aria-pressed="true"] {{ background:var(--ink); color:var(--ground); border-color:var(--ink); }}
button.chip:focus-visible {{ outline:2px solid var(--route); outline-offset:2px; }}
.card {{
  background:var(--panel); border:1px solid var(--line); border-radius:3px;
  margin:14px 0; overflow:hidden;
}}
.chead {{ display:flex; flex-wrap:wrap; align-items:baseline; gap:10px 14px;
  padding:12px 16px; border-bottom:1px solid var(--line-soft); }}
.tag {{ font:600 10px/1 ui-sans-serif,sans-serif; letter-spacing:.1em;
  padding:4px 7px; border-radius:2px; text-transform:uppercase; }}
.t-SPIKE {{ background:var(--bad-bg); color:var(--bad); }}
.t-RECOVERY {{ background:var(--ok-bg); color:var(--ok); }}
.t-MODERATE {{ background:var(--warn-bg); color:var(--warn); }}
.t-STEADY {{ background:var(--route-soft); color:var(--route); }}
.tno {{ font:600 16px ui-monospace,Menlo,Consolas,monospace; }}
.meta {{ color:var(--ink-2); font-size:12.5px; }}
.score {{ margin-left:auto; font:13px ui-monospace,Menlo,Consolas,monospace; }}
.score em {{ font-style:normal; color:var(--ink-3); }}
.scroll {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font:13px/1.45 ui-monospace,Menlo,Consolas,monospace; }}
th {{
  font:600 10px/1 ui-sans-serif,sans-serif; letter-spacing:.1em; text-transform:uppercase;
  color:var(--ink-3); text-align:right; padding:9px 10px; border-bottom:1px solid var(--line);
  white-space:nowrap;
}}
th:first-child {{ text-align:left; }}
td {{ padding:5px 10px; text-align:right; border-bottom:1px solid var(--line-soft); white-space:nowrap; }}
td:first-child {{ text-align:left; font-weight:600; }}
tr.obs td {{ color:var(--ink-2); background:linear-gradient(90deg,var(--route-soft),transparent 60%); }}
tr.now td {{ border-bottom:2px solid var(--route); }}
tr.now td:first-child::after {{ content:" ◀ now"; color:var(--route); font-weight:400; font-size:11px; }}
tr.miss {{ background:var(--bad-bg); }}
tr.spike td:first-child {{ box-shadow:inset 3px 0 0 var(--ok); }}
tr.miss td:first-child {{ box-shadow:inset 3px 0 0 var(--bad); }}
tr.recov td:first-child {{ box-shadow:inset 3px 0 0 var(--route); }}
.err {{ font-weight:600; }}
.e0 {{ color:var(--ink-3); }} .e1 {{ color:var(--ok); }}
.e2 {{ color:var(--warn); }} .e3 {{ color:var(--bad); }}
.bar {{ display:inline-block; height:9px; border-radius:1px; vertical-align:middle; }}
.bwrap {{ width:120px; text-align:left; }}
.bmid {{ display:inline-block; width:56px; text-align:right; }}
.note {{ font:500 10px/1 ui-sans-serif,sans-serif; letter-spacing:.08em;
  text-transform:uppercase; color:var(--ink-3); }}
.empty {{ padding:40px; text-align:center; color:var(--ink-3); }}
footer {{ margin-top:30px; padding-top:16px; border-top:1px solid var(--line);
  color:var(--ink-3); font-size:12px; }}
</style>

<div class="wrap">
<header>
  <h1>JBP–STA corridor · rollout review</h1>
  <p class="sub">Model <strong>v2 epoch 23</strong> · true autoregressive rollout, no observations after the decision point ·
  test days 2025-09-26 → 09-30</p>
</header>

<div class="strip" id="strip"></div>
<div class="filters" id="filters"></div>
<div id="list"></div>

<footer>
Each row past the marked decision point is forecast from the model's own previous prediction — the journey is
rewritten and the full network snapshot rebuilt at every step, so nothing downstream sees a real observation.
<strong>SPIKE</strong> marks a rise over 30&nbsp;min against the delay at decision time;
<strong>SPIKE MISSED</strong> means the model predicted less than half that rise.
</footer>
</div>

<script>
const J = {data};
const F = {{scenario:"all", klass:"all"}};

const strip = document.getElementById("strip");
strip.innerHTML = [
  ["{len(journeys)}", "journeys"],
  ["{n_sp}", "spike cases"],
  ["{tot_mae:.1f}", "mean MAE min"],
  ["{tot_per:.1f}", "mean persistence"],
  ["{beat}/{len(journeys)}", "beat persistence"],
].map(([v,l]) => `<div class="stat"><b>${{v}}</b><span>${{l}}</span></div>`).join("");

const scen = ["all","SPIKE","RECOVERY","MODERATE","STEADY"];
const klasses = ["all", ...new Set(J.map(j=>j.klass))];
const fbox = document.getElementById("filters");
fbox.innerHTML =
  scen.map(s=>`<button class="chip" data-k="scenario" data-v="${{s}}" aria-pressed="${{s==='all'}}">${{s}}</button>`).join("")
  + `<span style="width:14px"></span>`
  + klasses.map(c=>`<button class="chip" data-k="klass" data-v="${{c}}" aria-pressed="${{c==='all'}}">${{c.replace(/_/g," ")}}</button>`).join("");

fbox.addEventListener("click", e => {{
  const b = e.target.closest("button.chip"); if (!b) return;
  F[b.dataset.k] = b.dataset.v;
  fbox.querySelectorAll(`button[data-k="${{b.dataset.k}}"]`).forEach(x =>
    x.setAttribute("aria-pressed", x.dataset.v === b.dataset.v));
  render();
}});

function errCell(e) {{
  if (e === null || e === undefined || e === "") return `<td></td><td class="bwrap"></td>`;
  const v = parseInt(e, 10), a = Math.abs(v);
  const cls = a <= 3 ? "e1" : a <= 10 ? "e2" : a <= 20 ? "e2" : "e3";
  const w = Math.min(a * 2.0, 56);
  const col = a <= 3 ? "var(--ok)" : a <= 10 ? "var(--warn)" : "var(--bad)";
  const bar = v < 0
    ? `<span class="bmid"><span class="bar" style="width:${{w}}px;background:${{col}}"></span></span>`
    : `<span class="bmid"></span><span class="bar" style="width:${{w}}px;background:${{col}}"></span>`;
  return `<td class="err ${{cls}}">${{v>0?"+":""}}${{v}}</td><td class="bwrap">${{bar}}</td>`;
}}

function card(j) {{
  const rows = j.rows.map(r => {{
    const forecast = r.pred !== null && r.pred !== undefined && r.pred !== "";
    const cls = [forecast ? "" : "obs", r.now ? "now" : "", r.flag].filter(Boolean).join(" ");
    return `<tr class="${{cls}}">
      <td>${{r.stn}}</td><td>${{r.sched}}</td><td>${{r.actual||"—"}}</td>
      <td>${{r.delay==="?"?"—":(parseInt(r.delay,10)>0?"+":"")+r.delay}}</td>
      <td>${{forecast?((parseInt(r.pred,10)>0?"+":"")+r.pred):""}}</td>
      ${{errCell(forecast ? r.err : "")}}
      <td class="note">${{r.hop?("h"+r.hop):""}} ${{r.flag==="miss"?"missed":r.flag==="spike"?"caught":r.flag==="recov"?"recovery":""}}</td>
    </tr>`;
  }}).join("");
  const g = j.gain >= 0 ? "var(--ok)" : "var(--bad)";
  return `<article class="card">
    <div class="chead">
      <span class="tag t-${{j.scenario}}">${{j.scenario}}</span>
      <span class="tno">${{j.train}}</span>
      <span class="meta">${{j.klass.replace(/_/g," ")}} · ${{j.dir}} · ${{j.date}} · issued ${{j.issued}} at ${{j.now_delay>0?"+":""}}${{j.now_delay}} min</span>
      <span class="score">MAE <strong>${{j.mae.toFixed(1)}}</strong> <em>vs persistence ${{j.persist.toFixed(1)}}</em>
        <strong style="color:${{g}}">${{j.gain>0?"+":""}}${{j.gain.toFixed(1)}}</strong></span>
    </div>
    <div class="scroll"><table>
      <thead><tr><th>station</th><th>sched</th><th>actual</th><th>delay</th><th>forecast</th><th>error</th><th></th><th></th></tr></thead>
      <tbody>${{rows}}</tbody>
    </table></div>
  </article>`;
}}

function render() {{
  const sel = J.filter(j =>
    (F.scenario === "all" || j.scenario === F.scenario) &&
    (F.klass === "all" || j.klass === F.klass));
  document.getElementById("list").innerHTML =
    sel.length ? sel.map(card).join("") : `<div class="empty">No journeys match that combination.</div>`;
}}
render();
</script>
""", encoding="utf-8")

print(f"parsed {len(journeys)} journeys -> {DST}")
print("  scenarios:", {s: sum(1 for j in journeys if j['scenario'] == s)
                       for s in ("SPIKE", "RECOVERY", "MODERATE", "STEADY")})
print("  classes  :", sorted({j["klass"] for j in journeys}))
print("  rows     :", sum(len(j["rows"]) for j in journeys))
