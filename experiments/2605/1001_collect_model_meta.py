from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

ROOT = Path(__file__).resolve().parents[2]
DATADIR = ROOT / "data"
TOOL = "model-meta-crawler"
SOURCE = "modelsdev_api"
API_URL = "https://models.dev/api.json"
PARSER = "modelsdev_api.v0"
USER_AGENT = "tk-model-meta-crawler/0.1"

Capability = Literal[
  "tools", "parallel_tools", "structured_output", "json_mode",
  "reasoning", "reasoning_effort", "prompt_cache", "system_message",
  "web_search", "attachments", "vision_input", "audio_input", "audio_output",
]
Limit = Literal["context", "input", "output", "batch_context"]
Cost = Literal[
  "input", "output", "cache_read", "cache_write", "reasoning",
  "audio_input", "audio_output", "image_input", "image_output",
  "video_input", "video_output",
]
Modality = Literal["text", "image", "audio", "video", "pdf", "file"]
Mode = Literal["chat", "responses", "messages", "anthropic", "openai_compat"]

FIRST_PARTY = {
  "anthropic", "cohere", "deepseek", "google", "mistral", "moonshotai",
  "openai", "perplexity", "xai", "zai", "zhipuai",
}
COST_KEY: dict[str, Cost] = {"input_audio": "audio_input", "output_audio": "audio_output"}
CAP_MAP: list[tuple[str, Capability]] = [
  ("tool_call", "tools"), ("structured_output", "structured_output"),
  ("reasoning", "reasoning"), ("attachment", "attachments"),
]
MODEL_COLS = [
  "route", "raw_id", "name", "family", "context", "output", "input_cost",
  "output_cost", "tools", "structured_output", "reasoning", "modalities",
]


@dataclass
class NormalizedRecord:
  src: str
  route: str
  raw_id: str
  key: str
  name: str | None = None
  family: str | None = None
  model_key: str | None = None
  aliases: list[str] = field(default_factory=list)
  caps: dict[Capability, bool] = field(default_factory=dict)
  lim: dict[Limit, int] = field(default_factory=dict)
  cost: dict[Cost, float] = field(default_factory=dict)
  tiers: dict[str, dict[Cost, float]] = field(default_factory=dict)
  in_mod: list[Modality] = field(default_factory=list)
  out_mod: list[Modality] = field(default_factory=list)
  mode: Mode | None = None
  meta: dict[str, Any] = field(default_factory=dict)
  trace: dict[str, Any] = field(default_factory=dict)
  raw: dict[str, Any] = field(default_factory=dict)


def norm_id(s: str) -> str:
  return re.sub(r"\s+", "-", s.strip().lower())


def default_outdir() -> Path:
  return DATADIR / "out" / TOOL / datetime.now().strftime("%y%m")


def get_vnext(outdir: Path) -> str:
  if not outdir.exists(): return "v0"
  vs = [int(s) for d in outdir.iterdir() if d.is_dir()
        and ".v" in d.name and (s := d.name.rsplit(".v", 1)[1]).isdigit()]
  return f"v{max(vs, default=-1) + 1}"


def setup_outdir(outdir: Path | None, run: str | None) -> Path:
  d = (outdir or default_outdir()) / (run or f"modelsdev.{get_vnext(outdir or default_outdir())}")
  for p in [d / "raw" / SOURCE, d / "normalized", d / "reports"]:
    p.mkdir(parents=True, exist_ok=True)
  return d


def get_api(url: str = API_URL) -> tuple[dict[str, Any], str]:
  ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
  req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
  with urllib.request.urlopen(req, timeout=60) as r:
    return json.load(r), ts


def get_model_key(route: str, raw_id: str) -> str | None:
  x = norm_id(raw_id)
  if "/" in x: return x
  return f"{route}/{x}" if route in FIRST_PARTY else None


def get_caps(m: dict[str, Any], cost: dict[str, Any]) -> dict[Capability, bool]:
  mods = m.get("modalities", {})
  ins, outs = set(mods.get("input", [])), set(mods.get("output", []))
  caps: dict[Capability, bool] = {}
  for src, dst in CAP_MAP:
    if src in m: caps[dst] = bool(m[src])
  if ins:
    caps["vision_input"] = "image" in ins
    caps["audio_input"] = "audio" in ins
  if outs: caps["audio_output"] = "audio" in outs
  if "cache_read" in cost or "cache_write" in cost: caps["prompt_cache"] = True
  return caps


def get_costs(cost: dict[str, Any]) -> dict[Cost, float]:
  return {cast(Cost, COST_KEY.get(k, k)): float(v) for k, v in cost.items()
          if isinstance(v, int | float)}


def get_record(pid: str, p: dict[str, Any], mid: str, m: dict[str, Any],
               ts: str) -> NormalizedRecord:
  raw_id, x = str(m.get("id", mid)), norm_id(str(m.get("id", mid)))
  cost, limit, mods = m.get("cost", {}), m.get("limit", {}), m.get("modalities", {})
  meta = {
    "provider_name": p.get("name"), "endpoint": p.get("npm"), "api": p.get("api"),
    "doc": p.get("doc"), "release_date": m.get("release_date"),
    "knowledge_cutoff": m.get("knowledge"), "updated_at": m.get("last_updated"),
    "status": m.get("status"), "open_weights": m.get("open_weights"),
    "temperature": m.get("temperature"), "row_id": mid if mid != raw_id else None,
  }
  aliases = [mid] if mid != raw_id else []
  return NormalizedRecord(
    src=SOURCE, route=pid, raw_id=raw_id, key=f"{pid}/{x}",
    name=str(m["name"]) if "name" in m else None,
    family=str(m["family"]) if "family" in m else None,
    model_key=get_model_key(pid, raw_id), aliases=aliases,
    caps=get_caps(m, cost),
    lim={cast(Limit, k): int(v) for k, v in limit.items() if isinstance(v, int)},
    cost=get_costs(cost),
    tiers={k: get_costs(v) for k, v in cost.items() if isinstance(v, dict)},
    in_mod=cast(list[Modality], mods.get("input", []) or []),
    out_mod=cast(list[Modality], mods.get("output", []) or []),
    meta={k: v for k, v in meta.items() if v not in (None, [], {})},
    trace={"url": API_URL, "path": f"{pid}.models.{mid}",
           "extracted_at": ts, "parser": PARSER},
    raw={"provider": {k: v for k, v in p.items() if k != "models"}, "model": m},
  )


def get_records(data: dict[str, Any], ts: str) -> list[NormalizedRecord]:
  return [get_record(pid, p, mid, m, ts) for pid, p in data.items()
          for mid, m in p.get("models", {}).items()]


def flat(r: NormalizedRecord) -> dict[str, Any]:
  caps, lim, cost, meta = r.caps, r.lim, r.cost, r.meta
  return {
    "key": r.key, "model_key": r.model_key or "",
    "route": r.route, "raw_id": r.raw_id, "name": r.name or "",
    "family": r.family or "", "context": lim.get("context", ""),
    "input": lim.get("input", ""), "output": lim.get("output", ""),
    "input_cost": cost.get("input", ""), "output_cost": cost.get("output", ""),
    "cache_read": cost.get("cache_read", ""), "cache_write": cost.get("cache_write", ""),
    "tools": caps.get("tools", ""), "structured_output": caps.get("structured_output", ""),
    "reasoning": caps.get("reasoning", ""), "prompt_cache": caps.get("prompt_cache", ""),
    "release_date": meta.get("release_date", ""), "updated_at": meta.get("updated_at", ""),
    "modalities": "/".join(r.in_mod) + " -> " + "/".join(r.out_mod),
    "status": meta.get("status", ""), "api": meta.get("api", ""),
  }


def md_table(rows: list[dict[str, Any]], cols: list[str], n: int = 20) -> str:
  rs = rows[:n]
  head = "| " + " | ".join(cols) + " |"
  sep = "| " + " | ".join(["---"] * len(cols)) + " |"
  body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rs]
  return "\n".join([head, sep, *body])




def get_report_rows(records: list[NormalizedRecord]) -> dict[str, list[dict[str, Any]]]:
  rows = [flat(r) for r in records]
  return {
    "routes": [{"route": k, "models": v} for k, v in
               Counter(r.route for r in records).most_common(30)],
    "largest_context": sorted(rows, key=lambda r: int(r["context"] or 0), reverse=True),
    "highest_output_cost": sorted(rows, key=lambda r: float(r["output_cost"] or 0), reverse=True),
    "recent": sorted([r for r in rows if r["release_date"]],
                     key=lambda r: r["release_date"], reverse=True),
    "focus": [r for r in rows if re.search(
      r"gpt-5|claude|gemini|deepseek|kimi|minimax|glm", r["key"], re.I)],
  }


def get_summary(records: list[NormalizedRecord]) -> dict[str, Any]:
  caps = Counter(k for r in records for k, v in r.caps.items() if v)
  return {
    "records": len(records), "routes": len({r.route for r in records}),
    "with_costs": sum(bool(r.cost) for r in records),
    "with_context": sum("context" in r.lim for r in records),
    "with_model_key": sum(bool(r.model_key) for r in records),
    "tiered_costs": sum(bool(r.tiers) for r in records),
    "capabilities_true": dict(caps.most_common()),
  }


def write_app(d: Path) -> Path:
  app = d / "reports" / "app.py"
  app.write_text(r'''
from pathlib import Path
import json, sys
import pandas as pd
import streamlit as st

run = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parents[1]
csv_path = run / "normalized" / "modelsdev_records.csv"
jsonl_path = run / "normalized" / "modelsdev_records.jsonl"

@st.cache_data
def load():
  df = pd.read_csv(csv_path).fillna("")
  recs = {r["key"]: r for r in map(json.loads, jsonl_path.read_text().splitlines())}
  for c in ["context", "output", "input_cost", "output_cost"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
  return df, recs

df, recs = load()
st.set_page_config(page_title="Models.dev metadata", layout="wide")
st.title("Models.dev metadata")
st.caption(str(run))
a, b, c, d = st.columns(4)
a.metric("records", len(df)); b.metric("routes", df.route.nunique())
c.metric("with cost", int(df.input_cost.notna().sum()))
d.metric("with context", int(df.context.notna().sum()))

with st.sidebar:
  q = st.text_input("search")
  routes = st.multiselect("route", sorted(df.route.unique()))
  caps = st.multiselect("cap", ["tools", "structured_output", "reasoning", "prompt_cache"])
  min_ctx = st.number_input("min context", 0, step=1000)
  cols = st.multiselect("columns", list(df.columns), default=[
    "route", "raw_id", "name", "family", "context", "output", "input_cost",
    "output_cost", "tools", "structured_output", "reasoning", "modalities",
  ])

x = df
if q:
  mask = x.astype(str).apply(lambda s: s.str.contains(q, case=False, regex=False)).any(axis=1)
  x = x[mask]
if routes: x = x[x.route.isin(routes)]
for cap in caps: x = x[x[cap].astype(str).str.lower().eq("true")]
if min_ctx: x = x[x.context.ge(min_ctx)]

st.subheader(f"records ({len(x)})")
st.dataframe(x[cols], use_container_width=True, height=560)

left, right = st.columns([1, 2])
with left:
  st.subheader("routes")
  st.dataframe(x.route.value_counts().rename_axis("route").reset_index(name="n"),
               use_container_width=True, height=320)
with right:
  st.subheader("record JSON")
  key = st.selectbox("key", x.key.tolist()[:1000]) if len(x) else None
  if key: st.json(recs.get(key, {}), expanded=False)
'''.strip() + "\n", encoding="utf-8")
  return app


def write_reports(d: Path, records: list[NormalizedRecord], ts: str) -> dict[str, Path]:
  rows, summary = get_report_rows(records), get_summary(records)
  report = d / "reports" / "report.md"
  report.write_text("\n\n".join([
    "# Models.dev metadata MVP",
    f"Fetched: `{ts}`  ", f"Records: `{summary['records']}`  ",
    f"Routes: `{summary['routes']}`  ",
    f"With costs/context: `{summary['with_costs']}` / `{summary['with_context']}`  ",
    f"With `model_key`: `{summary['with_model_key']}`  ",
    f"Tiered costs: `{summary['tiered_costs']}`",
    "## Capabilities present",
    md_table([{"capability": k, "true": v}
              for k, v in summary["capabilities_true"].items()],
             ["capability", "true"], 20),
    "## Routes by model count", md_table(rows["routes"], ["route", "models"], 30),
    "## Largest context windows", md_table(rows["largest_context"], MODEL_COLS, 20),
    "## Highest output prices", md_table(rows["highest_output_cost"], MODEL_COLS, 20),
    "## Recent releases", md_table(rows["recent"], MODEL_COLS + ["release_date"], 25),
    "## Focus families", md_table(rows["focus"], MODEL_COLS + ["release_date"], 50),
  ]) + "\n", encoding="utf-8")
  return {"report": report, "app": write_app(d)}


def write_outputs(d: Path, data: dict[str, Any], records: list[NormalizedRecord],
                  ts: str) -> dict[str, Path]:
  raw = d / "raw" / SOURCE / "api.json"
  norm = d / "normalized" / "modelsdev_records.jsonl"
  csv_path = d / "normalized" / "modelsdev_records.csv"
  raw.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
  norm.write_text("\n".join(json.dumps(asdict(r), ensure_ascii=False)
                             for r in records) + "\n", encoding="utf-8")
  with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(flat(records[0]).keys()))
    w.writeheader(); w.writerows(flat(r) for r in records)
  return {"raw": raw, "jsonl": norm, "csv": csv_path, **write_reports(d, records, ts)}


def cmd_run(args: argparse.Namespace) -> None:
  d = setup_outdir(args.outdir, args.run)
  data, ts = get_api(args.url)
  records = get_records(data, ts)
  paths = write_outputs(d, data, records, ts)
  summary = get_summary(records)
  print(json.dumps({"outdir": str(d), **summary,
                    "paths": {k: str(v) for k, v in paths.items()}}, indent=2))


def cmd_show(args: argparse.Namespace) -> None:
  report = Path(args.run_dir) / "reports" / "report.md"
  print(report.read_text(encoding="utf-8"))


def cmd_app(args: argparse.Namespace) -> None:
  run = Path(args.run_dir)
  viewer = Path(__file__).resolve().parent / "viewer.html"
  print(f"Open {viewer} in a browser, then load:")
  print(f"  JSONL: {run / 'normalized' / 'modelsdev_records.jsonl'}")
  print(f"  CSV:   {run / 'normalized' / 'modelsdev_records.csv'}")
  import webbrowser
  webbrowser.open(viewer.as_uri())


def main(argv: list[str] | None = None) -> None:
  p = argparse.ArgumentParser(description="Fetch and normalize Models.dev metadata")
  sub = p.add_subparsers(dest="cmd")
  run = sub.add_parser("run")
  run.add_argument("--url", default=API_URL)
  run.add_argument("--outdir", type=Path)
  run.add_argument("--run")
  run.set_defaults(fn=cmd_run)
  show = sub.add_parser("show")
  show.add_argument("run_dir")
  show.set_defaults(fn=cmd_show)
  app = sub.add_parser("app")
  app.add_argument("run_dir")
  app.set_defaults(fn=cmd_app)
  if argv is None and len(sys.argv) == 1: argv = ["run"]
  args = p.parse_args(argv)
  args.fn(args)


if __name__ == "__main__":
  main()
