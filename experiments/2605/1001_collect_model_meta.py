from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

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

FIRST_PARTY = {
  "anthropic", "cohere", "deepseek", "google", "mistral", "moonshotai",
  "openai", "perplexity", "xai", "zai", "zhipuai",
}
COST_KEY = {"input_audio": "audio_input", "output_audio": "audio_output"}
CAP_MAP: list[tuple[str, Capability]] = [
  ("tool_call", "tools"), ("structured_output", "structured_output"),
  ("reasoning", "reasoning"), ("attachment", "attachments"),
]
MODEL_COLS = [
  "provider", "raw_id", "name", "family", "context", "output", "input_cost",
  "output_cost", "tools", "structured_output", "reasoning", "modalities",
]


@dataclass
class NormalizedRecord:
  source: str
  provider: str
  raw_id: str
  key: str
  underlying_key: str | None = None
  aliases: list[str] = field(default_factory=list)
  capabilities: dict[Capability, bool] = field(default_factory=dict)
  limits: dict[str, int] = field(default_factory=dict)
  costs_usd_1m: dict[str, float] = field(default_factory=dict)
  meta: dict[str, Any] = field(default_factory=dict)
  provenance: dict[str, Any] = field(default_factory=dict)
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


def get_underlying(provider: str, raw_id: str) -> str | None:
  x = norm_id(raw_id)
  if "/" in x: return x
  return f"{provider}/{x}" if provider in FIRST_PARTY else None


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


def get_costs(cost: dict[str, Any]) -> dict[str, float]:
  return {COST_KEY.get(k, k): float(v) for k, v in cost.items()
          if isinstance(v, int | float)}


def get_record(pid: str, p: dict[str, Any], mid: str, m: dict[str, Any],
               ts: str) -> NormalizedRecord:
  raw_id, x = str(m.get("id", mid)), norm_id(str(m.get("id", mid)))
  cost, limit, mods = m.get("cost", {}), m.get("limit", {}), m.get("modalities", {})
  tiers = {k: v for k, v in cost.items() if isinstance(v, dict)}
  meta = {
    "name": m.get("name"), "family": m.get("family"),
    "input_modalities": mods.get("input"), "output_modalities": mods.get("output"),
    "provider_name": p.get("name"), "endpoint": p.get("npm"), "api": p.get("api"),
    "doc": p.get("doc"), "release_date": m.get("release_date"),
    "knowledge_cutoff": m.get("knowledge"), "updated_at": m.get("last_updated"),
    "status": m.get("status"), "open_weights": m.get("open_weights"),
    "temperature": m.get("temperature"), "cost_tiers": tiers or None,
    "row_id": mid if mid != raw_id else None,
  }
  aliases = sorted({str(v) for v in [mid, m.get("name")] if v and str(v) != raw_id})
  return NormalizedRecord(
    source=SOURCE, provider=pid, raw_id=raw_id, key=f"{pid}/{x}",
    underlying_key=get_underlying(pid, raw_id), aliases=aliases,
    capabilities=get_caps(m, cost),
    limits={k: int(v) for k, v in limit.items() if isinstance(v, int)},
    costs_usd_1m=get_costs(cost),
    meta={k: v for k, v in meta.items() if v not in (None, [], {})},
    provenance={"url": API_URL, "path": f"{pid}.models.{mid}",
                "extracted_at": ts, "parser": PARSER},
    raw={"provider": {k: v for k, v in p.items() if k != "models"}, "model": m},
  )


def get_records(data: dict[str, Any], ts: str) -> list[NormalizedRecord]:
  return [get_record(pid, p, mid, m, ts) for pid, p in data.items()
          for mid, m in p.get("models", {}).items()]


def flat(r: NormalizedRecord) -> dict[str, Any]:
  caps, lim, cost, meta = r.capabilities, r.limits, r.costs_usd_1m, r.meta
  return {
    "key": r.key, "underlying_key": r.underlying_key or "",
    "provider": r.provider, "raw_id": r.raw_id, "name": meta.get("name", ""),
    "family": meta.get("family", ""), "context": lim.get("context", ""),
    "input": lim.get("input", ""), "output": lim.get("output", ""),
    "input_cost": cost.get("input", ""), "output_cost": cost.get("output", ""),
    "cache_read": cost.get("cache_read", ""), "cache_write": cost.get("cache_write", ""),
    "tools": caps.get("tools", ""), "structured_output": caps.get("structured_output", ""),
    "reasoning": caps.get("reasoning", ""), "prompt_cache": caps.get("prompt_cache", ""),
    "release_date": meta.get("release_date", ""), "updated_at": meta.get("updated_at", ""),
    "modalities": "/".join(meta.get("input_modalities", []) or []) + " -> "
                  + "/".join(meta.get("output_modalities", []) or []),
    "status": meta.get("status", ""), "api": meta.get("api", ""),
  }


def md_table(rows: list[dict[str, Any]], cols: list[str], n: int = 20) -> str:
  rs = rows[:n]
  head = "| " + " | ".join(cols) + " |"
  sep = "| " + " | ".join(["---"] * len(cols)) + " |"
  body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rs]
  return "\n".join([head, sep, *body])


def html_table(rows: list[dict[str, Any]], cols: list[str], n: int = 50) -> str:
  rs = rows[:n]
  th = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
  trs = []
  for r in rs:
    tds = "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in cols)
    trs.append(f"<tr>{tds}</tr>")
  return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def get_report_rows(records: list[NormalizedRecord]) -> dict[str, list[dict[str, Any]]]:
  rows = [flat(r) for r in records]
  return {
    "providers": [{"provider": k, "models": v} for k, v in
                  Counter(r.provider for r in records).most_common(30)],
    "largest_context": sorted(rows, key=lambda r: int(r["context"] or 0), reverse=True),
    "highest_output_cost": sorted(rows, key=lambda r: float(r["output_cost"] or 0), reverse=True),
    "recent": sorted([r for r in rows if r["release_date"]],
                     key=lambda r: r["release_date"], reverse=True),
    "focus": [r for r in rows if re.search(
      r"gpt-5|claude|gemini|deepseek|kimi|minimax|glm", r["key"], re.I)],
  }


def get_summary(records: list[NormalizedRecord]) -> dict[str, Any]:
  caps = Counter(k for r in records for k, v in r.capabilities.items() if v)
  return {
    "records": len(records), "providers": len({r.provider for r in records}),
    "with_costs": sum(bool(r.costs_usd_1m) for r in records),
    "with_context": sum("context" in r.limits for r in records),
    "with_underlying_key": sum(bool(r.underlying_key) for r in records),
    "tiered_costs": sum("cost_tiers" in r.meta for r in records),
    "capabilities_true": dict(caps.most_common()),
  }


def write_reports(d: Path, records: list[NormalizedRecord], ts: str) -> dict[str, Path]:
  rows, summary = get_report_rows(records), get_summary(records)
  report = d / "reports" / "report.md"
  report.write_text("\n\n".join([
    "# Models.dev metadata MVP",
    f"Fetched: `{ts}`  ", f"Records: `{summary['records']}`  ",
    f"Providers: `{summary['providers']}`  ",
    f"With costs/context: `{summary['with_costs']}` / `{summary['with_context']}`  ",
    f"With `underlying_key`: `{summary['with_underlying_key']}`  ",
    f"Tiered costs: `{summary['tiered_costs']}`",
    "## Capabilities present",
    md_table([{"capability": k, "true": v}
              for k, v in summary["capabilities_true"].items()],
             ["capability", "true"], 20),
    "## Providers by model count", md_table(rows["providers"], ["provider", "models"], 30),
    "## Largest context windows", md_table(rows["largest_context"], MODEL_COLS, 20),
    "## Highest output prices", md_table(rows["highest_output_cost"], MODEL_COLS, 20),
    "## Recent releases", md_table(rows["recent"], MODEL_COLS + ["release_date"], 25),
    "## Focus families", md_table(rows["focus"], MODEL_COLS + ["release_date"], 50),
  ]) + "\n", encoding="utf-8")
  style = "body{font-family:ui-sans-serif,system-ui;margin:2rem} table{border-collapse:collapse;font-size:13px} th,td{border:1px solid #ddd;padding:4px 7px} th{background:#f5f5f5;position:sticky;top:0} code{background:#f6f6f6;padding:1px 3px}"
  html_doc = ["<!doctype html><meta charset='utf-8'>", f"<style>{style}</style>",
              "<h1>Models.dev metadata MVP</h1>",
              f"<p>Fetched <code>{html.escape(ts)}</code>; "
              f"{summary['records']} records; {summary['providers']} providers.</p>"]
  for title, name, cols, n in [
    ("Capabilities present", "capabilities", ["capability", "true"], 20),
    ("Providers by model count", "providers", ["provider", "models"], 30),
    ("Largest context windows", "largest_context", MODEL_COLS, 50),
    ("Highest output prices", "highest_output_cost", MODEL_COLS, 50),
    ("Recent releases", "recent", MODEL_COLS + ["release_date"], 50),
    ("Focus families", "focus", MODEL_COLS + ["release_date"], 200),
  ]:
    src = [{"capability": k, "true": v}
           for k, v in summary["capabilities_true"].items()] if name == "capabilities" else rows[name]
    html_doc += [f"<h2>{title}</h2>", html_table(src, cols, n)]
  index = d / "reports" / "index.html"
  index.write_text("\n".join(html_doc), encoding="utf-8")
  return {"report": report, "html": index}


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
  if argv is None and len(sys.argv) == 1: argv = ["run"]
  args = p.parse_args(argv)
  args.fn(args)


if __name__ == "__main__":
  main()
