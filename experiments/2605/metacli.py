# %%
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import http.server
import json
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from collections import Counter
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from tqdm import tqdm
from typing import Any
from typing import Literal
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
DATADIR = ROOT / "data"
TOOL = "model-meta-crawler"
SOURCE = "modelsdev_api"
API_URL = "https://models.dev/api.json"
PARSER = "modelsdev_api.v0"
MODELS_UA = "tk-model-meta-crawler/0.1"
VIBES_UA = "tk-vibes-collector/0.1"
HN_API = "https://hn.algolia.com/api/v1/search_by_date"
RESOLVER_VERSION = "v3-conservative-version-phrase"

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
Sentiment = Literal["positive", "negative", "neutral", "mixed"]
VibeSource = Literal[
  "twitter", "reddit", "hackernews", "discord", "blog", "benchmark",
  "changelog", "manual",
]
VibeDimension = Literal[
  "overall", "coding", "reasoning", "instruction_following", "creativity",
  "speed", "reliability", "cost_value", "regression",
]

FIRST_PARTY = {
  "anthropic", "cohere", "deepseek", "google", "mistral", "moonshotai",
  "openai", "perplexity", "xai", "zai", "zhipuai",
}
COST_KEY: dict[str, Cost] = {
  "input_audio": "audio_input",
  "output_audio": "audio_output",
}
CAP_MAP: tuple[tuple[str, Capability], ...] = (
  ("tool_call", "tools"),
  ("structured_output", "structured_output"),
  ("reasoning", "reasoning"),
  ("attachment", "attachments"),
)
MODEL_COLS = [
  "route", "raw_id", "name", "family", "context", "output", "input_cost",
  "output_cost", "tools", "structured_output", "reasoning", "modalities",
]
HN_QUERIES = [
  "Claude", "GPT", "Gemini", "LLM", "Sonnet", "deepseek", "grok", "llama",
  "kimi",
]
REDDIT_SUBS = [
  "LocalLLaMA", "ChatGPT", "MachineLearning", "artificial", "ClaudeAI",
  "OpenAI", "singularity",
]
MODEL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
  (re.compile(p, re.I), k) for p, k in [
    (r"\bgpt[\-\s]?5o\b", "openai/gpt-5o"),
    (r"\bgpt[\-\s]?5\b", "openai/gpt-5"),
    (r"\bo4[\-\s]?mini\b", "openai/o4-mini"),
    (r"\bo3\b", "openai/o3"),
    (r"\bo1\b", "openai/o1"),
    (r"\bcodex\b", "openai/codex"),
    (r"\bclaude[\-\s]?opus[\-\s]?4\b", "anthropic/claude-opus-4"),
    (r"\bclaude[\-\s]?sonnet[\-\s]?4\b", "anthropic/claude-sonnet-4"),
    (r"\bclaude[\-\s]?haiku[\-\s]?4\b", "anthropic/claude-haiku-4"),
    (r"\bsonnet[\-\s]?4\b", "anthropic/claude-sonnet-4"),
    (r"\bopus[\-\s]?4\b", "anthropic/claude-opus-4"),
    (r"\bclaude[\-\s]?sonnet\b", "anthropic/claude-sonnet"),
    (r"\bclaude[\-\s]?opus\b", "anthropic/claude-opus"),
    (r"\bclaude[\-\s]?haiku\b", "anthropic/claude-haiku"),
    (r"\bgemini[\-\s]?2\.5[\-\s]?pro\b", "google/gemini-2.5-pro"),
    (r"\bgemini[\-\s]?2\.5[\-\s]?flash\b", "google/gemini-2.5-flash"),
    (r"\bgemini[\-\s]?2[\-\s]?pro\b", "google/gemini-2.0-pro"),
    (r"\bgemini[\-\s]?pro\b", "google/gemini-pro"),
    (r"\bdeepseek[\-\s]?r1\b", "deepseek/deepseek-r1"),
    (r"\bdeepseek[\-\s]?v3\b", "deepseek/deepseek-v3"),
    (r"\bdeepseek[\-\s]?coder\b", "deepseek/deepseek-coder"),
    (r"\bkimi[\-\s]?k2\b", "moonshotai/kimi-k2"),
    (r"\bkimi\b", "moonshotai/kimi"),
    (r"\bgrok[\-\s]?4\b", "xai/grok-4"),
    (r"\bgrok[\-\s]?3\b", "xai/grok-3"),
    (r"\bllama[\-\s]?4\b", "meta/llama-4"),
    (r"\bllama[\-\s]?3\b", "meta/llama-3"),
  ]
]
DIM_KEYWORDS: dict[VibeDimension, list[str]] = {
  "coding": ["coding", "code", "programming", "swe-bench", "aider", "copilot",
             "vscode", "cursor"],
  "reasoning": ["reasoning", "math", "logic", "think", "chain of thought", "cot"],
  "speed": ["fast", "slow", "latency", "speed", "ttft", "tokens per second", "tps"],
  "reliability": ["reliable", "consistent", "hallucin", "refuses", "lazy", "truncat"],
  "regression": [
    "worse", "nerfed", "downgrad", "regress", "lobotom", "dumber", "degraded",
  ],
  "cost_value": ["cheap", "expensive", "cost", "pricing", "value", "bang for buck"],
  "creativity": ["creative", "writing", "prose", "story", "poetry"],
  "instruction_following": ["instruction", "follows", "obedient", "system prompt"],
}
POS_KW = {
  "great", "amazing", "excellent", "love", "best", "impressive", "fantastic",
  "goat", "insane",
}
NEG_KW = {
  "terrible", "awful", "worse", "bad", "trash", "garbage", "disappointing",
  "broken", "unusable", "nerfed", "lobotomized", "dumber", "sucks",
}


# %% Models.dev data
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


@dataclass
class VibeSignal:
  model_key: str
  source: VibeSource
  sentiment: Sentiment
  dimensions: list[VibeDimension] = field(default_factory=list)
  intensity: float = 0.0
  text: str = ""
  author: str = ""
  author_credibility: float = 0.5
  url: str = ""
  timestamp: str = ""
  engagement: dict[str, int] = field(default_factory=dict)
  meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class VibeAggregate:
  model_key: str
  window_start: str
  window_end: str
  n_signals: int = 0
  mean_intensity: float = 0.0
  sentiment_dist: dict[str, int] = field(default_factory=dict)
  top_dimensions: list[str] = field(default_factory=list)
  trending: str = "unknown"
  credibility_weighted_intensity: float = 0.0
  notable_signals: list[str] = field(default_factory=list)


@dataclass
class VibeCfg:
  outdir: Path | None = None
  run: str | None = None
  days: int = 14
  hn_queries: list[str] = field(default_factory=lambda: list(HN_QUERIES))
  hn_max_pages: int = 20
  hn_hits_per_page: int = 50
  reddit_subs: list[str] = field(default_factory=lambda: list(REDDIT_SUBS))
  reddit_limit: int = 500
  user_agent: str = VIBES_UA


def norm_id(s: str) -> str:
  return re.sub(r"\s+", "-", s.strip().lower())


def get_default_outdir(datadir: Path = DATADIR, tool: str = TOOL) -> Path:
  return datadir / "out" / tool / datetime.now().strftime("%y%m")


def get_vnext(outdir: Path) -> str:
  if not outdir.exists(): return "v0"
  vs = [int(s) for d in outdir.iterdir() if d.is_dir()
        and ".v" in d.name and (s := d.name.rsplit(".v", 1)[1]).isdigit()]
  return f"v{max(vs, default=-1) + 1}"


def setup_models_dir(outdir: Path | None, run: str | None) -> Path:
  base = outdir or get_default_outdir()
  d = base / (run or f"modelsdev.{get_vnext(base)}")
  for p in [d / "raw" / SOURCE, d / "normalized", d / "reports"]:
    p.mkdir(parents=True, exist_ok=True)
  return d


def setup_vibes_dir(cfg: VibeCfg) -> Path:
  base = cfg.outdir or get_default_outdir()
  d = base / (cfg.run or f"vibes.{get_vnext(base)}")
  d.mkdir(parents=True, exist_ok=True)
  return d


def get_latest(root: Path, glob: str) -> Path | None:
  xs = [p for p in root.rglob(glob) if p.is_file()]
  return max(xs, key=lambda p: p.stat().st_mtime, default=None)


def get_api(url: str = API_URL) -> tuple[dict[str, Any], str]:
  ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
  req = urllib.request.Request(url, headers={"User-Agent": MODELS_UA})
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
  cost = m.get("cost", {})
  limit = m.get("limit", {})
  mods = m.get("modalities", {})
  meta = {
    "provider_name": p.get("name"), "endpoint": p.get("npm"),
    "api": p.get("api"), "doc": p.get("doc"),
    "release_date": m.get("release_date"),
    "knowledge_cutoff": m.get("knowledge"),
    "updated_at": m.get("last_updated"), "status": m.get("status"),
    "open_weights": m.get("open_weights"),
    "temperature": m.get("temperature"),
    "row_id": mid if mid != raw_id else None,
  }
  return NormalizedRecord(
    src=SOURCE, route=pid, raw_id=raw_id, key=f"{pid}/{x}",
    name=str(m["name"]) if "name" in m else None,
    family=str(m["family"]) if "family" in m else None,
    model_key=get_model_key(pid, raw_id),
    aliases=[mid] if mid != raw_id else [],
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
    "cache_read": cost.get("cache_read", ""),
    "cache_write": cost.get("cache_write", ""),
    "tools": caps.get("tools", ""),
    "structured_output": caps.get("structured_output", ""),
    "reasoning": caps.get("reasoning", ""),
    "prompt_cache": caps.get("prompt_cache", ""),
    "release_date": meta.get("release_date", ""),
    "updated_at": meta.get("updated_at", ""),
    "modalities": "/".join(r.in_mod) + " -> " + "/".join(r.out_mod),
    "status": meta.get("status", ""), "api": meta.get("api", ""),
  }


def md_table(rows: list[dict[str, Any]], cols: list[str], n: int = 20) -> str:
  rs = rows[:n]
  head = "| " + " | ".join(cols) + " |"
  sep = "| " + " | ".join(["---"] * len(cols)) + " |"
  body = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |"
          for r in rs]
  return "\n".join([head, sep, *body])


def get_report_rows(records: list[NormalizedRecord]) -> dict[str, list[dict[str, Any]]]:
  rows = [flat(r) for r in records]
  return {
    "routes": [{"route": k, "models": v} for k, v in
               Counter(r.route for r in records).most_common(30)],
    "largest_context": sorted(rows, key=lambda r: int(r["context"] or 0), reverse=True),
    "highest_output_cost": sorted(
      rows, key=lambda r: float(r["output_cost"] or 0), reverse=True,
    ),
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
import json
import sys

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
  caps = st.multiselect(
    "cap", ["tools", "structured_output", "reasoning", "prompt_cache"],
  )
  min_ctx = st.number_input("min context", 0, step=1000)
  cols = st.multiselect("columns", list(df.columns), default=[
    "route", "raw_id", "name", "family", "context", "output", "input_cost",
    "output_cost", "tools", "structured_output", "reasoning", "modalities",
  ])

x = df
if q:
  mask = x.astype(str).apply(
    lambda s: s.str.contains(q, case=False, regex=False),
  ).any(axis=1)
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


# %% Vibes

def resolve_models(text: str) -> list[str]:
  seen: set[str] = set()
  keys = [k for pat, k in MODEL_PATTERNS
          if pat.search(text) and not (k in seen or seen.add(k))]
  return [k for k in keys if not any(o.startswith(k) and o != k for o in keys)]


def infer_sentiment(text: str) -> tuple[Sentiment, float]:
  low = text.lower()
  pos = sum(1 for w in POS_KW if w in low)
  neg = sum(1 for w in NEG_KW if w in low)
  if pos and not neg: return "positive", min(0.3 + pos * 0.2, 1.0)
  if neg and not pos: return "negative", max(-0.3 - neg * 0.2, -1.0)
  if pos and neg: return "mixed", (pos - neg) * 0.15
  return "neutral", 0.0


def infer_dimensions(text: str) -> list[VibeDimension]:
  low = text.lower()
  return [dim for dim, kws in DIM_KEYWORDS.items()
          if any(k in low for k in kws)]


def fetch_json(url: str, ua: str, headers: dict[str, str] | None = None) -> Any:
  req = urllib.request.Request(url, headers={"User-Agent": ua, **(headers or {})})
  with urllib.request.urlopen(req, timeout=30) as r:
    return json.load(r)


def get_hn_signals(cfg: VibeCfg) -> list[VibeSignal]:
  cutoff = int((datetime.now(timezone.utc) - timedelta(days=cfg.days)).timestamp())
  signals: list[VibeSignal] = []
  seen_ids: set[str] = set()
  for query in cfg.hn_queries:
    for page in range(cfg.hn_max_pages):
      q = urllib.parse.quote_plus(query)
      url = (f"{HN_API}?tags=comment&query={q}"
             f"&numericFilters=created_at_i>{cutoff}"
             f"&page={page}&hitsPerPage={cfg.hn_hits_per_page}")
      try: data = fetch_json(url, cfg.user_agent)
      except Exception as e:
        print(f"    hn/{query} p{page}: {e}", file=sys.stderr); break
      hits = data.get("hits", [])
      if not hits: break
      signals += get_hn_hit_signals(hits, seen_ids)
      time.sleep(0.3)
    print(f"    hn/{query}: {len(signals)} signals so far")
  return signals


def get_hn_hit_signals(
  hits: list[dict[str, Any]], seen_ids: set[str],
) -> list[VibeSignal]:
  signals: list[VibeSignal] = []
  for h in hits:
    oid = h.get("objectID", "")
    if oid in seen_ids: continue
    seen_ids.add(oid)
    text = html.unescape(re.sub(r"<[^>]+>", " ", h.get("comment_text") or ""))
    models = resolve_models(text)
    if not models: continue
    sent, intensity = infer_sentiment(text)
    dims = infer_dimensions(text)
    ts = datetime.fromtimestamp(h["created_at_i"], tz=timezone.utc).isoformat()
    signals += [VibeSignal(
      model_key=mk, source="hackernews", sentiment=sent, dimensions=dims,
      intensity=intensity, text=text[:500], author=h.get("author", ""),
      url=f"https://news.ycombinator.com/item?id={oid}", timestamp=ts,
      engagement={"points": h.get("points") or 0},
    ) for mk in models]
  return signals


def get_reddit_signals(cfg: VibeCfg) -> list[VibeSignal]:
  signals: list[VibeSignal] = []
  for sub in cfg.reddit_subs:
    url = (
      f"https://old.reddit.com/r/{sub}/comments.json"
      f"?limit={cfg.reddit_limit}&raw_json=1"
    )
    try: data = fetch_json(url, cfg.user_agent, {"Accept": "application/json"})
    except Exception as e:
      print(f"  reddit/{sub}: {e}", file=sys.stderr); continue
    signals += [s for c in data.get("data", {}).get("children", [])
                for s in get_reddit_comment_signals(sub, c.get("data", {}))]
    time.sleep(1.0)
  return signals


def get_reddit_comment_signals(sub: str, d: dict[str, Any]) -> list[VibeSignal]:
  text = d.get("body", "")
  models = resolve_models(text)
  if not models: return []
  sent, intensity = infer_sentiment(text)
  dims = infer_dimensions(text)
  ts = datetime.fromtimestamp(d.get("created_utc", 0), tz=timezone.utc).isoformat()
  return [VibeSignal(
    model_key=mk, source="reddit", sentiment=sent, dimensions=dims,
    intensity=intensity, text=text[:500], author=d.get("author", ""),
    url=f"https://reddit.com{d.get('permalink', '')}", timestamp=ts,
    engagement={"score": d.get("score", 0),
                "controversiality": d.get("controversiality", 0)},
    meta={"subreddit": sub},
  ) for mk in models]


def ingest_signals(path: Path) -> list[VibeSignal]:
  fields = VibeSignal.__dataclass_fields__
  return [VibeSignal(**{k: v for k, v in json.loads(line).items() if k in fields})
          for line in path.read_text(encoding="utf-8").splitlines()
          if line.strip()]


def aggregate_weekly(
  signals: list[VibeSignal], window_days: int = 7,
) -> list[VibeAggregate]:
  by_model: dict[str, list[VibeSignal]] = defaultdict(list)
  for s in signals: by_model[s.model_key].append(s)
  now = datetime.now(timezone.utc)
  w_start, w_end = (now - timedelta(days=window_days)).isoformat(), now.isoformat()
  aggs = [get_vibe_aggregate(mk, sigs, w_start, w_end)
          for mk, sigs in sorted(by_model.items())]
  return sorted([a for a in aggs if a], key=lambda a: a.n_signals, reverse=True)


def get_vibe_aggregate(
  mk: str, sigs: list[VibeSignal], w_start: str, w_end: str,
) -> VibeAggregate | None:
  recent = [s for s in sigs if s.timestamp >= w_start]
  if not recent: return None
  n = len(recent)
  intensities = [s.intensity for s in recent]
  cred_sum = sum(s.author_credibility for s in recent) or 1.0
  cred_w = sum(s.intensity * s.author_credibility for s in recent)
  sent_dist = Counter(s.sentiment for s in recent)
  dims = Counter(d for s in recent for d in s.dimensions)
  mid = n // 2
  first = sum(intensities[:mid]) / max(mid, 1)
  second = sum(intensities[mid:]) / max(n - mid, 1)
  trending = (
    "up" if second - first > 0.1 else
    "down" if second - first < -0.1 else "stable"
  )
  notable = sorted(recent, key=lambda s: sum(s.engagement.values()), reverse=True)[:3]
  return VibeAggregate(
    model_key=mk, window_start=w_start, window_end=w_end,
    n_signals=n, mean_intensity=sum(intensities) / n,
    sentiment_dist=dict(sent_dist),
    top_dimensions=[d for d, _ in dims.most_common(3)],
    trending=trending, credibility_weighted_intensity=cred_w / cred_sum,
    notable_signals=[s.url for s in notable if s.url],
  )


def write_signals(d: Path, signals: list[VibeSignal]) -> Path:
  p = d / "vibes_signals.jsonl"
  with p.open("a", encoding="utf-8") as f:
    for s in signals: f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
  print(f"  wrote {len(signals)} signals -> {p}")
  return p


def write_aggregates(d: Path, aggs: list[VibeAggregate]) -> Path:
  p = d / "vibes_weekly.csv"
  if not aggs: return p
  fields = list(asdict(aggs[0]).keys())
  with p.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for a in aggs:
      row = asdict(a)
      for k in ("sentiment_dist", "top_dimensions", "notable_signals"):
        row[k] = json.dumps(row[k])
      w.writerow(row)
  print(f"  wrote {len(aggs)} aggregates -> {p}")
  return p


def write_trending_md(d: Path, aggs: list[VibeAggregate]) -> Path:
  p = d / "vibes_trending.md"
  lines = [
    "# Vibes Trending",
    f"Window: {aggs[0].window_start[:10]} -> {aggs[0].window_end[:10]}" if aggs else "",
    "", "| model | n | intensity | trending | dims |",
    "| --- | --- | --- | --- | --- |",
  ]
  for a in aggs[:30]:
    dims = ", ".join(a.top_dimensions[:3])
    lines.append(
      f"| {a.model_key} | {a.n_signals} | {a.mean_intensity:+.2f} "
      f"| {a.trending} | {dims} |"
    )
  hot = [a for a in aggs if a.trending == "up" and a.mean_intensity > 0]
  cold = [a for a in aggs if a.trending == "down" or a.mean_intensity < -0.2]
  lines += ["", "## Hot", ""]
  lines += [f"- **{a.model_key}** ({a.n_signals} signals, {a.mean_intensity:+.2f})"
            for a in hot[:10]]
  lines += ["", "## Cold", ""]
  lines += [f"- **{a.model_key}** ({a.n_signals} signals, {a.mean_intensity:+.2f})"
            for a in cold[:10]]
  p.write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(f"  wrote trending -> {p}")
  return p


# %% LLM vibe resolution

def get_model_catalog(
  records_path: Path, all_routes: bool = False,
) -> list[dict[str, Any]]:
  rows = read_data_file(records_path)
  seen: set[str] = set()
  out = []
  for r in rows:
    if not isinstance(r, dict): continue
    if not all_routes and r.get("route") not in FIRST_PARTY: continue
    if (key := r.get("model_key") or r.get("key")) in seen: continue
    seen.add(key)
    out.append({
      "key": key, "route": r.get("route", ""), "raw_id": r.get("raw_id", ""),
      "name": r.get("name", ""), "family": r.get("family", ""),
      "aliases": r.get("aliases", []),
    })
  return out


def get_resolve_items(signals: list[Any]) -> list[dict[str, Any]]:
  return [{
    "signal_id": str(i), "current_model_key": s.get("model_key", ""),
    "source": s.get("source", ""), "text": s.get("text", ""),
    "timestamp": s.get("timestamp", ""), "url": s.get("url", ""),
  } for i, s in enumerate(signals) if isinstance(s, dict) and s.get("text")]


def get_resolve_messages(
  catalog: list[dict[str, Any]], batch: list[dict[str, Any]], args: argparse.Namespace,
) -> list[dict[str, str]]:
  n = args.max_keys if args.allow_multi else 1
  system = (
    "Map vibe comments to canonical model catalog keys. Return strict compact "
    "JSON only: {mappings:[{signal_id:string,resolved_model_keys:string[],"
    "confidence:number,reason_code:string}]}. current_model_key is noisy; "
    "do not use it as evidence. Use only explicit text mentions. Do not map "
    "broad family mentions to versioned keys unless the version appears in "
    f"text. Return at most {n} key(s). Use [] when ambiguous or unsure. "
    f"No prose."
  )
  return [{"role": "system", "content": system}, {"role": "user", "content":
    json.dumps({"catalog": catalog, "signals": batch}, ensure_ascii=False)}]


def get_token_count_value(x: Any) -> int:
  if isinstance(x, dict): return int(x.get("total_tokens") or x.get("totalTokens"))
  return int(getattr(x, "total_tokens", None) or getattr(x, "totalTokens"))


def count_tokens(model: str, messages: list[dict[str, str]]) -> int | None:
  import asyncio
  import litellm
  try:
    return get_token_count_value(asyncio.run(litellm.acount_tokens(
      model=model, messages=messages,
    )))
  except Exception as e:
    print(f"warning: LiteLLM count_tokens endpoint failed: {e}", file=sys.stderr)
  try: return int(litellm.token_counter(model=model, messages=messages))
  except Exception as e:
    print(f"warning: LiteLLM local token count failed: {e}", file=sys.stderr)
    return None


def get_usage_value(u: Any, *keys: str) -> int:
  for k in keys:
    v = u.get(k) if isinstance(u, dict) else getattr(u, k, None)
    if v is not None: return int(v)
  return 0


def record_call(args: argparse.Namespace, stats: dict[str, Any]) -> None:
  args.calls += 1
  args.run_cost += stats["call_cost_usd"]
  args.run_input_tokens += stats["input_tokens"]
  args.run_output_tokens += stats["output_tokens"]
  args.run_total_tokens += stats["total_tokens"]
  row = {**stats, "run_cost_usd": args.run_cost,
         "run_input_tokens": args.run_input_tokens,
         "run_output_tokens": args.run_output_tokens,
         "run_total_tokens": args.run_total_tokens}
  with args.stats_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row) + "\n")
  if args.pbar:
    args.pbar.set_postfix_str(
      f"call ${stats['call_cost_usd']:.4f} run ${args.run_cost:.4f} "
      f"tok {stats['total_tokens']} cache {args.cache_hits}"
    )


def get_completion_json(
  args: argparse.Namespace, messages: list[dict[str, str]], prompt_tokens: int | None,
) -> dict[str, Any]:
  import litellm
  r = litellm.completion(
    model=args.model, messages=messages, temperature=0,
    response_format={"type": "json_object"}, max_tokens=args.max_output_tokens,
  )
  u = getattr(r, "usage", {}) or {}
  in_t = get_usage_value(u, "prompt_tokens", "input_tokens") or prompt_tokens or 0
  out_t = get_usage_value(u, "completion_tokens", "output_tokens")
  stats = {
    "model": args.model, "input_tokens": in_t, "output_tokens": out_t,
    "total_tokens": get_usage_value(u, "total_tokens") or in_t + out_t,
    "call_cost_usd": float(litellm.completion_cost(completion_response=r) or 0),
  }
  record_call(args, stats)
  text = r.choices[0].message.content or "{}"
  try: return json.loads(text)
  except json.JSONDecodeError as e:
    bad = args.outdir / f"bad_resolver_json.{args.calls}.txt"
    bad.write_text(text, encoding="utf-8")
    reason = getattr(r.choices[0], "finish_reason", "")
    raise ValueError(
      f"invalid resolver JSON ({e}); finish={reason}; "
      f"chars={len(text)}; saved={bad}"
    ) from e


def has_version_evidence(key: str, text: str) -> bool:
  low = re.sub(r"[^a-z0-9]+", " ", text.lower())
  norm_key = re.sub(r"[^a-z0-9]+", " ", key.lower())
  specials = re.findall(r"\b(?:o\d|\d+[a-z]+)\b", norm_key)
  if specials: return all(re.search(rf"\b{re.escape(x)}\b", low) for x in specials)
  nums = [n for n in re.findall(r"\d+", key) if len(n) < 6]
  return not nums or re.search(rf"\b{' '.join(nums)}\b", low) is not None


def sanitize_resolutions(
  rows: list[dict[str, Any]], keys: set[str], args: argparse.Namespace,
  batch: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  by_id = {x["signal_id"]: x for x in batch}
  out = []
  for r in rows:
    conf = float(r.get("confidence") or 0)
    text = by_id.get(str(r.get("signal_id")), {}).get("text", "")
    xs = [k for k in r.get("resolved_model_keys", []) if k in keys]
    xs = [k for k in xs if has_version_evidence(k, text)]
    if not args.allow_multi and len(xs) > 1:
      xs = []
      r = {**r, "confidence": 0, "reason_code": "too_many_keys"}
    if len(xs) > args.max_keys:
      xs = []
      r = {**r, "confidence": 0, "reason_code": "too_many_keys"}
    out.append({**r, "resolved_model_keys": xs})
  return out


def get_cache_path(args: argparse.Namespace, batch: list[dict[str, Any]]) -> Path:
  payload = {"model": args.model, "catalog": args.catalog_hash,
             "version": RESOLVER_VERSION, "max_keys": args.max_keys,
             "allow_multi": args.allow_multi, "max_output_tokens": args.max_output_tokens, "batch": batch}
  key = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()
  return args.cache_dir / f"{key}.json"


def get_cached(
  args: argparse.Namespace, batch: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
  if args.dry_run or args.no_cache: return None
  p = get_cache_path(args, batch)
  if not p.exists(): return None
  args.cache_hits += 1
  if args.pbar:
    args.pbar.set_postfix_str(
      f"run ${args.run_cost:.4f} cache {args.cache_hits}"
    )
  return json.loads(p.read_text(encoding="utf-8"))


def save_cache(
  args: argparse.Namespace, batch: list[dict[str, Any]], rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  if not args.dry_run and not args.no_cache:
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    get_cache_path(args, batch).write_text(json.dumps(rows, ensure_ascii=False))
  return rows


def resolve_batch(args: argparse.Namespace, catalog: list[dict[str, Any]],
                  batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
  if cached := get_cached(args, batch): return cached
  if len(batch) > args.max_call_batch:
    mid = len(batch) // 2
    rows = (resolve_batch(args, catalog, batch[:mid]) +
            resolve_batch(args, catalog, batch[mid:]))
    return save_cache(args, batch, rows)
  keys = {m["key"] for m in catalog}
  messages = get_resolve_messages(catalog, batch, args)
  n = count_tokens(args.model, messages)
  if n and n > args.max_tokens and len(batch) > 1:
    mid = len(batch) // 2
    rows = (resolve_batch(args, catalog, batch[:mid]) +
            resolve_batch(args, catalog, batch[mid:]))
    return save_cache(args, batch, rows)
  if n and n > args.max_tokens:
    print(f"warning: one-signal batch has {n} tokens", file=sys.stderr)
  if args.dry_run:
    print(f"dry-run: {len(batch)} signals, {n or 'unknown'} tokens")
    return []
  try: data = get_completion_json(args, messages, n)
  except Exception as e:
    print(f"warning: resolver batch failed: {e}", file=sys.stderr)
    if len(batch) > 1:
      mid = len(batch) // 2
      rows = (resolve_batch(args, catalog, batch[:mid]) +
              resolve_batch(args, catalog, batch[mid:]))
      return save_cache(args, batch, rows)
    rows = [{"signal_id": batch[0]["signal_id"], "resolved_model_keys": [],
             "confidence": 0, "reason": f"resolver failed: {e}"}]
    return save_cache(args, batch, rows)
  rows = data.get("mappings", data if isinstance(data, list) else [])
  return save_cache(args, batch, sanitize_resolutions(rows, keys, args, batch))


def mv_resolved_signals(
  signals: list[Any], resolutions: list[dict[str, Any]], min_conf: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  by_id = {str(r.get("signal_id")): r for r in resolutions
           if float(r.get("confidence", 0) or 0) >= min_conf}
  resolved, unresolved = [], []
  for i, s in enumerate(signals):
    r, keys = by_id.get(str(i)), []
    if r: keys = [k for k in r.get("resolved_model_keys", []) if k]
    if not keys: unresolved.append(s); continue
    for k in keys:
      x = {**s, "model_key": k, "meta": {**s.get("meta", {}), "resolution": r}}
      resolved.append(x)
  return resolved, unresolved


def write_jsonl(path: Path, rows: list[Any]) -> Path:
  path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) +
                  ("\n" if rows else ""), encoding="utf-8")
  return path


def write_resolution_outputs(
  outdir: Path, signals: list[Any], resolutions: list[dict[str, Any]]
) -> dict[str, Path]:
  outdir.mkdir(parents=True, exist_ok=True)
  resolved, unresolved = mv_resolved_signals(signals, resolutions)
  return {
    "resolutions": write_jsonl(outdir / "vibes_resolutions.jsonl", resolutions),
    "resolved": write_jsonl(outdir / "vibes_signals_resolved.jsonl", resolved),
    "unresolved": write_jsonl(outdir / "vibes_unresolved.jsonl", unresolved),
  }


# %% Viewer

def read_data_file(path: Path | None) -> list[Any]:
  if not path or not path.exists(): return []
  text = path.read_text(encoding="utf-8")
  s = text.strip()
  if not s: return []
  if path.suffix.lower() == ".csv":
    return list(csv.DictReader(text.splitlines()))
  if s.startswith("["):
    data = json.loads(s)
    return data if isinstance(data, list) else [data]
  if s.startswith("{") and "\n" not in s:
    return [json.loads(s)]
  return [json.loads(line) for line in text.splitlines() if line.strip()]


def send_bytes(
  h: http.server.BaseHTTPRequestHandler, status: int, ctype: str, data: bytes,
) -> None:
  h.send_response(status)
  h.send_header("Content-Type", ctype)
  h.send_header("Content-Length", str(len(data)))
  h.end_headers()
  h.wfile.write(data)


def send_json(h: http.server.BaseHTTPRequestHandler, value: Any) -> None:
  data = json.dumps(value, ensure_ascii=False).encode("utf-8")
  send_bytes(h, 200, "application/json; charset=utf-8", data)


def build_viewer_handler(records_path: Path | None, vibes_path: Path | None):
  viewer_html = (Path(__file__).resolve().parent / "viewer.html").read_bytes()
  records, vibes = read_data_file(records_path), read_data_file(vibes_path)

  class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
      route = self.path.split("?", 1)[0]
      if route in ("/", "/index.html"):
        send_bytes(self, 200, "text/html; charset=utf-8", viewer_html)
      elif route == "/api/records": send_json(self, records)
      elif route == "/api/vibes": send_json(self, vibes)
      elif route == "/api/files":
        send_json(self, {
          "records": str(records_path or ""),
          "vibes": str(vibes_path or ""),
        })
      else: self.send_error(404)

    def log_message(self, fmt: str, *args: Any) -> None:
      pass

  return Handler


def serve_viewer(
  records_path: Path | None, vibes_path: Path | None, host: str, port: int,
  no_open: bool,
) -> None:
  print(f"records: {records_path or 'none found'}")
  print(f"vibes:   {vibes_path or 'none found'}")
  handler = build_viewer_handler(records_path, vibes_path)
  server = http.server.HTTPServer((host, port), handler)
  url = f"http://{host}:{port}"
  print(f"serving on {url}")
  if not no_open: threading.Timer(0.3, lambda: webbrowser.open(url)).start()
  try: server.serve_forever()
  except KeyboardInterrupt: pass
  finally: server.server_close()
  print("\nshutdown")


# %% Commands

def cmd_models(args: argparse.Namespace) -> None:
  d = setup_models_dir(args.outdir, args.run)
  data, ts = get_api(args.url)
  records = get_records(data, ts)
  paths = write_outputs(d, data, records, ts)
  summary = get_summary(records)
  print(json.dumps({"outdir": str(d), **summary,
                    "paths": {k: str(v) for k, v in paths.items()}}, indent=2))


def cmd_show(args: argparse.Namespace) -> None:
  report = Path(args.run_dir) / "reports" / "report.md"
  print(report.read_text(encoding="utf-8"))


def get_vibe_cfg(args: argparse.Namespace) -> VibeCfg:
  return VibeCfg(
    outdir=args.outdir, run=getattr(args, "run", None), days=args.days,
    hn_queries=args.hn_query or list(HN_QUERIES),
    hn_max_pages=args.hn_max_pages, hn_hits_per_page=args.hn_hits_per_page,
    reddit_subs=args.reddit_sub or list(REDDIT_SUBS),
    reddit_limit=args.reddit_limit, user_agent=args.user_agent,
  )


def cmd_vibes(args: argparse.Namespace) -> None:
  cfg = get_vibe_cfg(args)
  d = setup_vibes_dir(cfg)
  print(f"collecting vibes -> {d}")
  print("  fetching hackernews...")
  hn = get_hn_signals(cfg)
  print(f"    {len(hn)} signals")
  print("  fetching reddit...")
  reddit = get_reddit_signals(cfg)
  print(f"    {len(reddit)} signals")
  signals = hn + reddit
  print(
    f"  total: {len(signals)} signals, "
    f"{len({s.model_key for s in signals})} models"
  )
  write_signals(d, signals)
  aggs = aggregate_weekly(signals, window_days=cfg.days)
  write_aggregates(d, aggs)
  if aggs: write_trending_md(d, aggs)
  print(json.dumps({"outdir": str(d), "signals": len(signals),
                    "models": len({s.model_key for s in signals}),
                    "aggregates": len(aggs)}, indent=2))


def cmd_vibes_aggregate(args: argparse.Namespace) -> None:
  p = args.run_dir / "vibes_signals.jsonl"
  if not p.exists(): sys.exit(f"not found: {p}")
  signals = ingest_signals(p)
  print(f"  loaded {len(signals)} signals from {p}")
  aggs = aggregate_weekly(signals, window_days=args.days)
  write_aggregates(args.run_dir, aggs)
  if aggs: write_trending_md(args.run_dir, aggs)


def cmd_vibes_ingest(args: argparse.Namespace) -> None:
  args.run_dir.mkdir(parents=True, exist_ok=True)
  extra = ingest_signals(args.signals_file)
  print(f"  ingested {len(extra)} manual signals")
  write_signals(args.run_dir, extra)


def cmd_vibes_resolve(args: argparse.Namespace) -> None:
  signals = read_data_file(args.signals_file)
  if args.limit: signals = signals[:args.limit]
  catalog = get_model_catalog(args.models_file, args.all_routes)
  args.catalog_hash = hashlib.sha1(
    json.dumps(catalog, sort_keys=True).encode(),
  ).hexdigest()
  args.cache_dir = args.cache_dir or args.outdir / "cache"
  args.run_cost = 0.0
  args.run_input_tokens = args.run_output_tokens = args.run_total_tokens = 0
  args.calls = args.cache_hits = 0
  args.pbar = None
  run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
  args.stats_path = args.outdir / f"vibes_resolve_calls.{run_id}.jsonl"
  args.summary_path = args.outdir / f"vibes_resolve_summary.{run_id}.json"
  args.outdir.mkdir(parents=True, exist_ok=True)
  args.stats_path.touch()
  items = get_resolve_items(signals)
  batches = [items[i:i + args.batch_size]
             for i in range(0, len(items), args.batch_size)]
  resolutions = []
  with tqdm(batches, desc="resolve vibes", unit="batch") as pbar:
    args.pbar = pbar
    for b in pbar: resolutions.extend(resolve_batch(args, catalog, b))
    args.pbar = None
  paths = write_resolution_outputs(
    args.outdir, signals, resolutions
  )
  summary = {"signals": len(signals), "items": len(items),
             "resolutions": len(resolutions), "calls": args.calls,
             "cache_hits": args.cache_hits,
             "input_tokens": args.run_input_tokens,
             "output_tokens": args.run_output_tokens,
             "total_tokens": args.run_total_tokens,
             "cost_usd": round(args.run_cost, 6),
             "stats": str(args.stats_path), "summary": str(args.summary_path),
             "paths": {k: str(v) for k, v in paths.items()}}
  args.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
  print(json.dumps(summary, indent=2))


def cmd_viewer(args: argparse.Namespace) -> None:
  records_path = args.records or get_latest(args.datadir, "modelsdev_records.jsonl")
  vibes_path = args.vibes or get_latest(args.datadir, "vibes_signals.jsonl")
  serve_viewer(records_path, vibes_path, args.host, args.port, args.no_open)


def add_vibes_args(p: argparse.ArgumentParser) -> None:
  p.add_argument("--days", type=int, default=14)
  p.add_argument("--hn-query", action="append", default=[])
  p.add_argument("--hn-max-pages", type=int, default=20)
  p.add_argument("--hn-hits-per-page", type=int, default=50)
  p.add_argument("--reddit-sub", action="append", default=[])
  p.add_argument("--reddit-limit", type=int, default=500)
  p.add_argument("--user-agent", default=VIBES_UA)


def build_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description="Models.dev metadata/vibes CLI")
  sub = p.add_subparsers(dest="cmd")
  models = sub.add_parser("models", help="fetch and normalize Models.dev metadata")
  models.add_argument("--url", default=API_URL)
  models.add_argument("--outdir", type=Path)
  models.add_argument("--run")
  models.set_defaults(fn=cmd_models)
  show = sub.add_parser("show", help="print a generated report")
  show.add_argument("run_dir", type=Path)
  show.set_defaults(fn=cmd_show)
  vibes = sub.add_parser("vibes", help="collect and aggregate public model vibes")
  vibes.add_argument("--outdir", type=Path)
  vibes.add_argument("--run")
  add_vibes_args(vibes)
  vibes.set_defaults(fn=cmd_vibes)
  agg = sub.add_parser("vibes-aggregate", help="aggregate an existing vibes run")
  agg.add_argument("run_dir", type=Path)
  agg.add_argument("--days", type=int, default=14)
  agg.set_defaults(fn=cmd_vibes_aggregate)
  ingest = sub.add_parser("vibes-ingest", help="append manual vibe signals")
  ingest.add_argument("run_dir", type=Path)
  ingest.add_argument("signals_file", type=Path)
  ingest.set_defaults(fn=cmd_vibes_ingest)
  resolve = sub.add_parser("vibes-resolve", help="LLM-resolve vibe model keys")
  resolve.add_argument("signals_file", type=Path)
  resolve.add_argument("models_file", type=Path)
  resolve.add_argument("--outdir", type=Path, default=Path("vibes-resolved"))
  resolve.add_argument("--model", default="gemini/gemini-2.5-flash")
  resolve.add_argument("--batch-size", type=int, default=100)
  resolve.add_argument("--max-call-batch", type=int, default=10)
  resolve.add_argument("--max-tokens", type=int, default=800_000)
  resolve.add_argument("--max-output-tokens", type=int, default=None)
  resolve.add_argument("--max-keys", type=int, default=1)
  resolve.add_argument("--allow-multi", action="store_true")
  resolve.add_argument("--limit", type=int, default=0)
  resolve.add_argument("--cache-dir", type=Path)
  resolve.add_argument("--no-cache", action="store_true")
  resolve.add_argument("--all-routes", action="store_true")
  resolve.add_argument("--dry-run", action="store_true")
  resolve.set_defaults(fn=cmd_vibes_resolve)
  viewer = sub.add_parser("viewer", help="serve viewer.html with latest data")
  viewer.add_argument("--records", type=Path)
  viewer.add_argument("--vibes", type=Path)
  viewer.add_argument("--datadir", type=Path, default=DATADIR / "out" / TOOL)
  viewer.add_argument("--host", default="127.0.0.1")
  viewer.add_argument("--port", type=int, default=8877)
  viewer.add_argument("--no-open", action="store_true")
  viewer.set_defaults(fn=cmd_viewer)
  return p


def main(argv: list[str] | None = None) -> None:
  argv = ["models"] if argv is None and len(sys.argv) == 1 else argv
  args = build_parser().parse_args(argv)
  if not hasattr(args, "fn"): build_parser().error("missing command")
  args.fn(args)


if __name__ == "__main__":
  main()
