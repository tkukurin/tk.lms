# %%
from __future__ import annotations

import csv, html, json, re, sys, time, urllib.parse, urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import simple_parsing as sp

# ── Config ────────────────────────────────────────────────────────────────

@dataclass
class Cfg:
  cmd: str = "run"                         # run | aggregate | ingest
  outdir: Path | None = None
  models_jsonl: Path | None = None
  days: int = 14
  run_dir: Path | None = None              # for aggregate/ingest
  signals_file: Path | None = None         # for ingest
  hn_queries: list[str] = field(
    default_factory=lambda: ["Claude", "GPT", "Gemini",
                             "LLM", "Sonnet", "deepseek",
                             "grok", "llama", "kimi"])
  hn_max_pages: int = 20
  hn_hits_per_page: int = 50
  reddit_subs: list[str] = field(
    default_factory=lambda: ["LocalLLaMA", "ChatGPT",
                             "MachineLearning", "artificial",
                             "ClaudeAI", "OpenAI",
                             "singularity"])
  reddit_limit: int = 500
  user_agent: str = "tk-vibes-collector/0.1"
  root: Path = field(
    default_factory=lambda: Path(__file__).resolve().parents[2])
  tool: str = "model-meta-crawler"


# ── Types ─────────────────────────────────────────────────────────────────

Sentiment = Literal["positive", "negative", "neutral", "mixed"]
VibeSource = Literal[
  "twitter", "reddit", "hackernews", "discord",
  "blog", "benchmark", "changelog", "manual",
]
VibeDimension = Literal[
  "overall", "coding", "reasoning", "instruction_following",
  "creativity", "speed", "reliability", "cost_value", "regression",
]


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


# ── Model mention resolution ─────────────────────────────────────────────

MODEL_PATTERNS: list[tuple[re.Pattern, str]] = [
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
  "coding": ["coding", "code", "programming", "swe-bench",
             "aider", "copilot", "vscode", "cursor"],
  "reasoning": ["reasoning", "math", "logic", "think",
                "chain of thought", "cot"],
  "speed": ["fast", "slow", "latency", "speed", "ttft",
            "tokens per second", "tps"],
  "reliability": ["reliable", "consistent", "hallucin",
                  "refuses", "lazy", "truncat"],
  "regression": ["worse", "nerfed", "downgrad", "regress",
                 "lobotom", "dumber", "degraded"],
  "cost_value": ["cheap", "expensive", "cost", "pricing",
                 "value", "bang for buck"],
  "creativity": ["creative", "writing", "prose", "story",
                 "poetry"],
  "instruction_following": ["instruction", "follows",
                            "obedient", "system prompt"],
}

POS_KW = {"great", "amazing", "excellent", "love", "best",
           "impressive", "fantastic", "goat", "insane"}
NEG_KW = {"terrible", "awful", "worse", "bad", "trash",
           "garbage", "disappointing", "broken", "unusable",
           "nerfed", "lobotomized", "dumber", "sucks"}


# ── Resolution & inference ────────────────────────────────────────────────

def resolve_models(text: str) -> list[str]:
  seen: set[str] = set()
  keys = [k for pat, k in MODEL_PATTERNS
          if pat.search(text) and not (k in seen or seen.add(k))]
  return [k for k in keys
          if not any(o.startswith(k) and o != k for o in keys)]


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


# ── Adapters ──────────────────────────────────────────────────────────────

def fetch_json(url: str, ua: str, headers: dict | None = None) -> Any:
  hdrs = {"User-Agent": ua, **(headers or {})}
  req = urllib.request.Request(url, headers=hdrs)
  with urllib.request.urlopen(req, timeout=30) as r:
    return json.load(r)


HN_API = "https://hn.algolia.com/api/v1/search_by_date"

def get_hn_signals(cfg: Cfg) -> list[VibeSignal]:
  cutoff = int((datetime.now(timezone.utc)
                - timedelta(days=cfg.days)).timestamp())
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
        print(f"    hn/{query} p{page}: {e}",
              file=sys.stderr); break
      hits = data.get("hits", [])
      if not hits: break
      for h in hits:
        oid = h.get("objectID", "")
        if oid in seen_ids: continue
        seen_ids.add(oid)
        text = html.unescape(
          re.sub(r"<[^>]+>", " ", h.get("comment_text") or ""))
        models = resolve_models(text)
        if not models: continue
        sent, intensity = infer_sentiment(text)
        dims = infer_dimensions(text)
        ts = datetime.fromtimestamp(
          h["created_at_i"], tz=timezone.utc).isoformat()
        for mk in models:
          signals.append(VibeSignal(
            model_key=mk, source="hackernews",
            sentiment=sent, dimensions=dims, intensity=intensity,
            text=text[:500], author=h.get("author", ""),
            url=f"https://news.ycombinator.com/item?id={oid}",
            timestamp=ts,
            engagement={"points": h.get("points") or 0},
          ))
      time.sleep(0.3)
    print(f"    hn/{query}: {len(signals)} signals so far")
  return signals


def get_reddit_signals(cfg: Cfg) -> list[VibeSignal]:
  signals: list[VibeSignal] = []
  for sub in cfg.reddit_subs:
    url = (f"https://old.reddit.com/r/{sub}/comments.json"
           f"?limit={cfg.reddit_limit}&raw_json=1")
    try:
      data = fetch_json(url, cfg.user_agent,
                        {"Accept": "application/json"})
    except Exception as e:
      print(f"  reddit/{sub}: {e}", file=sys.stderr); continue
    for c in data.get("data", {}).get("children", []):
      d = c.get("data", {})
      text = d.get("body", "")
      models = resolve_models(text)
      if not models: continue
      sent, intensity = infer_sentiment(text)
      dims = infer_dimensions(text)
      ts = datetime.fromtimestamp(
        d.get("created_utc", 0), tz=timezone.utc).isoformat()
      for mk in models:
        signals.append(VibeSignal(
          model_key=mk, source="reddit",
          sentiment=sent, dimensions=dims, intensity=intensity,
          text=text[:500], author=d.get("author", ""),
          url=f"https://reddit.com{d.get('permalink', '')}",
          timestamp=ts,
          engagement={"score": d.get("score", 0),
                      "controversiality":
                        d.get("controversiality", 0)},
          meta={"subreddit": sub},
        ))
    time.sleep(1.0)
  return signals


def ingest_signals(path: Path) -> list[VibeSignal]:
  return [VibeSignal(**{k: v for k, v in json.loads(line).items()
                        if k in VibeSignal.__dataclass_fields__})
          for line in path.read_text().splitlines()
          if line.strip()]


# ── Aggregation ───────────────────────────────────────────────────────────

def aggregate_weekly(
  signals: list[VibeSignal], window_days: int = 7,
) -> list[VibeAggregate]:
  by_model: dict[str, list[VibeSignal]] = defaultdict(list)
  for s in signals: by_model[s.model_key].append(s)
  now = datetime.now(timezone.utc)
  w_start = (now - timedelta(days=window_days)).isoformat()
  w_end = now.isoformat()
  aggs = []
  for mk, sigs in sorted(by_model.items()):
    recent = [s for s in sigs if s.timestamp >= w_start]
    if not recent: continue
    n = len(recent)
    intensities = [s.intensity for s in recent]
    cred_sum = sum(s.author_credibility for s in recent) or 1.0
    cred_w = sum(s.intensity * s.author_credibility
                 for s in recent)
    sent_dist = Counter(s.sentiment for s in recent)
    dims = Counter(d for s in recent for d in s.dimensions)
    mid = n // 2
    first = sum(intensities[:mid]) / max(mid, 1)
    second = sum(intensities[mid:]) / max(n - mid, 1)
    delta = second - first
    trending = ("up" if delta > 0.1 else
                "down" if delta < -0.1 else "stable")
    notable = sorted(
      recent, key=lambda s: sum(s.engagement.values()),
      reverse=True)[:3]
    aggs.append(VibeAggregate(
      model_key=mk, window_start=w_start, window_end=w_end,
      n_signals=n, mean_intensity=sum(intensities) / n,
      sentiment_dist=dict(sent_dist),
      top_dimensions=[d for d, _ in dims.most_common(3)],
      trending=trending,
      credibility_weighted_intensity=cred_w / cred_sum,
      notable_signals=[s.url for s in notable if s.url],
    ))
  return sorted(aggs, key=lambda a: a.n_signals, reverse=True)


# ── Output ────────────────────────────────────────────────────────────────

def setup_outdir(cfg: Cfg) -> Path:
  base = cfg.outdir or cfg.root / "data" / "out" / cfg.tool
  d = base / datetime.now().strftime("%y%m") / "vibes"
  d.mkdir(parents=True, exist_ok=True)
  return d


def write_signals(d: Path, signals: list[VibeSignal]) -> Path:
  p = d / "vibes_signals.jsonl"
  with p.open("a", encoding="utf-8") as f:
    for s in signals:
      f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
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
      for k in ("sentiment_dist", "top_dimensions",
                "notable_signals"):
        row[k] = json.dumps(row[k])
      w.writerow(row)
  print(f"  wrote {len(aggs)} aggregates -> {p}")
  return p


def write_trending_md(d: Path, aggs: list[VibeAggregate]) -> Path:
  p = d / "vibes_trending.md"
  lines = [
    "# Vibes Trending",
    (f"Window: {aggs[0].window_start[:10]} -> "
     f"{aggs[0].window_end[:10]}") if aggs else "",
    "", "| model | n | intensity | trending | dims |",
    "| --- | --- | --- | --- | --- |",
  ]
  for a in aggs[:30]:
    dims = ", ".join(a.top_dimensions[:3])
    lines.append(f"| {a.model_key} | {a.n_signals} | "
                 f"{a.mean_intensity:+.2f} | {a.trending} "
                 f"| {dims} |")
  hot = [a for a in aggs
         if a.trending == "up" and a.mean_intensity > 0]
  cold = [a for a in aggs
          if a.trending == "down" or a.mean_intensity < -0.2]
  lines += ["", "## Hot", ""]
  lines += [f"- **{a.model_key}** ({a.n_signals} signals, "
            f"{a.mean_intensity:+.2f})" for a in hot[:10]]
  lines += ["", "## Cold", ""]
  lines += [f"- **{a.model_key}** ({a.n_signals} signals, "
            f"{a.mean_intensity:+.2f})" for a in cold[:10]]
  p.write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(f"  wrote trending -> {p}")
  return p


# ── Commands ──────────────────────────────────────────────────────────────

def cmd_run(cfg: Cfg) -> None:
  d = setup_outdir(cfg)
  print(f"collecting vibes -> {d}")
  print("  fetching hackernews...")
  hn = get_hn_signals(cfg)
  print(f"    {len(hn)} signals")
  print("  fetching reddit...")
  reddit = get_reddit_signals(cfg)
  print(f"    {len(reddit)} signals")
  all_signals = hn + reddit
  print(f"  total: {len(all_signals)} signals, "
        f"{len({s.model_key for s in all_signals})} models")
  write_signals(d, all_signals)
  aggs = aggregate_weekly(all_signals, window_days=cfg.days)
  write_aggregates(d, aggs)
  if aggs: write_trending_md(d, aggs)
  print(json.dumps({"outdir": str(d), "signals": len(all_signals),
                    "models": len({s.model_key for s in all_signals}),
                    "aggregates": len(aggs)}, indent=2))


def cmd_aggregate(cfg: Cfg) -> None:
  d = cfg.run_dir or sys.exit("need --run_dir")
  p = d / "vibes_signals.jsonl"
  if not p.exists(): sys.exit(f"not found: {p}")
  signals = ingest_signals(p)
  print(f"  loaded {len(signals)} signals from {p}")
  aggs = aggregate_weekly(signals, window_days=cfg.days)
  write_aggregates(d, aggs)
  if aggs: write_trending_md(d, aggs)


def cmd_ingest(cfg: Cfg) -> None:
  d = cfg.run_dir or sys.exit("need --run_dir")
  d.mkdir(parents=True, exist_ok=True)
  f = cfg.signals_file or sys.exit("need --signals_file")
  extra = ingest_signals(f)
  print(f"  ingested {len(extra)} manual signals")
  write_signals(d, extra)


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
  cfg = sp.parse(Cfg)
  {"run": cmd_run, "aggregate": cmd_aggregate,
   "ingest": cmd_ingest}[cfg.cmd](cfg)


if __name__ == "__main__":
  main()
