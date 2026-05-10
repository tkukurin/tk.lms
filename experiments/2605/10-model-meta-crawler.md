# 260510 — Model Metadata Meta-Crawler

Thesis: an evidence-backed meta-crawler can identify stale, conflicting, or underspecified LLM model metadata across open-source agent/gateway repositories by normalizing each source into a shared schema, tracking historical changes with PyDriller, and reporting discrepancies with enough provenance for maintainers to decide which source is wrong.

## Background

Starting points and related sources found during reconnaissance:

- `anomalyco/opencode` — open-source coding agent; it uses Models.dev internally and transforms Models.dev records into its provider runtime schema.[^opencode]
- `anomalyco/models.dev` — canonical upstream for OpenCode model metadata; data lives as provider/model TOML files and is exposed as `https://models.dev/api.json`; current public repo has thousands of model/provider files and frequent model-update commits.[^modelsdev]
- `earendil-works/pi` — Pi agent monorepo; `@earendil-works/pi-ai` contains a generated model registry (`packages/ai/src/models.generated.ts`) plus generator scripts and provider adapters.[^pi]
- `BerriAI/litellm` — gateway/SDK with `model_prices_and_context_window.json`, broad provider coverage, pricing, context windows, modes, and many capability flags.[^litellm]
- `Aider-AI/aider` — agent-facing consumer of LLM metadata; likely useful as a downstream sanity check rather than a primary source because it depends heavily on LiteLLM-style model naming.[^aider]
- Non-repo APIs worth adding as first-class adapters: OpenRouter `/api/v1/models`, Models.dev `/api.json`, Vercel AI Gateway model catalog, provider-owned docs/APIs where stable.

## Approach

### Step 1: Define the normalized model record

Create a source-independent record that preserves both raw source data and normalized fields:

- identity: `source`, `provider`, `raw_id`, route-level `key`, optional `underlying_key`, aliases, route/region/variant
- limits: context, input, output, max tokens, tier-specific limits
- costs: input/output/cache-read/cache-write/reasoning/audio/image/video, normalized to USD per 1M tokens where possible
- capabilities: tool calling, structured output, JSON/response schema, reasoning, reasoning effort levels, attachments, modalities, web search, prompt caching, system messages, parallel tools
- lifecycle: release date, knowledge cutoff, last updated, status/deprecation date
- transport: endpoint mode (`chat`, `responses`, `messages`, `anthropic`, `openai-compatible`), base URL, SDK/provider package
- provenance: repository URL, file path, commit hash, commit date, commit message, extraction timestamp, parser version

Keep raw fields alongside normalized fields so every discrepancy can be traced back to the source value.

Critique of the first sketch:

- `src` conflates adapter provenance with provider route identity.
- `caps: set[...]` loses the false/unknown distinction.
- `limits` and `costs_1m` are right as dicts, but need fixed keys and units.
- `meta: dict[str, str]` is too narrow for dates, booleans, lists, and numbers.
- `raw` is not provenance; normalized rows need openable source location data.
- Provider route and underlying model are distinct facts. OpenRouter, Bedrock,
  Vertex, and first-party records may legitimately differ.

Updated suggestion, still one shallow type plus one capability alias:

```python
from dataclasses import dataclass, field
from typing import Any, Literal

Capability = Literal[
    "tools", "parallel_tools", "structured_output", "json_mode",
    "reasoning", "reasoning_effort", "prompt_cache", "system_message",
    "web_search", "attachments", "vision_input", "audio_input", "audio_output",
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
```

Conventions:

- `source`: adapter/source, e.g. `litellm_repo`, `modelsdev_api`.
- `provider`: normalized route/provider, e.g. `openai`, `openrouter`, `bedrock`.
- `key`: route-level comparison key, usually `{provider}/{normalized_raw_id}`.
- `underlying_key`: first-party model key only when confidently known.
- `capabilities`: missing means unknown; `False` means explicitly unsupported.
- `limits`: token counts; expected keys: `context`, `input`, `output`,
  `max_completion`, `batch_context`.
- `costs_usd_1m`: USD per 1M units; expected keys: `input`, `output`,
  `cache_read`, `cache_write`, `reasoning`, `audio_input`, `audio_output`,
  `image_input`, `image_output`, `video_input`.
- `meta`: normalized extras; expected keys include `input_modalities`,
  `output_modalities`, `endpoint`, `base_url`, `region`, `route_variant`,
  `release_date`, `knowledge_cutoff`, `updated_at`, `status`,
  `deprecation_date`, `tier`.
- `provenance`: repository/API URL, path, commit, extraction timestamp, parser.
- `raw`: original payload or minimally extracted source row.

Keep `key` route-specific by default. Use `underlying_key` only from adapter facts
or an explicit high-confidence alias map. Compare route facts (`costs_usd_1m`,
regional limits) separately from underlying-model facts (`release_date`, broad
modalities).

### Step 2: Build source adapters

Adapters return normalized records and source provenance. Initial adapters:

- `modelsdev_repo`: clone `anomalyco/models.dev`, parse provider/model TOML, resolve `extends` before normalization.
- `modelsdev_api`: fetch `https://models.dev/api.json` to detect build-time/API differences from repository TOML.
- `opencode_repo`: parse `packages/opencode/src/provider` transformations and any bundled/generated model artifacts; primarily validates how OpenCode consumes Models.dev rather than acting as a separate canonical source.
- `pi_repo`: parse `packages/ai/src/models.generated.ts` and generator inputs/scripts, preserving `api`, `baseUrl`, `contextWindow`, `maxTokens`, `thinkingLevelMap`, `compat`, headers, and modalities.
- `litellm_repo`: parse `model_prices_and_context_window.json`, convert per-token fields to per-1M-token, map `supports_*` flags to capabilities.
- `openrouter_api`: fetch current OpenRouter catalog and normalize provider-prefixed IDs.

Each adapter should save raw extracts after every major step under `data/out/model-meta-crawler/2605/<run>/raw/<source>/` and normalized JSONL under `data/out/model-meta-crawler/2605/<run>/normalized/`.

### Step 3: Use PyDriller for change history

Use PyDriller for repository analysis, not for one-off HEAD reads:

- Traverse only metadata-relevant paths to avoid crawling entire monorepos unnecessarily.
- For each commit touching relevant files, capture commit hash, date, author, message, modified path, change type, and before/after source snippets.
- Re-run the source adapter at selected commits or parse `source_code_before` / `source_code` directly when cheap.
- Emit a model metadata event stream: `source`, `commit`, `field`, `old`, `new`, `raw_model_id`, `canonical_model_key`.
- Measure lag: when one source updates a model or field, how long until other sources converge or disagree.

Relevant initial path filters:

- Models.dev: `providers/**`, `packages/core/src/schema.ts`, generator/build scripts.
- OpenCode: `packages/opencode/src/provider/**`, model generation/import code, config schema.
- Pi: `packages/ai/src/models.generated.ts`, `packages/ai/src/models.ts`, `packages/ai/scripts/generate-models.ts`, provider adapters.
- LiteLLM: `model_prices_and_context_window.json`, provider support JSON, auto-update workflows/scripts.

### Step 4: Canonicalize identities cautiously

Canonicalization is the hardest part. Start conservative:

- Exact provider+ID matches are direct comparisons.
- Known alias rules handle date suffixes, `latest`, provider prefixes, region prefixes (`us.`, `eu.`, `global.`), and route prefixes (`openrouter/`, `vertex_ai/`, `bedrock/`).
- Do not collapse provider-wrapped variants by default; compare them as provider routes unless an explicit alias map says they are the same underlying model.
- Maintain an editable alias map with confidence levels: `exact`, `wrapper`, `family`, `unknown`.

This avoids false positives like treating OpenRouter `anthropic/claude-sonnet-4.6` and first-party Anthropic `claude-sonnet-4-6` as identical in all limits/costs when OpenRouter may legitimately differ.

### Step 5: Score discrepancies

Report discrepancies as structured findings, not a flat diff.

- `critical`: capability conflict likely to break clients, e.g. tool calling or structured output true vs false.
- `high`: large context/output/cost disagreement, missing deprecation, or model exists in one primary source but not another after alias matching.
- `medium`: naming, date, modality, or route metadata disagreement.
- `low`: formatting/rounding/source-only differences.

Numeric tolerances:

- Costs: exact after unit normalization unless source documents region/provider markups; otherwise flag relative deltas over 1%.
- Limits: exact for hard output/context limits; allow documented route-specific reductions.
- Dates: exact for release/deprecation; missing dates are separate from conflicts.

### Step 6: Collect user vibes

The technical metadata tells you what a model *can* do and what it costs.
User vibes tell you whether it *actually works well* for real tasks — and
whether the crowd thinks it's getting better or worse over time.

This is a different kind of data: subjective, temporal, noisy, and
aggregated from many sources. It should NOT live inside `NormalizedRecord`.
Instead, introduce a parallel signal type that links to models via the same
`key`/`underlying_key` identity system.

#### Data model

```python
from dataclasses import dataclass, field
from typing import Any, Literal

Sentiment = Literal["positive", "negative", "neutral", "mixed"]
VibeSource = Literal[
    "twitter", "reddit", "hackernews", "discord",
    "blog", "benchmark", "changelog", "manual",
]
VibeDimension = Literal[
    "overall", "coding", "reasoning", "instruction_following",
    "creativity", "speed", "reliability", "cost_value",
    "regression",  # "it got worse"
]

@dataclass
class VibeSignal:
    """One observation: a tweet, post, thread, benchmark note, etc."""
    model_key: str                          # links to NormalizedRecord.key or underlying_key
    source: VibeSource
    sentiment: Sentiment
    dimensions: list[VibeDimension] = field(default_factory=list)
    intensity: float = 0.0                  # -1.0 (very negative) to +1.0 (very positive)
    text: str = ""                          # original snippet (truncated)
    author: str = ""                        # handle/username, empty if anonymous
    author_credibility: float = 0.5         # 0=unknown, 1=known-expert/benchmarker
    url: str = ""                           # source URL
    timestamp: str = ""                     # ISO 8601
    engagement: dict[str, int] = field(default_factory=dict)  # likes, retweets, replies
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class VibeAggregate:
    """Rolled-up sentiment per model per time window."""
    model_key: str
    window_start: str                       # ISO date
    window_end: str
    n_signals: int = 0
    mean_intensity: float = 0.0
    sentiment_dist: dict[Sentiment, int] = field(default_factory=dict)
    top_dimensions: list[VibeDimension] = field(default_factory=list)
    trending: Literal["up", "down", "stable", "unknown"] = "unknown"
    credibility_weighted_intensity: float = 0.0  # weight by author_credibility
    notable_signals: list[str] = field(default_factory=list)  # URLs of high-engagement signals
```

#### Design principles

- **Separate storage**: vibes go in `data/out/model-meta-crawler/<run>/vibes/`,
  not mixed with normalized technical records.
- **Link via identity**: the `model_key` field uses the same canonicalization
  from Step 4. Mapping "claude sonnet" in a tweet to
  `anthropic/claude-sonnet-4-20250514` is the hard part — reuse alias maps.
- **Temporal first-class**: every signal has a timestamp; aggregates are
  windowed (daily/weekly). Vibes decay — a rave review from 3 months ago
  matters less than this week's complaints.
- **Credibility weighting**: a known ML researcher or benchmark operator's
  signal > random account. Start with manual tiers, automate later.
- **Dimension tagging**: "coding is worse" maps to `["coding", "regression"]`;
  "fastest model I've used" maps to `["speed"]`. This lets you cross-reference
  vibes against technical capabilities.
- **No ground truth**: unlike technical metadata, vibes have no "correct" value.
  Report distributions and trends, not discrepancies.

#### Sources (initial)

- **Twitter/X**: search for model names + sentiment keywords; filter by
  engagement thresholds to reduce noise. Use API or scraping (nitter-style).
- **Reddit** (r/LocalLLaMA, r/ChatGPT, r/MachineLearning): post/comment
  sentiment about specific models.
- **Hacker News**: comments on model release/benchmark threads.
- **Benchmark changelogs**: when Chatbot Arena, LiveBench, or SWE-bench
  publish new results, the relative position change *is* a vibe signal.
- **Manual/curated**: your own notes or high-signal threads you bookmark.

#### Entity resolution challenge

Mapping informal model mentions to canonical keys is harder than repo parsing:
- "sonnet" → which version? Use recency heuristic + context.
- "gpt5" vs "gpt-5" vs "o3" — need fuzzy mention matcher.
- Nicknames/memes ("Claude the Coder") — maintain a slang alias map.
- Multi-model comparisons ("sonnet > opus for coding") yield signals for both.

Start with exact/substring matching against the alias map from Step 4,
plus a manual review queue for unresolved mentions.

#### Outputs

- `vibes_signals.jsonl` — raw individual signals with provenance.
- `vibes_weekly.csv` — per-model weekly aggregates.
- `vibes_trending.md` — human-readable "what's hot / what's not" summary.
- Join with `normalized_records.jsonl` on `model_key` for the viewer:
  show technical facts + crowd sentiment side by side.

### Step 7: Produce reviewable reports

Outputs for each run:

- `raw_records.jsonl` — source-specific raw extracts with provenance.
- `normalized_records.jsonl` — normalized records used for comparisons.
- `discrepancies.csv` — one finding per row, including severity and field path.
- `discrepancies.md` — human-readable summary grouped by model/provider.
- `timeline.csv` — PyDriller-derived change events and lag calculations.
- `alias_review.csv` — proposed low-confidence alias links for manual review.

Every report row should include source file/API provenance so a maintainer can open the exact upstream location.

## Experiment matrix

- Phase 1: current-state comparison for `models.dev`, Pi, LiteLLM, OpenRouter API.
- Phase 2: add OpenCode consumer-level validation and Models.dev API-vs-repo checks.
- Phase 3: historical lag analysis with PyDriller over the last 90 days or last N metadata-touching commits.
- Phase 4: provider-doc spot checks for a handpicked set of high-value models.
- Phase 5: vibes collection from Twitter + Reddit for focus model families; weekly aggregates; viewer integration.

Initial model families for manual validation:

- OpenAI GPT-5.x / Codex variants
- Anthropic Claude Sonnet/Opus/Haiku 4.x
- Google Gemini 2.5/3.x
- DeepSeek V3.2/R1 variants
- Moonshot Kimi K2.x
- MiniMax M2.x
- Z.ai GLM 4.6/4.7/5.x

## Decisions

- **D1: Treat Models.dev as a source, not ground truth** — OpenCode uses it internally, but the crawler's purpose is to detect disagreement; no source should be hard-coded as correct.
- **D2: Store raw and normalized records** — discrepancies need auditability and parser debugging.
- **D3: Keep provider route separate from underlying model** — wrappers legitimately change costs, limits, and capabilities.
- **D4: Use PyDriller for history, direct parsers for snapshots** — PyDriller is ideal for commit/event extraction; source-specific parsers are cleaner for current-state normalization.
- **D5: Start with static files and public APIs** — avoids API keys and makes the first crawler reproducible.
- **D6: Prefer explicit alias maps over fuzzy matching** — fuzzy identity matching is useful for candidates but too risky for final discrepancies.
- **D7: Save after each major step** — raw fetch, normalized records, comparisons, and reports should be persisted independently for post-mortem debugging.
- **D8: Keep vibes separate from technical metadata** — `NormalizedRecord` stays factual; `VibeSignal` is a parallel data stream joined on `model_key`. No schema changes needed to the existing record type.
- **D9: Vibes are distributions, not verdicts** — report sentiment distributions and trends; never present aggregated vibes as factual claims about model quality.

## Success criteria

- [ ] Parse at least four primary sources: Models.dev repo/API, Pi, LiteLLM, and OpenRouter API.
- [ ] Normalize at least 90% of parsed chat/text-generation records into shared identity, limit, cost, and capability fields.
- [ ] Produce a discrepancy report with provenance for every compared field.
- [ ] Correctly unit-normalize LiteLLM per-token prices and Models.dev/Pi per-1M-token prices.
- [ ] Recover PyDriller change timelines for at least two repositories over a bounded period.
- [ ] Identify at least 10 actionable discrepancies or confirm low discrepancy rate with supporting CSV evidence.
- [ ] Keep false-positive rate under 20% on a manual review sample of 50 findings.
- [ ] Collect vibes for at least 10 focus models over a 2-week window.
- [ ] Resolve ≥80% of model mentions to canonical keys without manual intervention.
- [ ] Produce a weekly vibes summary joinable with technical metadata in the viewer.

## Risks and mitigations

- Identity normalization can over-collapse unrelated route variants. Mitigation: conservative matching and alias confidence labels.
- Generated TypeScript model registries may be hard to parse robustly. Mitigation: prefer generator inputs where available; otherwise use a TS/JS parser or evaluate in a sandboxed subprocess without network/API keys.
- Public APIs may change between runs. Mitigation: persist raw API responses with timestamps.
- Repositories may be large. Mitigation: shallow clone for snapshots; PyDriller path filters and bounded date windows for history.
- Some differences are legitimate provider-specific behavior. Mitigation: model comparisons should distinguish first-party, wrapper, region, and tier variants.
- Vibe signals are noisy and biased toward vocal minorities. Mitigation: credibility weighting, engagement thresholds, and always reporting N alongside aggregates.
- Model mention resolution is ambiguous ("sonnet" could be any version). Mitigation: recency heuristic + context window; unresolved mentions go to review queue, not auto-assigned.
- Twitter/X API access is unstable and expensive. Mitigation: support multiple collection methods (API, RSS, manual paste); degrade gracefully to fewer sources.

## Next implementation slice

1. Keep the experiment entrypoint consolidated in `experiments/2605/metacli.py`; split into `src/tk/` only when reuse justifies it.
2. Add `pydriller` plus TOML/JS parsing dependencies if missing.
3. Implement adapters for Models.dev API, LiteLLM JSON, and Pi generated registry first.
4. Emit normalized JSONL and a minimal `discrepancies.csv` for the initial model-family whitelist.
5. Add PyDriller timelines after the static adapters produce useful current-state diffs.

[^opencode]: https://github.com/anomalyco/opencode
[^modelsdev]: https://github.com/anomalyco/models.dev
[^pi]: https://github.com/earendil-works/pi
[^litellm]: https://github.com/BerriAI/litellm
[^aider]: https://github.com/Aider-AI/aider
