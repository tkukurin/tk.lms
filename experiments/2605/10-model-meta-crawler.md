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

No published academic baseline seems required for this first phase: the task is an engineering data quality experiment, not model evaluation.

## Approach

### Step 1: Define the normalized model record

Create a source-independent record that preserves both raw source data and normalized fields:

- identity: `source`, `serving_provider`, `raw_model_id`, `canonical_model_key`, aliases, route/region/variant
- limits: context, input, output, max tokens, tier-specific limits
- costs: input/output/cache-read/cache-write/reasoning/audio/image/video, normalized to USD per 1M tokens where possible
- capabilities: tool calling, structured output, JSON/response schema, reasoning, reasoning effort levels, attachments, modalities, web search, prompt caching, system messages, parallel tools
- lifecycle: release date, knowledge cutoff, last updated, status/deprecation date
- transport: endpoint mode (`chat`, `responses`, `messages`, `anthropic`, `openai-compatible`), base URL, SDK/provider package
- provenance: repository URL, file path, commit hash, commit date, commit message, extraction timestamp, parser version

Keep raw fields alongside normalized fields so every discrepancy can be traced back to the source value.

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

### Step 6: Produce reviewable reports

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

Initial model families for manual validation:

- OpenAI GPT-5.x / Codex variants
- Anthropic Claude Sonnet/Opus/Haiku 4.x
- Google Gemini 2.5/3.x
- DeepSeek V3.2/R1 variants
- Moonshot Kimi K2.x
- MiniMax M2.x
- Z.ai GLM 4.6/4.7/5.x

## Decisions

- **D1: Treat Models.dev as a source, not ground truth** — OpenCode uses it internally, but the crawler’s purpose is to detect disagreement; no source should be hard-coded as correct.
- **D2: Store raw and normalized records** — discrepancies need auditability and parser debugging.
- **D3: Keep provider route separate from underlying model** — wrappers legitimately change costs, limits, and capabilities.
- **D4: Use PyDriller for history, direct parsers for snapshots** — PyDriller is ideal for commit/event extraction; source-specific parsers are cleaner for current-state normalization.
- **D5: Start with static files and public APIs** — avoids API keys and makes the first crawler reproducible.
- **D6: Prefer explicit alias maps over fuzzy matching** — fuzzy identity matching is useful for candidates but too risky for final discrepancies.
- **D7: Save after each major step** — raw fetch, normalized records, comparisons, and reports should be persisted independently for post-mortem debugging.

## Success criteria

- [ ] Parse at least four primary sources: Models.dev repo/API, Pi, LiteLLM, and OpenRouter API.
- [ ] Normalize at least 90% of parsed chat/text-generation records into shared identity, limit, cost, and capability fields.
- [ ] Produce a discrepancy report with provenance for every compared field.
- [ ] Correctly unit-normalize LiteLLM per-token prices and Models.dev/Pi per-1M-token prices.
- [ ] Recover PyDriller change timelines for at least two repositories over a bounded period.
- [ ] Identify at least 10 actionable discrepancies or confirm low discrepancy rate with supporting CSV evidence.
- [ ] Keep false-positive rate under 20% on a manual review sample of 50 findings.

## Risks and mitigations

- Identity normalization can over-collapse unrelated route variants. Mitigation: conservative matching and alias confidence labels.
- Generated TypeScript model registries may be hard to parse robustly. Mitigation: prefer generator inputs where available; otherwise use a TS/JS parser or evaluate in a sandboxed subprocess without network/API keys.
- Public APIs may change between runs. Mitigation: persist raw API responses with timestamps.
- Repositories may be large. Mitigation: shallow clone for snapshots; PyDriller path filters and bounded date windows for history.
- Some differences are legitimate provider-specific behavior. Mitigation: model comparisons should distinguish first-party, wrapper, region, and tier variants.

## Next implementation slice

1. Add a small Python package/module for `model_meta_crawler` under `src/tk/` or a single experiment script under `experiments/2605/1001_collect_model_meta.py`.
2. Add `pydriller` plus TOML/JS parsing dependencies if missing.
3. Implement adapters for Models.dev API, LiteLLM JSON, and Pi generated registry first.
4. Emit normalized JSONL and a minimal `discrepancies.csv` for the initial model-family whitelist.
5. Add PyDriller timelines after the static adapters produce useful current-state diffs.

[^opencode]: https://github.com/anomalyco/opencode
[^modelsdev]: https://github.com/anomalyco/models.dev
[^pi]: https://github.com/earendil-works/pi
[^litellm]: https://github.com/BerriAI/litellm
[^aider]: https://github.com/Aider-AI/aider
