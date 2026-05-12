# %%
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pydriller",
#     "tqdm",
# ]
# ///
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from tqdm import tqdm
from typing import Any

REPO_URL = "https://github.com/BerriAI/litellm.git"
RAW_BASE = "https://raw.githubusercontent.com/BerriAI/litellm/main"
PRICES_PATH = "model_prices_and_context_window.json"
ENDPOINTS_PATH = "provider_endpoints_support.json"
DOCS_PREFIX = "docs/my-website/docs/providers/"
PROVIDER_GLOBS = (DOCS_PREFIX, PRICES_PATH, ENDPOINTS_PATH)
UA = "tk-litellm-provider-fetcher/0.1"
TMPDIR_PREFIX = "litellm_providers_"
DATADIR = Path(__file__).resolve().parents[2] / "data"
TOOL = "litellm-providers"


# %% Data types
@dataclass
class ProviderEntry:
  provider: str
  models: list[str] = field(default_factory=list)
  supports: dict[str, bool] = field(default_factory=dict)
  sample_pricing: dict[str, float] = field(default_factory=dict)


@dataclass
class CommitDiff:
  hash: str
  date: datetime
  author: str
  msg: str
  files: list[str]
  insertions: int
  deletions: int


# %% Helpers
def get_default_outdir(datadir: Path = DATADIR, tool: str = TOOL) -> Path:
  return datadir / "out" / tool / datetime.now().strftime("%y%m")


def get_vnext(outdir: Path) -> str:
  if not outdir.exists(): return "v0"
  vs = [int(s) for d in outdir.iterdir() if d.is_dir()
        and ".v" in d.name and (s := d.name.rsplit(".v", 1)[1]).isdigit()]
  return f"v{max(vs, default=-1) + 1}"


def setup_outdir(outdir: Path | None) -> Path:
  base = outdir or get_default_outdir()
  d = base / f"litellm.{get_vnext(base)}"
  d.mkdir(parents=True, exist_ok=True)
  return d


def fetch_raw(path: str) -> bytes:
  url = f"{RAW_BASE}/{path}"
  req = urllib.request.Request(url, headers={"User-Agent": UA})
  with urllib.request.urlopen(req, timeout=60) as r:
    return r.read()


def fetch_json_raw(path: str) -> Any:
  return json.loads(fetch_raw(path))


# %% Provider extraction from model_prices_and_context_window.json
def get_provider_from_key(model_key: str) -> str:
  if "/" in model_key:
    return model_key.split("/", 1)[0]
  return "openai"


def get_providers_from_prices(data: dict[str, Any]) -> dict[str, ProviderEntry]:
  providers: dict[str, ProviderEntry] = {}
  for key, info in data.items():
    if not isinstance(info, dict): continue
    prov = info.get("litellm_provider", get_provider_from_key(key))
    if prov not in providers:
      providers[prov] = ProviderEntry(provider=prov)
    entry = providers[prov]
    entry.models.append(key)
    if (inp := info.get("input_cost_per_token")) and not entry.sample_pricing:
      entry.sample_pricing = {
        "input_per_1m": inp * 1_000_000,
        "output_per_1m": info.get("output_cost_per_token", 0) * 1_000_000,
      }
    for cap in ("supports_function_calling", "supports_vision",
                "supports_response_schema", "supports_tool_choice"):
      if info.get(cap): entry.supports[cap] = True
  return providers


# %% Provider extraction from provider_endpoints_support.json
def get_endpoint_support(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
  providers = data.get("providers", {})
  result: dict[str, dict[str, Any]] = {}
  for name, info in providers.items():
    if not isinstance(info, dict): continue
    eps = info.get("endpoints", {})
    result[name] = {
      "display_name": info.get("display_name", name),
      "url": info.get("url", ""),
      **{k: v for k, v in eps.items() if isinstance(v, bool)},
    }
  return result


# %% PyDriller: recent diffs on provider-related files
def shallow_clone(dest: Path, since: datetime) -> Path:
  since_str = since.strftime("%Y-%m-%d")
  cmd = [
    "git", "clone", "--single-branch", "--branch", "main",
    f"--shallow-since={since_str}", "--no-tags",
    "--filter=blob:none", REPO_URL, str(dest),
  ]
  print(f"  shallow clone since {since_str}...")
  r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
  if r.returncode != 0:
    print(f"  clone stderr: {r.stderr[:300]}", file=sys.stderr)
    raise RuntimeError(f"git clone failed: exit {r.returncode}")
  return dest


def get_recent_diffs(days: int = 14, max_commits: int = 200) -> list[CommitDiff]:
  from pydriller import Repository
  since = datetime.now(timezone.utc) - timedelta(days=days)
  diffs: list[CommitDiff] = []
  with TemporaryDirectory(prefix=TMPDIR_PREFIX) as tmp:
    repo_path = shallow_clone(Path(tmp) / "litellm", since)
    commits_seen = 0
    repo = Repository(
      str(repo_path),
      since=since,
      only_modifications_with_file_types=[".json", ".md"],
    )
    for commit in tqdm(repo.traverse_commits(), desc="mining commits",
                       total=max_commits):
      if commits_seen >= max_commits: break
      commits_seen += 1
      touched = [m.new_path or m.old_path for m in commit.modified_files
                 if m.new_path or m.old_path]
      provider_files = [f for f in touched if any(
        f and (f.startswith(g) or f == g) for g in PROVIDER_GLOBS
      )]
      if not provider_files: continue
      ins = sum(m.added_lines for m in commit.modified_files
                if (m.new_path or m.old_path) in provider_files)
      dels = sum(m.deleted_lines for m in commit.modified_files
                 if (m.new_path or m.old_path) in provider_files)
      diffs.append(CommitDiff(
        hash=commit.hash[:10],
        date=commit.committer_date,
        author=commit.author.name,
        msg=commit.msg.split("\n", 1)[0][:120],
        files=provider_files,
        insertions=ins,
        deletions=dels,
      ))
  return diffs


# %% Output
def write_providers_json(providers: dict[str, ProviderEntry], outdir: Path) -> Path:
  out = outdir / "providers.json"
  rows = []
  for p in sorted(providers.values(), key=lambda x: -len(x.models)):
    rows.append({
      "provider": p.provider,
      "model_count": len(p.models),
      "supports": p.supports,
      "sample_pricing": p.sample_pricing,
      "models_sample": p.models[:10],
    })
  out.write_text(json.dumps(rows, indent=2))
  print(f"  wrote {len(rows)} providers -> {out}")
  return out


def write_diffs_json(diffs: list[CommitDiff], outdir: Path) -> Path:
  out = outdir / "recent_diffs.json"
  rows = [{
    "hash": d.hash, "date": d.date.isoformat(), "author": d.author,
    "msg": d.msg, "files": d.files, "ins": d.insertions, "del": d.deletions,
  } for d in diffs]
  out.write_text(json.dumps(rows, indent=2))
  print(f"  wrote {len(rows)} diffs -> {out}")
  return out


def write_endpoints_json(endpoints: dict[str, dict[str, Any]],
                         outdir: Path) -> Path:
  out = outdir / "endpoint_support.json"
  out.write_text(json.dumps(endpoints, indent=2))
  print(f"  wrote {len(endpoints)} endpoint entries -> {out}")
  return out


def write_summary_md(providers: dict[str, ProviderEntry],
                     diffs: list[CommitDiff], outdir: Path) -> Path:
  out = outdir / "summary.md"
  lines = ["# LiteLLM Provider Data Summary", ""]
  lines.append(f"**Generated**: {datetime.now(timezone.utc).isoformat()}")
  lines.append(f"**Source**: {REPO_URL}")
  lines.append(f"**Providers**: {len(providers)}")
  total_models = sum(len(p.models) for p in providers.values())
  lines.append(f"**Total model entries**: {total_models}")
  lines.append("")
  lines.append("## Top providers by model count")
  lines.append("| Provider | Models | Capabilities |")
  lines.append("|----------|--------|-------------|")
  top = sorted(providers.values(), key=lambda x: -len(x.models))[:25]
  for p in top:
    caps = ", ".join(k.replace("supports_", "") for k in sorted(p.supports))
    lines.append(f"| {p.provider} | {len(p.models)} | {caps} |")
  if diffs:
    lines += ["", "## Recent diffs (provider-related)", ""]
    lines.append("| Date | Author | Message | Files | +/- |")
    lines.append("|------|--------|---------|-------|-----|")
    for d in diffs[:30]:
      fs = ", ".join(Path(f).name for f in d.files[:3])
      if len(d.files) > 3: fs += f" +{len(d.files)-3}"
      lines.append(
        f"| {d.date:%Y-%m-%d} | {d.author[:20]} | {d.msg[:60]} "
        f"| {fs} | +{d.insertions}/-{d.deletions} |"
      )
  else:
    lines += ["", "## Recent diffs", "", "_No provider-related diffs found._"]
  out.write_text("\n".join(lines))
  print(f"  wrote summary -> {out}")
  return out


# %% Commands
def cmd_fetch(outdir: Path | None = None) -> None:
  d = setup_outdir(outdir)
  print(f"output: {d}")
  print("fetching model_prices_and_context_window.json...")
  prices = fetch_json_raw(PRICES_PATH)
  providers = get_providers_from_prices(prices)
  print(f"  parsed {len(providers)} providers")
  print("fetching provider_endpoints_support.json...")
  try:
    endpoints_raw = fetch_json_raw(ENDPOINTS_PATH)
    endpoints = get_endpoint_support(endpoints_raw)
  except Exception as e:
    print(f"  endpoint fetch failed: {e}", file=sys.stderr)
    endpoints = {}
  write_providers_json(providers, d)
  if endpoints: write_endpoints_json(endpoints, d)
  (d / "raw").mkdir(exist_ok=True)
  (d / "raw" / PRICES_PATH).write_text(json.dumps(prices, indent=2))
  print(f"  saved raw {PRICES_PATH} ({len(prices)} entries)")
  write_summary_md(providers, [], d)


def cmd_diffs(days: int = 14, max_commits: int = 200,
              outdir: Path | None = None) -> None:
  d = setup_outdir(outdir)
  print(f"output: {d}")
  print(f"mining litellm commits (last {days} days, max {max_commits})...")
  diffs = get_recent_diffs(days=days, max_commits=max_commits)
  print(f"  found {len(diffs)} provider-related commits")
  write_diffs_json(diffs, d)
  if diffs:
    print("\n  recent provider changes:")
    for diff in diffs[:10]:
      print(f"    {diff.date:%m-%d} {diff.hash} "
            f"+{diff.insertions}/-{diff.deletions} {diff.msg[:70]}")


def cmd_all(days: int = 14, max_commits: int = 200,
            outdir: Path | None = None) -> None:
  d = setup_outdir(outdir)
  print(f"output: {d}")
  print("fetching model_prices_and_context_window.json...")
  prices = fetch_json_raw(PRICES_PATH)
  providers = get_providers_from_prices(prices)
  print(f"  parsed {len(providers)} providers")
  print("fetching provider_endpoints_support.json...")
  try:
    endpoints_raw = fetch_json_raw(ENDPOINTS_PATH)
    endpoints = get_endpoint_support(endpoints_raw)
  except Exception as e:
    print(f"  endpoint fetch failed: {e}", file=sys.stderr)
    endpoints = {}
  print(f"mining litellm commits (last {days} days, max {max_commits})...")
  diffs = get_recent_diffs(days=days, max_commits=max_commits)
  print(f"  found {len(diffs)} provider-related commits")
  write_providers_json(providers, d)
  if endpoints: write_endpoints_json(endpoints, d)
  write_diffs_json(diffs, d)
  (d / "raw").mkdir(exist_ok=True)
  (d / "raw" / PRICES_PATH).write_text(json.dumps(prices, indent=2))
  write_summary_md(providers, diffs, d)


# %% CLI
def build_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
    description="Fetch LiteLLM provider docs & mine recent diffs")
  sub = p.add_subparsers(dest="cmd")
  fetch = sub.add_parser("fetch", help="fetch provider data from GitHub")
  fetch.add_argument("--outdir", type=Path, default=None)
  diffs = sub.add_parser("diffs", help="mine recent provider-related commits")
  diffs.add_argument("--days", type=int, default=14)
  diffs.add_argument("--max-commits", type=int, default=200)
  diffs.add_argument("--outdir", type=Path, default=None)
  all_cmd = sub.add_parser("all", help="fetch + diffs")
  all_cmd.add_argument("--days", type=int, default=14)
  all_cmd.add_argument("--max-commits", type=int, default=200)
  all_cmd.add_argument("--outdir", type=Path, default=None)
  return p


def main() -> None:
  args = build_parser().parse_args()
  match args.cmd:
    case "fetch": cmd_fetch(outdir=args.outdir)
    case "diffs": cmd_diffs(days=args.days, max_commits=args.max_commits,
                            outdir=args.outdir)
    case "all": cmd_all(days=args.days, max_commits=args.max_commits,
                        outdir=args.outdir)
    case _: build_parser().print_help()


if __name__ == "__main__":
  main()
