"""claudit - decoupled experiment runner CLI.

Usage:
    claudit claude $cfg_file.py   # run claude with config (prompt, cwd, session_id)
    claudit list [base_dir]       # list claudit-* dirs in $TMPDIR (or base_dir)
    claudit mv $base [-o $dir]    # move sandbox to datadir/out/claudit/yymm/
    claudit diff $base            # diff norule/yerule directories
    claudit e2e $cfg_file.py      # full pipeline: norule -> CLAUDE.md -> yerule -> diff

Config file format (python):
    prompt = "your prompt here"
    cwd = "/path/to/sandbox"       # optional, defaults to cwd
    session_id = "abc123"          # optional, for continuing sessions
    proj_dir = "/path/to/seed"     # optional, seed project for e2e
    feat_generalize = "..."        # optional, prompt to extract CLAUDE.md
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from tk import datadir
from tk.eval import code as clib

CLAUDE_DIR = Path.home() / ".claude"
TMPDIR_PREFIX = "claudit-"

def default_outdir() -> Path:
    """Default output: datadir/out/claudit/yymm/"""
    from datetime import datetime
    yymm = datetime.now().strftime("%y%m")
    return datadir / "out" / "claudit" / yymm

def session_jsonl(sid: str) -> Path | None:
    return next((f for f in (CLAUDE_DIR / "projects").glob(f"*/{sid}.jsonl")), None)

SETTINGS_TEMPLATE = {
    "$schema": "https://json.schemastore.org/claude-code-settings.json",
    "permissions": {
        "allow": ["Write(**)", "Edit(**)", "Read(**)", "Bash(python*)", "Bash(python3*)", "Bash(ls:*)", "Bash(cat:*)"],
        "deny": [],
    },
}

def write_settings(d: Path):
    settings = {
        **SETTINGS_TEMPLATE,
        "permissions": {
            **SETTINGS_TEMPLATE["permissions"],
            "deny": [f"Write({Path.home()}/**)", f"Edit({Path.home()}/**)"],
            "allow": [f"Write({d}/**)", f"Edit({d}/**)", "Read(**)", "Bash(python*)", "Bash(python3*)", "Bash(ls:*)", "Bash(cat:*)"]
        }
    }
    (d / ".claude").mkdir(exist_ok=True, parents=True)
    (d / ".claude" / "settings.json").write_text(json.dumps(settings, indent=2))


def load_config(cfg_path: Path) -> dict:
    """Load a python config file and return its variables as dict."""
    ns = {}
    exec(cfg_path.read_text(), ns)
    return {k: v for k, v in ns.items() if not k.startswith("_")}


def run_claude(prompt: str, cwd: Path, session_id: str | None = None) -> dict:
    """Run claude [cli] non-interactively and return parsed JSON output.

    settings.json (allow/deny lists) defines *which* paths are accessible; --permission-mode=acceptEdits
    controls *how* permission prompts are handled (auto-accept edits without asking), so both layers apply.

    [cli]: https://docs.anthropic.com/en/docs/claude-code/cli-reference
    """
    cmd = ["claude", "-p", prompt, "--output-format", "json", "--permission-mode", "acceptEdits"]
    if session_id:
        cmd += ["-r", session_id]
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True, timeout=120)
    return json.loads(r.stdout)

def check_writes_contained(diffs: dict[str, str], cwd: Path) -> tuple[dict[str, str], list[str]]:
    """Post-hoc audit: returns (diffs, escaped_paths). settings.json is the actual guard."""
    resolved_cwd = cwd.resolve()
    outside = [p for p in diffs if not (cwd / p).resolve().is_relative_to(resolved_cwd)]
    return diffs, outside

def get_writes(session_id: str, cwd: Path) -> dict[str, str]:
    # TODO i am not sure this is 100% robust
    lines = session_jsonl(session_id).read_text().splitlines()
    writes = {}
    for line in lines:
        d = json.loads(line)
        if d.get("type") != "assistant": continue
        for c in d.get("message", {}).get("content", []):
            if c.get("type") == "tool_use" and c.get("name") == "Write":
                writes[c["input"]["file_path"]] = c["input"]["content"]
    return writes

def split_agents_diff(diffs: dict[str, str]) -> tuple[dict[str, str], str | None]:
    agent_keys = {k for k in diffs if Path(k).name in ("CLAUDE.md", "AGENTS.md")}
    content = "\n\n".join(diffs[k] for k in agent_keys) if agent_keys else None
    return {k: v for k, v in diffs.items() if k not in agent_keys}, content

def get_vnext(outdir: Path) -> str:
    if not outdir.exists(): return "v0"
    existing = [d.name for d in outdir.iterdir() if d.is_dir() and ".v" in d.name]
    versions = [int(n.split(".v")[1]) for n in existing if n.split(".v")[1].isdigit()]
    return f"v{max(versions, default=-1) + 1}"


def setup_sandbox(d: Path, proj_dir: Path | None = None, claude_md: str | None = None) -> Path:
    """Create sandbox dir, optionally seed from proj_dir, optionally write CLAUDE.md."""
    d.mkdir(parents=True, exist_ok=True)
    if proj_dir:
        shutil.copytree(proj_dir, d, dirs_exist_ok=True)
    (d / "CLAUDE.md").unlink(missing_ok=True)
    if claude_md:
        (d / "CLAUDE.md").write_text(claude_md)
    write_settings(d)
    return d


def run_and_collect(prompt: str, cwd: Path, session_id: str | None = None,
                    followup: str | None = None) -> tuple[str, dict[str, str], str | None]:
    """Run claude, optionally followup, return (session_id, writes, agents_content)."""
    r = run_claude(prompt, cwd, session_id)
    sid = r["session_id"]
    if followup:
        run_claude(followup, cwd, sid)
    writes, esc = check_writes_contained(get_writes(sid, cwd), cwd)
    if esc: print(f"[warn] writes escaped sandbox: {esc}")
    writes, agents = split_agents_diff(writes)
    return sid, writes, agents


def save_run(outdir: Path, name: str, session_id: str, writes: dict[str, str]) -> Path:
    """Save session transcript + writes to outdir/name/."""
    d = outdir / name
    d.mkdir(parents=True, exist_ok=True)
    jsonl = session_jsonl(session_id)
    if jsonl: shutil.copy(jsonl, d / "transcript.jsonl")
    for path, content in writes.items():
        (d / Path(path).name).write_text(content)
    return d


def meta_diff(d1: dict[str, str], d2: dict[str, str]) -> dict:
    result = {}
    all_files = d1.keys() | d2.keys()
    for filepath in all_files:
        fname = Path(filepath).name
        c1, c2 = d1.get(filepath), d2.get(filepath)
        if fname.endswith(".py") and c1 and c2:
            funcs1, funcs2 = clib.text2funcmap(c1), clib.text2funcmap(c2)
            fn_diff = clib.funcdiff(funcs1, funcs2)
            result[fname] = {
                "stats": {"norule_fn_count": len(funcs1), "yerule_fn_count": len(funcs2)},
                "functions": fn_diff,
            }
        elif c1 != c2:
            result[fname] = {"norule": c1, "yerule": c2}
    return result


# ─── CLI commands ───────────────────────────────────────────────────────────

def cmd_claude(args):
    """Run claude with config file. Outputs JSON with session_id."""
    cfg = load_config(Path(args.config))
    prompt = cfg.get("prompt")
    if not prompt:
        sys.exit("config must define 'prompt'")
    cwd = Path(cfg.get("cwd", ".")).resolve()
    session_id = cfg.get("session_id")
    if not cwd.exists():
        cwd.mkdir(parents=True)
    write_settings(cwd)
    result = run_claude(prompt, cwd, session_id)
    print(json.dumps(result, indent=2))


def cmd_list(args):
    """Find and list all claudit-* directories, ordered by mtime."""
    import tempfile
    from datetime import datetime
    base = Path(args.base_dir) if args.base_dir else Path(tempfile.gettempdir())
    dirs = [d for d in base.glob(f"{TMPDIR_PREFIX}*") if d.is_dir()]
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    for d in dirs[:args.limit]:
        mtime = datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"{mtime}  {d}")


def cmd_mv(args):
    """Move sandbox files + transcript to output directory."""
    base = Path(args.base).resolve()
    out = Path(args.out).resolve() if args.out else default_outdir() / base.name
    out.mkdir(parents=True, exist_ok=True)

    # copy transcript if session_id provided
    if args.session_id:
        jsonl = session_jsonl(args.session_id)
        if jsonl:
            shutil.copy(jsonl, out / "transcript.jsonl")

    # copy all files from base (flatten or preserve structure)
    for f in base.rglob("*"):
        if f.is_file() and f.name not in ("settings.json",):
            dest = out / f.name if args.flatten else out / f.relative_to(base)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, dest)
    print(f"moved {base} -> {out}")


def cmd_diff(args):
    """Diff norule/yerule directories under base."""
    base = Path(args.base).resolve()
    norule_dir = base / "norule"
    yerule_dir = base / "yerule"
    if not norule_dir.exists() or not yerule_dir.exists():
        sys.exit(f"expected {base}/norule and {base}/yerule")

    d1 = {str(f.relative_to(norule_dir)): f.read_text() for f in norule_dir.rglob("*") if f.is_file()}
    d2 = {str(f.relative_to(yerule_dir)): f.read_text() for f in yerule_dir.rglob("*") if f.is_file()}

    result = meta_diff(d1, d2)
    print(json.dumps(result, indent=2))


def cmd_e2e(args):
    """End-to-end: run norule -> extract CLAUDE.md -> run yerule -> diff."""
    import tempfile
    cfg = load_config(Path(args.config))
    prompt = cfg.get("prompt") or sys.exit("config must define 'prompt'")
    feat_gen = cfg.get("feat_generalize",
        "Generalize the coding preferences from this conversation into a CLAUDE.md rule.")
    proj_dir = Path(cfg["proj_dir"]) if cfg.get("proj_dir") else None

    base = Path(tempfile.mkdtemp(prefix=TMPDIR_PREFIX))
    print(f"sandbox: {base}")

    print("running norule...")
    sid1, w1, agents = run_and_collect(prompt, setup_sandbox(base / "norule", proj_dir), followup=feat_gen)
    print("running yerule...")
    sid2, w2, _ = run_and_collect(prompt, setup_sandbox(base / "yerule", proj_dir, agents))

    outdir = default_outdir()
    suffix = get_vnext(outdir)
    save_run(outdir, f"norule.{suffix}", sid1, w1)
    save_run(outdir, f"yerule.{suffix}", sid2, w2).joinpath("CLAUDE.md").write_text(agents or "")
    (outdir / f"meta.{suffix}.json").write_text(json.dumps(meta_diff(w1, w2), indent=2))
    print(f"saved {suffix} to {outdir}")


def main():
    parser = argparse.ArgumentParser(prog="claudit", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest="cmd", required=True)

    p_claude = subs.add_parser("claude", help="run claude with config file")
    p_claude.add_argument("config", help="path to config.py")
    p_claude.set_defaults(func=cmd_claude)

    p_list = subs.add_parser("list", help="list claudit-* dirs by mtime")
    p_list.add_argument("base_dir", nargs="?", help="base directory (default: $TMPDIR)")
    p_list.add_argument("-n", "--limit", type=int, default=30, help="max dirs to show (default: 30)")
    p_list.set_defaults(func=cmd_list)

    p_mv = subs.add_parser("mv", help="move sandbox to output")
    p_mv.add_argument("base", help="source base directory")
    p_mv.add_argument("--out", "-o", help=f"output directory (default: datadir/out/claudit/yymm/)")
    p_mv.add_argument("--session-id", "-s", help="session id for transcript")
    p_mv.add_argument("--flatten", "-f", action="store_true", help="flatten file structure")
    p_mv.set_defaults(func=cmd_mv)

    p_diff = subs.add_parser("diff", help="diff norule/yerule dirs")
    p_diff.add_argument("base", help="base dir containing norule/ and yerule/")
    p_diff.set_defaults(func=cmd_diff)

    p_e2e = subs.add_parser("e2e", help="end-to-end: norule -> extract CLAUDE.md -> yerule -> diff")
    p_e2e.add_argument("config", help="path to config.py (must define 'prompt')")
    p_e2e.set_defaults(func=cmd_e2e)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
