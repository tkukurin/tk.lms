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
    model = "sonnet"               # optional, model alias or full name (e.g. 'opus', 'claude-sonnet-4-5-20250929')
    proj_dir = "/path/to/seed"     # optional, seed project for e2e
    feat_generalize = "..."        # optional, prompt to extract CLAUDE.md

Implementation notes:
    - Uses --output-format=stream-json + --verbose for token-level progress (not turn-level)
    - Claude sessions tied to cwd; resume followups in same sandbox, split outputs later
    - Incremental saves: mv_sandbox after each step with dirs_exist_ok=True
    - Logs: stdout/stderr to tmpdir/logs/ for debugging; stderr printed on error
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm

import tk
from tk.eval import code as clib

CLAUDE_DIR = Path.home() / ".claude"
TMPDIR_PREFIX = "claudit-"



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


def run_claude(prompt: str, cwd: Path, session_id: str | None = None, logdir: Path | None = None, model: str | None = None) -> dict:
    """Run claude [cli] non-interactively and return parsed JSON output.

    settings.json (allow/deny lists) defines *which* paths are accessible; --permission-mode=acceptEdits
    controls *how* permission prompts are handled (auto-accept edits without asking), so both layers apply.

    [cli]: https://docs.anthropic.com/en/docs/claude-code/cli-reference
    """
    cmd = ["claude", "-p", prompt, "--output-format", "stream-json", "--permission-mode", "acceptEdits", "--verbose"]
    if session_id:
        cmd += ["-r", session_id]
    if model:
        cmd += ["--model", model]

    # Setup logging
    if logdir:
        logdir.mkdir(parents=True, exist_ok=True)
        stdout_log = logdir / f"claude_{session_id or 'new'}.stdout.log"
        stderr_log = logdir / f"claude_{session_id or 'new'}.stderr.log"
    else:
        stdout_log = stderr_log = None

    # Run with streaming output for progress tracking
    with tqdm(desc="Claude", unit=" events", dynamic_ncols=True, leave=True) as pbar:
        with subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        ) as proc:
            stdout_lines = []
            stderr_lines = []
            final_result = None

            # Read stdout for streaming JSON events
            if proc.stdout:
                for line in proc.stdout:
                    stdout_lines.append(line)
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        event = json.loads(line)
                        event_type = event.get("type", "")

                        # Update progress based on event type
                        if event_type == "assistant":
                            content = event.get("message", {}).get("content", [])
                            for item in content:
                                item_type = item.get("type", "")
                                if item_type == "thinking":
                                    pbar.set_postfix_str("thinking...")
                                    pbar.update(1)
                                elif item_type == "tool_use":
                                    tool = item.get("name", "tool")
                                    pbar.set_postfix_str(f"{tool}")
                                    pbar.update(1)
                                elif item_type == "text":
                                    text = item.get("text", "")[:40]
                                    if text:
                                        pbar.set_postfix_str(f"text: {text}...")
                                        pbar.update(1)
                        elif event_type == "tool_result":
                            tool = event.get("tool_name", "")
                            pbar.set_postfix_str(f"{tool} done")
                            pbar.update(1)
                        elif event_type == "result":
                            final_result = event

                    except json.JSONDecodeError:
                        pass  # Skip non-JSON lines

            # Read stderr separately
            if proc.stderr:
                stderr_text = proc.stderr.read()
                stderr_lines.append(stderr_text)

            proc.wait()

            stdout_text = "".join(stdout_lines)
            stderr_text = "".join(stderr_lines)

            # Write logs
            if stdout_log:
                stdout_log.write_text(stdout_text)
            if stderr_log:
                stderr_log.write_text(stderr_text)

            if proc.returncode != 0:
                print(f"\n[error] claude command failed with exit code {proc.returncode}")
                print(f"[error] command: {' '.join(cmd)}")
                if stderr_text:
                    print(f"[error] stderr:\n{stderr_text}")
                raise subprocess.CalledProcessError(proc.returncode, cmd, stdout_text, stderr_text)

            pbar.set_postfix_str("complete")

    # Return the final result object, or parse last JSON line if no result event
    if final_result:
        return final_result

    # Fallback: parse last valid JSON line
    for line in reversed(stdout_lines):
        line = line.strip()
        if line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    raise ValueError("No valid JSON result found in output")

def get_writes(session_id: str, cwd: Path) -> dict[str, str]:
    # TODO i am not sure this is 100% robust
    jsonl = session_jsonl(session_id)
    if not jsonl:
        return {}
    lines = jsonl.read_text().splitlines()
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
    existing = [d for d in outdir.iterdir() if d.is_dir() and d.name.startswith("v")]
    versions = [int(d.name[1:]) for d in existing if d.name[1:].isdigit()]
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

def copy_transcript(session_id: str, dest: Path) -> bool:
    """Copy transcript for session_id to dest/transcript.jsonl. Returns True if copied."""
    jsonl = session_jsonl(session_id)
    if jsonl and jsonl.exists():
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(jsonl, dest / "transcript.jsonl")
        return True
    return False

def mv_sandbox(base: Path, out: Path, session_id: str | None = None, flatten: bool = False) -> Path:
    """Move sandbox (entire tmpdir) to output directory. Returns output path."""
    if flatten:
        # flatten: copy all files to out root
        out.mkdir(parents=True, exist_ok=True)
        for f in base.rglob("*"):
            if f.is_file() and f.name not in ("settings.json",):
                shutil.copy(f, out / f.name)
    else:
        # preserve structure: copy entire tmpdir
        shutil.copytree(base, out, dirs_exist_ok=True, ignore=shutil.ignore_patterns("settings.json", ".claude"))

    # copy transcript if session_id provided (to specific subdir within out)
    if session_id:
        copy_transcript(session_id, out)

    return out


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
    model = cfg.get("model")
    if not cwd.exists():
        cwd.mkdir(parents=True)
    write_settings(cwd)

    # Setup log directory
    logdir = Path(tempfile.mkdtemp(prefix=f"{TMPDIR_PREFIX}logs-"))
    print(f"logs: {logdir}")

    result = run_claude(prompt, cwd, session_id, logdir=logdir, model=model)
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
    """Move sandbox (entire tmpdir) to output directory."""
    base = Path(args.base).resolve()
    out = Path(args.out).resolve() if args.out else tk.xpdir("out/claudit/%y%m") / base.name
    mv_sandbox(base, out, session_id=args.session_id, flatten=args.flatten)
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
    cfg = load_config(Path(args.config))
    prompt = cfg.get("prompt") or sys.exit("config must define 'prompt'")
    feat_gen = cfg.get("feat_generalize",
        "Generalize the coding preferences from this conversation into a CLAUDE.md rule.")
    proj_dir = Path(cfg["proj_dir"]) if cfg.get("proj_dir") else None
    model = cfg.get("model")

    base = Path(tempfile.mkdtemp(prefix=TMPDIR_PREFIX))
    logdir = base / "logs"
    print(f"sandbox: {base}")
    print(f"logs: {logdir}")

    outdir = tk.xpdir("out/claudit/%y%m")
    suffix = get_vnext(outdir)
    out = outdir / suffix

    print("\n[1/2] running norule...")
    norule_sandbox = setup_sandbox(base / "norule.step0", proj_dir)
    r11 = run_claude(prompt, norule_sandbox, logdir=logdir, model=model)
    sid1 = r11["session_id"]

    # Copy sandbox state (will save transcript in step1 after rename)
    out = mv_sandbox(base, out)
    print(f"saved norule (step 0) to {out}")
    print("generating CLAUDE.md...")

    r12 = run_claude(feat_gen, norule_sandbox, sid1, logdir=logdir, model=model)
    norule_sandbox = norule_sandbox.rename(norule_sandbox.with_suffix(".step1"))

    # Get all writes and split out CLAUDE.md
    w1 = get_writes(sid1, norule_sandbox)
    w1, agents = split_agents_diff(w1)

    print("\n[2/2] running yerule...")
    yerule_sandbox = setup_sandbox(base / "yerule", proj_dir, agents)
    r2 = run_claude(
        prompt, yerule_sandbox, logdir=logdir, model=model)
    sid2 = r2["session_id"]
    w2 = get_writes(sid2, yerule_sandbox)
    w2, _ = split_agents_diff(w2)
    
    # Save final state with transcripts
    # Note: norule.step0 and norule.step1 share sid1 transcript (same session, continued)
    out = mv_sandbox(base, out)
    copy_transcript(sid1, out / "norule.step0")  # Initial prompt + CLAUDE.md gen
    copy_transcript(sid1, out / "norule.step1")  # Same session, full transcript
    copy_transcript(sid2, out / "yerule")        # Separate session with rules
    
    (out / "meta.json").write_text(json.dumps(meta_diff(w1, w2), indent=2))
    print(f"saved {suffix} to {out}")

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
