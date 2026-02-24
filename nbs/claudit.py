"""claudit - single-step experiment runner.

test whether a CLAUDE.md coding-style rule improves Claude's output.

Setup:
    1. edit initial_prompt (the task)
    2. edit feat_generalize (extraction prompt)
    (3. set proj_dir to seed both sandboxes from an existing project)

how it works:
    run#1 (2 prompts to CC):
        1. Run initial_prompt in a temp sandbox (no CLAUDE.md)
        2. Run feat_generalize to extract coding preferences → CLAUDE.md
    run#2 (1 prompt to CC):
        3. Inject the extracted CLAUDE.md into a second temp sandbox
        4. Run the same initial_prompt again

    Both transcripts and written files are saved under data/norule.vN/ and
    data/yerule.vN/. A meta.vN.json records per-file diffs between the runs.

    then you can diff 1/2.
"""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from tk.eval import code as clib

CLAUDE_DIR = Path.home() / ".claude"

# if you want to start / copy existing project
proj_dir: Path | None = None

initial_prompt = (
    "create a new python file using best coding practices. add type annotations. "
    "I want a generic container class GptResult[T] which contains either int or str. "
    "We might extend in the future. create a demo with nice rendering of each result. "
    "use only python stdlib."
)
feat_generalize = "Generalize the coding preferences from this conversation into a CLAUDE.md rule."

session_jsonl = lambda sid: next(
    (f for f in (CLAUDE_DIR / "projects").glob(f"*/{sid}.jsonl")), None)

SETTINGS_TEMPLATE = {
    "$schema": "https://json.schemastore.org/claude-code-settings.json",
    # permissions reference: https://docs.anthropic.com/en/docs/claude-code/settings#permissions
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
    (d / ".claude" / "settings.json").write_text(json.dumps(
        settings, indent=2))


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
    existing = [d.name for d in outdir.iterdir() if d.is_dir() and ".v" in d.name]
    versions = [int(n.split(".v")[1]) for n in existing if n.split(".v")[1].isdigit()]
    return f"v{max(versions, default=-1) + 1}"

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
                "stats": {
                    "norule_fn_count": len(funcs1),
                    "yerule_fn_count": len(funcs2)
                },
                "functions": fn_diff,
            }
        elif c1 != c2:
            result[fname] = {"norule": c1, "yerule": c2}
    return result


if __name__ == "__main__":
    outdir = Path(__file__).parent
    suffix = get_vnext(outdir)

    print(f"sandbox on: {(base := Path(tempfile.mkdtemp(prefix='claudit-')))}")
    (norule_dir := base / "norule").mkdir()
    (yerule_dir := base / "yerule").mkdir()
    if proj_dir:  # norule: no CLAUDE.md
        shutil.copytree(proj_dir, norule_dir)
        (norule_dir / "CLAUDE.md").unlink(missing_ok=True)
    write_settings(norule_dir)
    r1 = run_claude(initial_prompt, norule_dir)
    sid1 = r1["session_id"]
    r2 = run_claude(feat_generalize, norule_dir, sid1)
    diffs1, esc1 = check_writes_contained(
        get_writes(sid1, norule_dir), norule_dir)
    if esc1: print(f"[warn] norule escaped: {esc1}")
    diffs1, agents_content = split_agents_diff(diffs1)

    if proj_dir:  # yerule: inject extracted CLAUDE.md
        shutil.copytree(proj_dir, yerule_dir)
        (yerule_dir / "CLAUDE.md").unlink(missing_ok=True)
    if agents_content: (yerule_dir / "CLAUDE.md").write_text(agents_content)
    write_settings(yerule_dir)
    r2 = run_claude(initial_prompt, yerule_dir)
    sid2 = r2["session_id"]
    diffs2, esc2 = check_writes_contained(
        get_writes(sid2, yerule_dir), yerule_dir)
    if esc2: print(f"[warn] yerule escaped: {esc2}")
    diffs2, _ = split_agents_diff(diffs2)

    s1 = session_jsonl(sid1)
    s2 = session_jsonl(sid2)
    for name, diffs, jsonl in [(f"norule.{suffix}", diffs1, s1), (f"yerule.{suffix}", diffs2, s2)]:
        d = outdir / name
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(jsonl, d / "transcript.jsonl")
        for path, content in diffs.items():
            (d / Path(path).name).write_text(content)

    yerule_claude = outdir / f"yerule.{suffix}" / "CLAUDE.md"
    yerule_claude.write_text(agents_content or "")

    meta_diffs = meta_diff(diffs1, diffs2)
    (outdir / f"meta.{suffix}.json").write_text(
        json.dumps(meta_diffs, indent=2))
    print(f"Saved experiment {suffix} to {outdir}")
