"""
# claudit.viz - visualization and filtering for claudit experiment data

Load and analyze versioned claudit experiment outputs:
- Load transcript.jsonl and meta.*.json for any vXXX version
- Filter conversations, code changes, tool uses
- Visualize with pandas + rich tables
"""

# %%
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from rich.panel import Panel

from tk import datadir

# %% [markdown]
"""## Constants (reuse from claudit.py)"""

# %%
CLAUDE_DIR = Path.home() / ".claude"
TMPDIR_PREFIX = "claudit-"

def default_outdir() -> Path:
    """Default output: datadir/out/claudit/yymm/"""
    from datetime import datetime
    yymm = datetime.now().strftime("%y%m")
    return datadir / "out" / "claudit" / yymm

# %% [markdown]
"""## Data Loading"""

# %%
@dataclass
class RunData:
    """Container for a single run's data (norule or yerule)."""
    version: str
    rule_type: str  # "norule" or "yerule"
    transcript: list[dict] | None
    files: dict[str, str]  # filename -> content
    path: Path

def load_run(outdir: Path, version: str, rule_type: str) -> RunData:
    """Load transcript + files for a specific version and rule type."""
    run_dir = outdir / f"{rule_type}.{version}"
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # load transcript
    transcript_path = run_dir / "transcript.jsonl"
    transcript = None
    if transcript_path.exists():
        transcript = [json.loads(line) for line in transcript_path.read_text().splitlines()]

    # load all files
    files = {}
    for f in run_dir.rglob("*"):
        if f.is_file() and f.name != "transcript.jsonl":
            files[f.name] = f.read_text()

    return RunData(version, rule_type, transcript, files, run_dir)

def load_meta(outdir: Path, version: str) -> dict:
    """Load meta.vXXX.json diff data."""
    meta_path = outdir / f"meta.{version}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    return json.loads(meta_path.read_text())

def list_versions(outdir: Path | None = None) -> list[str]:
    """List all available versions in output directory."""
    if outdir is None:
        outdir = default_outdir()
    if not outdir.exists():
        return []

    versions = set()
    for d in outdir.iterdir():
        if d.is_dir() and ".v" in d.name:
            version = d.name.split(".")[1]
            if version.startswith("v"):
                versions.add(version)
    return sorted(versions, key=lambda v: int(v[1:]))

# %% [markdown]
"""## Filtering"""

# %%
class Filter:
    """Filter operations for conversation, code, and tool data."""

    @staticmethod
    def convo(transcript: list[dict], **kwargs) -> list[dict]:
        """Filter conversation messages.

        Args:
            role: filter by role ("user" or "assistant")
            contains: filter messages containing text
            message_type: filter by message type in content
        """
        result = transcript

        if role := kwargs.get("role"):
            result = [m for m in result if m.get("type") == role]

        if contains := kwargs.get("contains"):
            def has_text(msg):
                if isinstance(msg.get("message"), dict):
                    content = msg["message"].get("content", [])
                    if isinstance(content, list):
                        return any(contains.lower() in str(c).lower() for c in content)
                return contains.lower() in str(msg).lower()
            result = [m for m in result if has_text(m)]

        if msg_type := kwargs.get("message_type"):
            def has_type(msg):
                if isinstance(msg.get("message"), dict):
                    content = msg["message"].get("content", [])
                    if isinstance(content, list):
                        return any(c.get("type") == msg_type for c in content if isinstance(c, dict))
                return False
            result = [m for m in result if has_type(m)]

        return result

    @staticmethod
    def code(transcript: list[dict], **kwargs) -> list[dict]:
        """Filter code-related tool uses (Write, Edit).

        Args:
            tool_name: filter by tool name ("Write", "Edit", etc.)
            file_pattern: filter by file path pattern
        """
        assistant_msgs = [m for m in transcript if m.get("type") == "assistant"]
        code_uses = []

        for msg in assistant_msgs:
            content = msg.get("message", {}).get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if item.get("type") == "tool_use":
                    tool_name = item.get("name")
                    if tool_name in ("Write", "Edit"):
                        code_uses.append({"message": msg, "tool_use": item})

        result = code_uses

        if tool_filter := kwargs.get("tool_name"):
            result = [c for c in result if c["tool_use"].get("name") == tool_filter]

        if pattern := kwargs.get("file_pattern"):
            def matches_pattern(c):
                file_path = c["tool_use"].get("input", {}).get("file_path", "")
                return pattern in file_path
            result = [c for c in result if matches_pattern(c)]

        return result

    @staticmethod
    def tools(transcript: list[dict], **kwargs) -> list[dict]:
        """Filter all tool uses.

        Args:
            tool_name: filter by specific tool name
            exclude_code: if True, exclude Write/Edit tools
        """
        assistant_msgs = [m for m in transcript if m.get("type") == "assistant"]
        tool_uses = []

        for msg in assistant_msgs:
            content = msg.get("message", {}).get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if item.get("type") == "tool_use":
                    tool_uses.append({"message": msg, "tool_use": item})

        result = tool_uses

        if tool_filter := kwargs.get("tool_name"):
            result = [t for t in result if t["tool_use"].get("name") == tool_filter]

        if kwargs.get("exclude_code"):
            result = [t for t in result if t["tool_use"].get("name") not in ("Write", "Edit")]

        return result

# %% [markdown]
"""## Visualization"""

# %%
class Viz:
    """Visualization utilities using pandas + rich."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()

    def convo(self, data: list[dict] | RunData, limit: int | None = None):
        """Visualize conversation flow."""
        if isinstance(data, RunData):
            if data.transcript is None:
                self.console.print("[red]No transcript available[/red]")
                return
            messages = data.transcript
        else:
            messages = data

        if limit:
            messages = messages[:limit]

        table = Table(title="Conversation Flow", show_header=True)
        table.add_column("#", style="dim", width=4)
        table.add_column("Role", width=10)
        table.add_column("Type", width=12)
        table.add_column("Preview", style="dim")

        for i, msg in enumerate(messages, 1):
            role = msg.get("type", "unknown")

            # extract preview
            preview = ""
            msg_content = msg.get("message", {})
            if isinstance(msg_content, dict):
                content = msg_content.get("content", [])
                if isinstance(content, list) and content:
                    first = content[0]
                    if isinstance(first, dict):
                        msg_type = first.get("type", "")
                        if msg_type == "text":
                            preview = first.get("text", "")[:80]
                        elif msg_type == "tool_use":
                            preview = f"[{first.get('name')}] {str(first.get('input', {}))[:60]}"

            table.add_row(str(i), role, msg_type if 'msg_type' in locals() else "", preview)

        self.console.print(table)

    def code_changes(self, data: list[dict] | RunData):
        """Visualize code changes (Write/Edit operations)."""
        if isinstance(data, RunData):
            if data.transcript is None:
                self.console.print("[red]No transcript available[/red]")
                return
            code_uses = Filter.code(data.transcript)
        else:
            code_uses = data

        for i, item in enumerate(code_uses, 1):
            tool = item["tool_use"]
            tool_name = tool.get("name")
            input_data = tool.get("input", {})
            file_path = input_data.get("file_path", "unknown")

            self.console.print(f"\n[bold cyan]{i}. {tool_name}: {file_path}[/bold cyan]")

            if content := input_data.get("content"):
                # try to detect language from extension
                lang = "python" if file_path.endswith(".py") else "text"
                syntax = Syntax(content[:500], lang, theme="monokai", line_numbers=True)
                self.console.print(Panel(syntax, title=f"{tool_name} content (truncated)"))

    def tool_stats(self, data: list[dict] | RunData):
        """Show tool usage statistics."""
        if isinstance(data, RunData):
            if data.transcript is None:
                self.console.print("[red]No transcript available[/red]")
                return
            tool_uses = Filter.tools(data.transcript)
        else:
            tool_uses = data

        # count by tool name
        tool_counts = {}
        for item in tool_uses:
            name = item["tool_use"].get("name", "unknown")
            tool_counts[name] = tool_counts.get(name, 0) + 1

        df = pd.DataFrame(list(tool_counts.items()), columns=["Tool", "Count"])
        df = df.sort_values("Count", ascending=False)

        table = Table(title="Tool Usage Statistics", show_header=True)
        table.add_column("Tool", style="cyan")
        table.add_column("Count", justify="right", style="green")

        for _, row in df.iterrows():
            table.add_row(row["Tool"], str(row["Count"]))

        self.console.print(table)

    def meta_diff(self, meta: dict):
        """Visualize meta diff between norule and yerule."""
        table = Table(title="File Differences", show_header=True)
        table.add_column("File", style="cyan")
        table.add_column("Type", width=12)
        table.add_column("Details", style="dim")

        for filename, diff_data in meta.items():
            if "functions" in diff_data:
                # python function diff
                stats = diff_data.get("stats", {})
                details = f"norule: {stats.get('norule_fn_count', 0)} fns, yerule: {stats.get('yerule_fn_count', 0)} fns"
                table.add_row(filename, "function diff", details)
            else:
                # text diff
                table.add_row(filename, "text diff", "content changed")

        self.console.print(table)

    def compare_runs(self, norule: RunData, yerule: RunData):
        """Compare norule and yerule runs side by side."""
        table = Table(title=f"Comparison: {norule.version}", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("No Rule", justify="right", style="red")
        table.add_column("With Rule", justify="right", style="green")

        # message counts
        norule_msgs = len(norule.transcript) if norule.transcript else 0
        yerule_msgs = len(yerule.transcript) if yerule.transcript else 0
        table.add_row("Messages", str(norule_msgs), str(yerule_msgs))

        # file counts
        table.add_row("Files", str(len(norule.files)), str(len(yerule.files)))

        # tool usage
        if norule.transcript and yerule.transcript:
            norule_tools = len(Filter.tools(norule.transcript))
            yerule_tools = len(Filter.tools(yerule.transcript))
            table.add_row("Tool Uses", str(norule_tools), str(yerule_tools))

            norule_code = len(Filter.code(norule.transcript))
            yerule_code = len(Filter.code(yerule.transcript))
            table.add_row("Code Changes", str(norule_code), str(yerule_code))

        self.console.print(table)

# %% [markdown]
"""## Example Usage"""

# %%
if __name__ == "__main__":
    # Example: load and visualize latest version
    outdir = default_outdir()
    versions = list_versions(outdir)

    if versions:
        latest = versions[-1]
        print(f"Loading version: {latest}")

        norule = load_run(outdir, latest, "norule")
        yerule = load_run(outdir, latest, "yerule")
        meta = load_meta(outdir, latest)

        viz = Viz()

        print("\n=== Comparison ===")
        viz.compare_runs(norule, yerule)

        print("\n=== No Rule Conversation ===")
        viz.convo(norule, limit=10)

        print("\n=== With Rule Tool Stats ===")
        viz.tool_stats(yerule)

        print("\n=== Meta Diff ===")
        viz.meta_diff(meta)
    else:
        print("No versions found in", outdir)
