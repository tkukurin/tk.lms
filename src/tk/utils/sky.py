"""Small SkyPilot helpers.

Project-specific constants belong at call sites. This module only keeps shared
Task construction, launch, and artifact pulling. Workdir syncing is handled by
SkyPilot itself; add a repo-root ``.skyignore`` when ignored local files should
be uploaded instead of using ``.gitignore``.
"""
from __future__ import annotations

import subprocess
import sky as _sky
from collections.abc import Sequence
from pathlib import Path


def task(
    cmd: str,
    *,
    cluster: str,
    gpus: str,
    output: tuple[str, Sequence[str]] | None,
    setup_lines: Sequence[str],
    run_prefix_lines: Sequence[str] = (),
    secrets: dict[str, str] | None = None,
    file_mounts: dict[str, str] | None = None,
    workdir: str = ".",
    infra: str = "gcp",
    disk_size: int = 128,
    python: str = ".venv/bin/python",
    **kw,
):
    """Build a SkyPilot task with setup/run logging and optional artifacts."""

    bucket = None
    globs: tuple[str, ...] = ()
    if output is not None:
        bucket, output_globs = output
        globs = tuple(dict.fromkeys((*output_globs, "run.log")))

    copy_lines = "\n".join(
        f"  cp -r {glob} /output/ 2>/dev/null || true" for glob in globs
    )
    log_lines = ""
    if bucket:
        log_lines = (
            "  mkdir -p /output/logs\n"
            f"  cp run.log /output/logs/{cluster}.run.log 2>/dev/null || true\n"
            f"  cp run.log /output/logs/{cluster}.$(date +%Y%m%d-%H%M%S).log 2>/dev/null || true"
        )
    trap = "" if not bucket else (
        f"finish() {{ s=$?; mkdir -p /output\n{log_lines}\n{copy_lines}\n  exit $s; }}; trap finish EXIT"
    )
    setup_trap = "" if not bucket else (
        f"setup_finish() {{ s=$?; mkdir -p /output/logs\n"
        f"  cp setup.log /output/logs/{cluster}.setup.log 2>/dev/null || true\n"
        f"  cp setup.log /output/logs/{cluster}.setup.$(date +%Y%m%d-%H%M%S).log 2>/dev/null || true\n"
        "  exit $s; }; trap setup_finish EXIT"
    )

    setup = "\n".join(["set -euo pipefail", setup_trap, "{", "set -x", *setup_lines, "} 2>&1 | tee setup.log", ""])
    run = "\n".join(["set -euo pipefail", trap, *run_prefix_lines, f"{python} -u {cmd} 2>&1 | tee run.log", ""])

    return _sky.Task(
        name=cluster,
        workdir=workdir,
        secrets=secrets or {},
        file_mounts=file_mounts or {},
        storage_mounts={"/output": _sky.Storage(name=bucket)} if bucket else {},
        resources=_sky.Resources(
            infra=infra,
            accelerators=gpus or None,
            disk_size=disk_size,
        ),
        setup=setup,
        run=run,
        **kw)


def exec(
    cmd: str,
    *,
    cluster: str,
    gpus: str,
    output: tuple[str, Sequence[str]] | None,
    setup_lines: Sequence[str],
    run_prefix_lines: Sequence[str] = (),
    idle_mins: int | None = None,
    secrets: dict | None = None,
    file_mounts: dict[str, str] | None = None,
    workdir: str = ".",
    infra: str = "gcp",
    disk_size: int = 128,
    python: str = ".venv/bin/python",
    **kw,
):
    """Build a SkyPilot task, launch it, and stream logs."""
    t = task(
        cmd,
        cluster=cluster,
        gpus=gpus,
        output=output,
        setup_lines=setup_lines,
        run_prefix_lines=run_prefix_lines,
        secrets=secrets or {},
        file_mounts=file_mounts,
        workdir=workdir,
        infra=infra,
        disk_size=disk_size,
        python=python,
        **kw,
    )
    l = _sky.launch(t, cluster_name=cluster, down=bool(idle_mins), idle_minutes_to_autostop=idle_mins)
    return _sky.stream_and_get(l)


def pull(names: Sequence[str], dest, *, bucket: str) -> Path:
    """Copy bucket objects into local ``dest`` without rerunning a task."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        subprocess.run(
            ["gcloud", "storage", "cp", "-r", f"gs://{bucket}/{name}", str(dest)],
            check=True,
        )
    return dest
