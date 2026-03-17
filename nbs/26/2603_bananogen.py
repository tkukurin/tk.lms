"""Nano Banana 2 — focused pilot: bash+email × serif+display, 30 samples/cell.

2×2 design: background (tmux, email) × font (serif, display)
Content: fixed bash prompt text
Metrics: Levenshtein ratio (continuous), font match (binary)

Tests:
  - Per-background permutation test on Levenshtein ratio
  - Per-background Fisher's exact on font match
  - Interaction: bootstrap CI on difference-of-differences
"""
# %%
import base64
import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import diskcache
import numpy as np
from google import genai
from google.genai import types
from rapidfuzz.distance import Levenshtein
from rich import print as rprint
from scipy import stats
from tqdm import tqdm

import tk

GEN_MODEL = "gemini-3.1-flash-image-preview"
JUDGE_MODEL = "gemini-3.1-preview"
BGDIR = tk.datadir / "bgnano2"
N_SAMPLES = 30
FONTS = ["serif", "display"]
RNG = np.random.default_rng(42)

cache = diskcache.Cache(tk.datadir / "out" / "bananogen" / "cache")
client = genai.Client()

SYSTEM_PROMPT = (
    "Generate faithful-looking documents with the exact content user provides "
    "and in the exact style they specify, using the user-provided image as background."
)

JUDGE_PROMPT = """\
You are evaluating an AI-generated document image.

1. OCR: Read ALL text visible in the image. Output it exactly as-is, preserving line breaks.
2. FONT: Classify the primary font style used in the image as exactly one of: serif, sans-serif, monospace, display.

Respond in this exact JSON format (no markdown fences):
{"ocr_text": "...", "detected_font": "..."}"""

BASH_TEXT = """\
toni@devbox:~/projects/webapp$ git status
On branch feature/auth-module
Changes not staged for commit:
  modified:   src/auth/handler.py
  modified:   src/auth/middleware.py
  deleted:    tests/test_old_auth.py

Untracked files:
  src/auth/oauth2.py
  src/auth/tokens.py

toni@devbox:~/projects/webapp$ python -m pytest tests/ -v --tb=short
========================= test session starts =========================
collected 47 items
tests/test_auth.py::test_login_success PASSED                    [  2%]
tests/test_auth.py::test_login_invalid_creds PASSED              [  4%]
tests/test_auth.py::test_token_refresh FAILED                    [  6%]
FAILED tests/test_auth.py::test_token_refresh - AssertionError: 401 != 200
===================== 2 passed, 1 failed in 0.34s ====================
toni@devbox:~/projects/webapp$"""

BACKGROUNDS = {
    "tmux": BGDIR / "tmux.png",
    "email": BGDIR / "email.png",
}

# %%
# ─── data structures ────────────────────────────────────────────────────────

@dataclass
class Sample:
    bg: str
    font: str
    idx: int  # 0..N_SAMPLES-1
    # gen
    image_b64: str = ""
    image_mime: str = ""
    gen_tokens: int = 0
    gen_cached: bool = False
    # eval
    ocr_text: str = ""
    detected_font: str = ""
    lev_ratio: float = 1.0
    font_match: bool = False
    eval_tokens: int = 0
    eval_cached: bool = False

    @property
    def key(self) -> str:
        return f"{self.bg}__{self.font}__{self.idx}"


# %%
# ─── generation ─────────────────────────────────────────────────────────────

def _gen_key(bg: str, font: str, idx: int, bg_bytes: bytes) -> str:
    h = hashlib.sha256(bg_bytes).hexdigest()[:12]
    return f"gen2__{bg}__{font}__{idx}__{h}"


def generate(sample: Sample) -> Sample:
    bg_bytes = BACKGROUNDS[sample.bg].read_bytes()
    ck = _gen_key(sample.bg, sample.font, sample.idx, bg_bytes)

    if ck in cache:
        cached = cache[ck]
        sample.image_b64 = cached["image_b64"]
        sample.image_mime = cached["image_mime"]
        sample.gen_tokens = cached["gen_tokens"]
        sample.gen_cached = True
        return sample

    mime = "image/png" if sample.bg.endswith(".png") else "image/jpeg"
    image_part = types.Part.from_bytes(data=bg_bytes, mime_type=mime)
    text = f"Use font style: {sample.font}. Output content:\n{BASH_TEXT}"

    response = client.models.generate_content(
        model=GEN_MODEL,
        contents=[types.Content(role="user", parts=[image_part, types.Part.from_text(text=text)])],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_modalities=["TEXT", "IMAGE"],
        ),
    )

    um = response.usage_metadata
    sample.gen_tokens = (um.total_token_count or 0) if um else 0

    if response.candidates:
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                sample.image_b64 = base64.b64encode(part.inline_data.data).decode()
                sample.image_mime = part.inline_data.mime_type
                break

    cache[ck] = {"image_b64": sample.image_b64, "image_mime": sample.image_mime,
                 "gen_tokens": sample.gen_tokens}
    return sample


# %%
# ─── judging ────────────────────────────────────────────────────────────────

def _eval_key(sample: Sample) -> str:
    return f"eval2__{sample.key}"


def judge(sample: Sample) -> Sample:
    if not sample.image_b64:
        return sample

    ck = _eval_key(sample)
    if ck in cache:
        cached = cache[ck]
        sample.ocr_text = cached["ocr_text"]
        sample.detected_font = cached["detected_font"]
        sample.lev_ratio = cached["lev_ratio"]
        sample.font_match = cached["font_match"]
        sample.eval_tokens = cached["eval_tokens"]
        sample.eval_cached = True
        return sample

    img_bytes = base64.b64decode(sample.image_b64)
    image_part = types.Part.from_bytes(data=img_bytes, mime_type=sample.image_mime)

    response = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=[types.Content(role="user", parts=[image_part, types.Part.from_text(text=JUDGE_PROMPT)])],
        config=types.GenerateContentConfig(response_modalities=["TEXT"]),
    )

    um = response.usage_metadata
    sample.eval_tokens = (um.total_token_count or 0) if um else 0

    raw = response.text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        parsed = json.loads(raw)
        sample.ocr_text = parsed.get("ocr_text", "")
        sample.detected_font = parsed.get("detected_font", "unknown").lower().strip()
    except json.JSONDecodeError:
        sample.ocr_text = raw
        sample.detected_font = "unknown"

    max_len = max(len(BASH_TEXT), len(sample.ocr_text), 1)
    sample.lev_ratio = Levenshtein.distance(BASH_TEXT, sample.ocr_text) / max_len
    sample.font_match = sample.detected_font == sample.font

    cache[ck] = {"ocr_text": sample.ocr_text, "detected_font": sample.detected_font,
                 "lev_ratio": sample.lev_ratio, "font_match": sample.font_match,
                 "eval_tokens": sample.eval_tokens}
    return sample


# %%
# ─── run ────────────────────────────────────────────────────────────────────

def run_experiment() -> list[Sample]:
    samples = [Sample(bg=bg, font=font, idx=i)
               for bg in BACKGROUNDS for font in FONTS for i in range(N_SAMPLES)]
    for s in tqdm(samples, desc="generate"):
        generate(s)
        hit = "HIT" if s.gen_cached else "MISS"
        tqdm.write(f"  [{hit}] {s.key}: {'img' if s.image_b64 else 'NO IMG'} {s.gen_tokens}tok")
    for s in tqdm(samples, desc="judge"):
        judge(s)
        hit = "HIT" if s.eval_cached else "MISS"
        tqdm.write(f"  [{hit}] {s.key}: lev={s.lev_ratio:.2f} font={'✓' if s.font_match else '✗'}")
    return samples


# %%
# ─── analysis ───────────────────────────────────────────────────────────────

def permutation_test(a: np.ndarray, b: np.ndarray, n_perm: int = 10_000) -> float:
    """Two-sided permutation test on mean difference."""
    obs = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    n_a = len(a)
    count = 0
    for _ in range(n_perm):
        RNG.shuffle(pooled)
        count += abs(pooled[:n_a].mean() - pooled[n_a:].mean()) >= obs
    return count / n_perm


def bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Bootstrap CI on mean(a) - mean(b). Returns (mean_diff, lo, hi)."""
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sa = RNG.choice(a, size=len(a), replace=True)
        sb = RNG.choice(b, size=len(b), replace=True)
        diffs[i] = sa.mean() - sb.mean()
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(diffs.mean()), float(lo), float(hi)


def analyze(samples: list[Sample]):
    """Per-background tests + interaction check."""

    def split(samples, bg, font):
        return np.array([s.lev_ratio for s in samples if s.bg == bg and s.font == font])

    def font_counts(samples, bg, font):
        ss = [s for s in samples if s.bg == bg and s.font == font]
        match = sum(s.font_match for s in ss)
        return match, len(ss) - match

    rprint("\n[bold]═══ Analysis ═══[/bold]")

    gen_tok = sum(s.gen_tokens for s in samples if not s.gen_cached)
    eval_tok = sum(s.eval_tokens for s in samples if not s.eval_cached)
    rprint(f"Cost: {gen_tok} gen tokens, {eval_tok} eval tokens "
           f"({sum(s.gen_cached for s in samples)}/{len(samples)} gen cached, "
           f"{sum(s.eval_cached for s in samples)}/{len(samples)} eval cached)")

    diffs = {}
    for bg in BACKGROUNDS:
        serif = split(samples, bg, "serif")
        display = split(samples, bg, "display")

        rprint(f"\n[bold]── {bg} ──[/bold]")
        rprint(f"  Levenshtein ratio: serif={serif.mean():.3f}±{serif.std():.3f}  "
               f"display={display.mean():.3f}±{display.std():.3f}")

        p_perm = permutation_test(serif, display)
        mean_d, lo, hi = bootstrap_ci(serif, display)
        rprint(f"  Permutation test p={p_perm:.4f}, diff={mean_d:.3f} CI=[{lo:.3f}, {hi:.3f}]")
        diffs[bg] = mean_d

        # Fisher's exact on font match
        s_match, s_miss = font_counts(samples, bg, "serif")
        d_match, d_miss = font_counts(samples, bg, "display")
        table = [[s_match, s_miss], [d_match, d_miss]]
        _, p_fisher = stats.fisher_exact(table)
        rprint(f"  Font match: serif={s_match}/{s_match+s_miss}  display={d_match}/{d_match+d_miss}  "
               f"Fisher p={p_fisher:.4f}")

    # Interaction: difference of differences
    rprint("\n[bold]── Interaction ──[/bold]")
    bgs = list(BACKGROUNDS.keys())
    serif_0 = split(samples, bgs[0], "serif")
    display_0 = split(samples, bgs[0], "display")
    serif_1 = split(samples, bgs[1], "serif")
    display_1 = split(samples, bgs[1], "display")

    n_boot = 10_000
    dod = np.empty(n_boot)
    for i in range(n_boot):
        d0 = RNG.choice(serif_0, len(serif_0), replace=True).mean() - RNG.choice(display_0, len(display_0), replace=True).mean()
        d1 = RNG.choice(serif_1, len(serif_1), replace=True).mean() - RNG.choice(display_1, len(display_1), replace=True).mean()
        dod[i] = d0 - d1
    lo, hi = np.percentile(dod, [2.5, 97.5])
    rprint(f"  Diff-of-diffs ({bgs[0]} vs {bgs[1]}): {dod.mean():.3f} CI=[{lo:.3f}, {hi:.3f}]")
    if lo <= 0 <= hi:
        rprint("  [green]CI includes 0 → font effect consistent across backgrounds (can pool)[/green]")
    else:
        rprint("  [yellow]CI excludes 0 → font effect differs by background (interaction)[/yellow]")


# %%
# ─── inline tests ───────────────────────────────────────────────────────────

def test_backgrounds_exist():
    for name, path in BACKGROUNDS.items():
        assert path.exists(), f"missing: {path}"
    rprint(f"[green]✓ test_backgrounds_exist ({len(BACKGROUNDS)})[/green]")


def test_cache_hit_miss():
    tc = diskcache.Cache(tk.datadir / "out" / "bananogen" / "cache" / "_test")
    k = "_test_hm"
    if k in tc: del tc[k]
    assert k not in tc
    tc[k] = 42
    assert k in tc and tc[k] == 42
    del tc[k]
    tc.close()
    rprint("[green]✓ test_cache_hit_miss[/green]")


def test_gen_key_deterministic():
    d = b"fake"
    assert _gen_key("a", "b", 0, d) == _gen_key("a", "b", 0, d)
    assert _gen_key("a", "b", 0, d) != _gen_key("a", "b", 1, d)
    rprint("[green]✓ test_gen_key_deterministic[/green]")


def test_sample_key():
    s = Sample(bg="tmux", font="serif", idx=7)
    assert s.key == "tmux__serif__7"
    rprint("[green]✓ test_sample_key[/green]")


def test_permutation():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    p = permutation_test(a, b)
    assert p > 0.5, f"identical should have high p, got {p}"
    a2 = np.array([100.0, 101.0, 102.0])
    p2 = permutation_test(a2, b)
    assert p2 < 0.1, f"different should have low p, got {p2}"
    rprint("[green]✓ test_permutation[/green]")


def test_bootstrap_ci():
    a = np.ones(30) * 10
    b = np.ones(30) * 10
    mean_d, lo, hi = bootstrap_ci(a, b)
    assert abs(mean_d) < 0.01 and lo <= 0 <= hi
    rprint("[green]✓ test_bootstrap_ci[/green]")


def test_levenshtein():
    assert Levenshtein.distance("hello", "hello") == 0
    assert Levenshtein.distance("abc", "xyz") == 3
    rprint("[green]✓ test_levenshtein[/green]")


def run_tests():
    test_backgrounds_exist()
    test_cache_hit_miss()
    test_gen_key_deterministic()
    test_sample_key()
    test_permutation()
    test_bootstrap_ci()
    test_levenshtein()
    rprint("\n[bold green]All tests passed![/bold green]")


# %%
if __name__ == "__main__":
    run_tests()

# %%
samples = run_experiment()
analyze(samples)
# %%
