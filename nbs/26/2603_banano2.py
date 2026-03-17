# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "google-genai",
#     "diskcache",
#     "pillow",
#     "rich",
#     "tqdm",
#     "rapidfuzz",
# ]
# ///
# NOTE: run via `uv run --with google-genai python nbs/26/2603_banano2.py`
# (uses project venv for tk, adds google-genai on top)
"""Test Nano Banana 2 (Gemini 3.1 Flash Image) text/character generation in documents.

Dimensions:
  - background: billboard, document (paper), bash terminal, email compose
    (loaded from data/bgnano/ with convention prefixNN.suffix)
  - content: receipt, instruction manual, ad, random english, random croatian, bash prompt
  - font style: serif, sans-serif, monospace, display

Prompt structure:
  system: "Generate faithful-looking documents with the exact content user
           provides and in the exact style they specify, using the user-provided
           image as background."
  chunk 1: image of the background
  chunk 2: "Use font style: {font}. Output content:\\n{text}"

Image resolution / aspect ratio: API defaults (no principled reason to change).

Evaluation:
  - gemini-3.1-preview as judge: OCR the generated image, identify output font
  - Levenshtein distance (rapidfuzz) between ground-truth text and OCR'd text
  - Font classification accuracy (expected vs detected)

Cost tracking: token counts from response.usage_metadata
Cache tracking: manual get/set on diskcache to distinguish hits from misses
"""
# %%
import base64
import hashlib
import io
import itertools as it
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import NamedTuple

import diskcache
from google import genai
from google.genai import types
from PIL import Image
from rapidfuzz.distance import Levenshtein
from rich import print as rprint
from tqdm import tqdm

import tk

GEN_MODEL = "gemini-3.1-flash-image-preview"
JUDGE_MODEL = "gemini-3.1-preview"
BGDIR = tk.datadir / "bgnano"

cache = diskcache.Cache(tk.xpdir("out/banano2/cache"))
client = genai.Client()  # uses GOOGLE_API_KEY env

# %%
# ─── backgrounds ────────────────────────────────────────────────────────────

class Background(NamedTuple):
    name: str       # e.g. "billboard01"
    path: Path
    __repr__ = lambda s: f"🖼️【{s.name}】"


def get_backgrounds(bgdir: Path = BGDIR) -> list[Background]:
    """Load backgrounds from bgdir matching prefixNN.suffix convention."""
    pat = re.compile(r"^[a-zA-Z]+\d{2}\.\w+$")
    files = sorted(f for f in bgdir.iterdir() if f.is_file() and pat.match(f.name))
    return [Background(f.stem, f) for f in files]


def load_bg(bg: Background) -> bytes:
    return bg.path.read_bytes()


# %%
# ─── content texts ──────────────────────────────────────────────────────────

class DocContent(NamedTuple):
    name: str
    text: str
    __repr__ = lambda s: f"📄【{s.name}•{s.text[:30]}...】"

CONTENTS = [
    DocContent("receipt", """\
RECEIPT — BuildRight Construction LLC
Date: 2026-03-15  Invoice #: BR-2026-0471

Description                    Qty    Unit Price    Total
─────────────────────────────────────────────────────────
Foundation excavation           1     $12,400.00    $12,400.00
Concrete pour (grade C30)       48m³     $185.00     $8,880.00
Steel rebar framing             1      $3,200.00     $3,200.00
Waterproofing membrane          1      $1,750.00     $1,750.00
─────────────────────────────────────────────────────────
                              Subtotal:              $26,230.00
                              Tax (8.5%):             $2,229.55
                              TOTAL:                 $28,459.55

Payment due within 30 days. Thank you for your business."""),

    DocContent("manual", """\
INSTRUCTION MANUAL — Model X-47 Precision Drill

1. SAFETY WARNINGS
   • Always wear protective eyewear and gloves.
   • Do not operate near flammable materials.
   • Disconnect power before changing drill bits.

2. ASSEMBLY
   a) Attach the side handle by threading it clockwise.
   b) Insert the drill bit into the chuck and tighten.
   c) Connect the battery pack until it clicks.

3. OPERATION
   Press the trigger gently to start. Use the speed dial
   (1–5) to adjust RPM. For reverse, toggle the switch
   above the trigger.

4. MAINTENANCE
   Clean after each use. Lubricate the chuck monthly."""),

    DocContent("ad", """\
★ GRAND OPENING — FRESH BITES BAKERY ★

🍞  Artisan breads baked daily at 5 AM
🥐  Croissants: butter, almond, chocolate
🎂  Custom cakes for every occasion

THIS WEEK ONLY:
  Buy 2 loaves, get 1 FREE!
  Free coffee with any pastry purchase!

Visit us at 742 Evergreen Terrace
Open Mon–Sat 6AM–8PM, Sun 7AM–3PM
Phone: (555) 012-3456  |  @freshbitesbakery"""),

    DocContent("english_random", """\
The history of cartography stretches back millennia. Ancient
Babylonians inscribed maps on clay tablets around 600 BCE,
depicting their known world as a flat disk surrounded by ocean.
Greek scholars like Eratosthenes calculated Earth's circumference
with remarkable accuracy using shadows and geometry. During the
Age of Exploration, Mercator's 1569 projection revolutionized
navigation by preserving straight-line compass bearings. Today,
satellite imagery and GIS systems can resolve features smaller
than a meter, yet the fundamental challenge remains: faithfully
representing a curved surface on a flat medium."""),

    DocContent("croatian_random", """\
Dubrovačke gradske zidine protežu se gotovo dva kilometra oko
povijesne jezgre grada. Izgradnja je započela u 8. stoljeću, a
današnji oblik duguju uglavnom pregradnjama iz 14. i 15. stoljeća.
Zidine su debele do šest metara na kopnenoj strani i do tri metra
prema moru. Tvrđava Minčeta, smještena na najvišoj točki, pruža
pogled na cijeli grad i obližnje otoke. Grad je preživio potres
1667. godine te Domovinski rat, čime svjedoči o izvanrednoj
otpornosti i graditeljskom umijeću svojih stanovnika."""),

    DocContent("bash_prompt", """\
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
toni@devbox:~/projects/webapp$"""),
]

# %%
# ─── font styles ────────────────────────────────────────────────────────────

FONTS = ["serif", "sans-serif", "monospace", "display"]

# %%
# ─── Gemini API with manual cache (hit/miss tracking) ──────────────────────

SYSTEM_PROMPT = (
    "Generate faithful-looking documents with the exact content user provides "
    "and in the exact style they specify, using the user-provided image as background."
)


@dataclass
class GemResult:
    """Result from a single generation call."""
    text_parts: list[str] = field(default_factory=list)
    image_parts: list[dict] = field(default_factory=list)  # [{mime_type, data(b64)}]
    prompt_tokens: int = 0
    candidates_tokens: int = 0
    total_tokens: int = 0
    cached: bool = False
    bg_name: str = ""
    content_name: str = ""
    font: str = ""


def _cache_key(bg_name: str, content_name: str, font: str, bg_bytes: bytes) -> str:
    """Deterministic cache key from trial parameters."""
    h = hashlib.sha256(bg_bytes).hexdigest()[:16]
    return f"gem__{bg_name}__{content_name}__{font}__{h}"


def gem(bg_name: str, content_name: str, font: str,
        bg_bytes: bytes, text: str) -> GemResult:
    """Call Gemini for image generation; manual cache with hit/miss tracking."""
    key = _cache_key(bg_name, content_name, font, bg_bytes)

    if key in cache:
        result = cache[key]
        result.cached = True
        return result

    mime = "image/png" if bg_name.endswith(".png") else "image/jpeg"
    image_part = types.Part.from_bytes(data=bg_bytes, mime_type=mime)
    user_text = f"Use font style: {font}. Output content:\n{text}"

    response = client.models.generate_content(
        model=GEN_MODEL,
        contents=[
            types.Content(role="user", parts=[
                image_part,
                types.Part.from_text(text=user_text),
            ]),
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_modalities=["TEXT", "IMAGE"],
        ),
    )

    result = GemResult(bg_name=bg_name, content_name=content_name, font=font)

    # usage / cost
    um = response.usage_metadata
    if um:
        result.prompt_tokens = um.prompt_token_count or 0
        result.candidates_tokens = um.candidates_token_count or 0
        result.total_tokens = um.total_token_count or 0

    if response.candidates:
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                result.text_parts.append(part.text)
            elif part.inline_data is not None:
                result.image_parts.append({
                    "mime_type": part.inline_data.mime_type,
                    "data": base64.b64encode(part.inline_data.data).decode(),
                })

    cache[key] = result
    return result


# %%
# ─── experiment ─────────────────────────────────────────────────────────────

class Trial(NamedTuple):
    bg: Background
    content: DocContent
    font: str
    __repr__ = lambda s: f"🧪【{s.bg.name}×{s.content.name}×{s.font}】"


def get_trials() -> list[Trial]:
    bgs = get_backgrounds()
    return [Trial(bg, content, font)
            for bg, content, font in it.product(bgs, CONTENTS, FONTS)]


@dataclass
class RunStats:
    total: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_prompt_tokens: int = 0
    total_candidates_tokens: int = 0
    total_tokens: int = 0


def xprun() -> tuple[dict[str, GemResult], RunStats]:
    """Run all trials. Returns {trial_key: GemResult} and aggregate stats."""
    trials = get_trials()
    results = {}
    stats = RunStats(total=len(trials))

    for trial in tqdm(trials, desc="banano2"):
        key = f"{trial.bg.name}__{trial.content.name}__{trial.font}"
        bg_bytes = load_bg(trial.bg)
        result = gem(trial.bg.name, trial.content.name, trial.font,
                     bg_bytes, trial.content.text)
        results[key] = result

        if result.cached:
            stats.cache_hits += 1
        else:
            stats.cache_misses += 1
            stats.total_prompt_tokens += result.prompt_tokens
            stats.total_candidates_tokens += result.candidates_tokens
            stats.total_tokens += result.total_tokens

        n_img = len(result.image_parts)
        hit = "HIT" if result.cached else "MISS"
        tqdm.write(f"  [{hit}] {key}: {n_img} img, {result.total_tokens} tok")

    return results, stats


# %%
# ─── evaluation: OCR + font detection via Gemini judge ──────────────────────

JUDGE_PROMPT = """\
You are evaluating an AI-generated document image.

1. OCR: Read ALL text visible in the image. Output it exactly as-is, preserving line breaks.
2. FONT: Classify the primary font style used in the image as exactly one of: serif, sans-serif, monospace, display.

Respond in this exact JSON format (no markdown fences):
{"ocr_text": "...", "detected_font": "..."}"""


@dataclass
class EvalResult:
    trial_key: str
    expected_text: str
    expected_font: str
    ocr_text: str
    detected_font: str
    levenshtein_dist: int
    levenshtein_ratio: float  # 0.0 = identical, 1.0 = completely different
    font_match: bool
    judge_tokens: int = 0
    cached: bool = False


def _eval_cache_key(trial_key: str) -> str:
    return f"eval__{trial_key}"


def judge_image(trial_key: str, image_b64: str, mime_type: str,
                expected_text: str, expected_font: str) -> EvalResult:
    """Use Gemini judge to OCR and detect font from a generated image."""
    ck = _eval_cache_key(trial_key)

    if ck in cache:
        result = cache[ck]
        result.cached = True
        return result

    img_bytes = base64.b64decode(image_b64)
    image_part = types.Part.from_bytes(data=img_bytes, mime_type=mime_type)

    response = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=[
            types.Content(role="user", parts=[
                image_part,
                types.Part.from_text(text=JUDGE_PROMPT),
            ]),
        ],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT"],
        ),
    )

    judge_tokens = 0
    if response.usage_metadata:
        judge_tokens = response.usage_metadata.total_token_count or 0

    raw = response.text.strip()
    # strip markdown fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        parsed = json.loads(raw)
        ocr_text = parsed.get("ocr_text", "")
        detected_font = parsed.get("detected_font", "unknown")
    except json.JSONDecodeError:
        ocr_text = raw
        detected_font = "unknown"

    lev_dist = Levenshtein.distance(expected_text, ocr_text)
    max_len = max(len(expected_text), len(ocr_text), 1)
    lev_ratio = lev_dist / max_len

    result = EvalResult(
        trial_key=trial_key,
        expected_text=expected_text,
        expected_font=expected_font,
        ocr_text=ocr_text,
        detected_font=detected_font.lower().strip(),
        levenshtein_dist=lev_dist,
        levenshtein_ratio=lev_ratio,
        font_match=(detected_font.lower().strip() == expected_font.lower().strip()),
        judge_tokens=judge_tokens,
    )
    cache[ck] = result
    return result


def xpeval(results: dict[str, GemResult]) -> list[EvalResult]:
    """Evaluate all generated images via judge model."""
    evals = []
    for key, res in tqdm(results.items(), desc="eval"):
        if not res.image_parts:
            continue
        img = res.image_parts[0]  # evaluate first image
        # find the matching content text
        content_text = next(
            (c.text for c in CONTENTS if c.name == res.content_name), "")
        ev = judge_image(key, img["data"], img["mime_type"],
                         content_text, res.font)
        hit = "HIT" if ev.cached else "MISS"
        tqdm.write(f"  [{hit}] {key}: lev={ev.levenshtein_ratio:.2f} font={'✓' if ev.font_match else '✗'}")
        evals.append(ev)
    return evals


# %%
# ─── display helpers ────────────────────────────────────────────────────────

def get_images(results: dict[str, GemResult]) -> dict[str, list[Image.Image]]:
    out = {}
    for key, res in results.items():
        imgs = []
        for ip in res.image_parts:
            img = Image.open(io.BytesIO(base64.b64decode(ip["data"])))
            imgs.append(img)
        out[key] = imgs
    return out


def show_grid(results: dict[str, GemResult], max_cols: int = 4):
    import matplotlib.pyplot as plt

    images = get_images(results)
    by_bg: dict[str, list[str]] = {}
    for key, res in results.items():
        by_bg.setdefault(res.bg_name, []).append(key)

    for bg_name, keys in by_bg.items():
        valid = [(k, images[k][0]) for k in keys if images.get(k)]
        if not valid:
            continue
        ncols = min(len(valid), max_cols)
        nrows = (len(valid) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
        if nrows == 1 and ncols == 1:
            axes = [[axes]]
        elif nrows == 1:
            axes = [axes]
        elif ncols == 1:
            axes = [[ax] for ax in axes]
        fig.suptitle(bg_name, fontsize=14)
        for idx, (k, img) in enumerate(valid):
            r, c = divmod(idx, ncols)
            axes[r][c].imshow(img)
            label = k.replace(f"{bg_name}__", "").replace("__", "\n")
            axes[r][c].set_title(label, fontsize=8)
            axes[r][c].axis("off")
        for idx in range(len(valid), nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r][c].axis("off")
        plt.tight_layout()
        plt.show()


def print_eval_summary(evals: list[EvalResult]):
    """Print aggregate eval metrics."""
    if not evals:
        rprint("[red]No eval results[/red]")
        return
    avg_lev = sum(e.levenshtein_ratio for e in evals) / len(evals)
    font_acc = sum(e.font_match for e in evals) / len(evals)
    total_judge_tok = sum(e.judge_tokens for e in evals)
    rprint(f"[bold]Eval summary[/bold] ({len(evals)} images)")
    rprint(f"  Avg Levenshtein ratio: {avg_lev:.3f} (0=perfect)")
    rprint(f"  Font accuracy: {font_acc:.1%}")
    rprint(f"  Judge tokens: {total_judge_tok}")


# %%
# ─── inline tests ───────────────────────────────────────────────────────────

def test_backgrounds_load():
    """Backgrounds in data/bgnano/ match prefixNN.suffix convention."""
    bgs = get_backgrounds()
    assert len(bgs) > 0, "no backgrounds found in data/bgnano/"
    for bg in bgs:
        assert re.match(r"^[a-zA-Z]+\d{2}$", bg.name), f"bad name: {bg.name}"
        assert bg.path.exists(), f"missing: {bg.path}"
    rprint(f"[green]✓ test_backgrounds_load: {len(bgs)} backgrounds[/green]")


def test_load_bg_bytes():
    """Can read background image bytes."""
    bgs = get_backgrounds()
    data = load_bg(bgs[0])
    assert isinstance(data, bytes)
    assert len(data) > 100, "image too small"
    rprint(f"[green]✓ test_load_bg_bytes: {len(data)} bytes[/green]")


def test_contents_nonempty():
    """All CONTENTS have name and nonempty text."""
    for c in CONTENTS:
        assert c.name, "empty name"
        assert len(c.text) > 20, f"{c.name}: text too short"
    rprint(f"[green]✓ test_contents_nonempty: {len(CONTENTS)} contents[/green]")


def test_cache_key_deterministic():
    """Cache keys are deterministic for same inputs."""
    data = b"fake image bytes"
    k1 = _cache_key("bg", "content", "serif", data)
    k2 = _cache_key("bg", "content", "serif", data)
    assert k1 == k2, "non-deterministic cache key"
    k3 = _cache_key("bg", "content", "mono", data)
    assert k1 != k3, "different font should give different key"
    rprint("[green]✓ test_cache_key_deterministic[/green]")


def test_cache_hit_miss_tracking():
    """Verify diskcache manual hit/miss tracking works."""
    test_cache = diskcache.Cache(tk.xpdir("out/banano2/cache") / "_test")
    test_key = "_test_hit_miss"
    # ensure clean
    if test_key in test_cache:
        del test_cache[test_key]

    assert test_key not in test_cache, "key should not exist yet"
    test_cache[test_key] = {"value": 42}
    assert test_key in test_cache, "key should exist after set"
    assert test_cache[test_key]["value"] == 42
    del test_cache[test_key]
    assert test_key not in test_cache, "key should be gone after delete"
    test_cache.close()
    rprint("[green]✓ test_cache_hit_miss_tracking[/green]")


def test_gem_result_dataclass():
    """GemResult serializable and fields propagate."""
    r = GemResult(bg_name="billboard01", content_name="receipt", font="serif",
                  prompt_tokens=100, total_tokens=200)
    d = asdict(r)
    assert d["bg_name"] == "billboard01"
    assert d["font"] == "serif"
    assert d["cached"] is False
    rprint("[green]✓ test_gem_result_dataclass[/green]")


def test_levenshtein_scoring():
    """Levenshtein distance and ratio computation."""
    assert Levenshtein.distance("hello", "hello") == 0
    assert Levenshtein.distance("hello", "helo") == 1
    d = Levenshtein.distance("abc", "xyz")
    assert d == 3
    rprint("[green]✓ test_levenshtein_scoring[/green]")


def test_eval_result_font_match():
    """EvalResult font_match logic."""
    er = EvalResult(
        trial_key="test", expected_text="a", expected_font="serif",
        ocr_text="a", detected_font="serif",
        levenshtein_dist=0, levenshtein_ratio=0.0, font_match=True)
    assert er.font_match is True
    er2 = EvalResult(
        trial_key="test", expected_text="a", expected_font="serif",
        ocr_text="a", detected_font="monospace",
        levenshtein_dist=0, levenshtein_ratio=0.0, font_match=False)
    assert er2.font_match is False
    rprint("[green]✓ test_eval_result_font_match[/green]")


def test_trials_count():
    """Trial count = backgrounds × contents × fonts."""
    bgs = get_backgrounds()
    trials = get_trials()
    expected = len(bgs) * len(CONTENTS) * len(FONTS)
    assert len(trials) == expected, f"{len(trials)} != {expected}"
    rprint(f"[green]✓ test_trials_count: {len(trials)} trials[/green]")


def run_tests():
    test_backgrounds_load()
    test_load_bg_bytes()
    test_contents_nonempty()
    test_cache_key_deterministic()
    test_cache_hit_miss_tracking()
    test_gem_result_dataclass()
    test_levenshtein_scoring()
    test_eval_result_font_match()
    test_trials_count()
    rprint("\n[bold green]All tests passed![/bold green]")


# %%
if __name__ == "__main__":
    run_tests()

# %%
results, stats = xprun()
# %%
# rprint(stats)
# rprint(f"Completed {len(results)} trials")
# evals = xpeval(results)
# print_eval_summary(evals)
# show_grid(results)
# %%
