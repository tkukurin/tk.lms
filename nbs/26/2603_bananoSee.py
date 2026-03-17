"""Review banano2 results from cache. Read-only viewer for iterating."""
# %%
import base64
import io
import sys
from dataclasses import asdict

import diskcache
from PIL import Image
from rich import print as rprint
from rich.table import Table

import tk

# Stub classes so pickle can deserialize cached dataclasses from banano2
# (pickle stores __main__.GemResult; we just need the names to exist —
# dataclass pickle reconstructs via __dict__, no field defs needed)
for _name in ("GemResult", "EvalResult", "RunStats"):
    setattr(sys.modules[__name__], _name, type(_name, (), {}))

cache = diskcache.Cache(tk.xpdir("out/banano2/cache"))

# %%
# ─── load all cached results ────────────────────────────────────────────────

def load_results() -> tuple[dict, dict]:
    """Load GemResults and EvalResults from cache. Returns (gen, eval) dicts."""
    gen, evl = {}, {}
    for key in cache.iterkeys():
        if isinstance(key, str) and key.startswith("gem__"):
            gen[key] = cache[key]
        elif isinstance(key, str) and key.startswith("eval__"):
            evl[key] = cache[key]
    return gen, evl

gen_results, eval_results = load_results()
rprint(f"[bold]{len(gen_results)}[/bold] gen results, [bold]{len(eval_results)}[/bold] eval results in cache")

# %%
# ─── summary table ──────────────────────────────────────────────────────────

def summary_table(gen: dict, evl: dict):
    t = Table(title="Banano2 Cache Summary")
    t.add_column("trial")
    t.add_column("images", justify="right")
    t.add_column("tokens", justify="right")
    t.add_column("lev_ratio", justify="right")
    t.add_column("font_match", justify="center")
    for key, res in sorted(gen.items()):
        n_img = len(res.image_parts) if hasattr(res, "image_parts") else len(res.get("image_parts", []))
        tok = res.total_tokens if hasattr(res, "total_tokens") else res.get("total_tokens", 0)
        short_key = key.replace("gem__", "")
        eval_key = f"eval__{short_key}"
        ev = evl.get(eval_key)
        lev = f"{ev.levenshtein_ratio:.2f}" if ev else "—"
        fm = "✓" if ev and ev.font_match else ("✗" if ev else "—")
        t.add_row(short_key, str(n_img), str(tok), lev, fm)
    rprint(t)

summary_table(gen_results, eval_results)

# %%
# ─── view images ────────────────────────────────────────────────────────────

def show_images(gen: dict, filter_str: str | None = None, max_cols: int = 4):
    """Show generated images. filter_str matches against trial key."""
    import matplotlib.pyplot as plt

    items = sorted(gen.items())
    if filter_str:
        items = [(k, v) for k, v in items if filter_str in k]

    imgs = []
    for key, res in items:
        parts = res.image_parts if hasattr(res, "image_parts") else res.get("image_parts", [])
        for ip in parts:
            data = ip["data"] if isinstance(ip, dict) else ip.data
            img = Image.open(io.BytesIO(base64.b64decode(data)))
            imgs.append((key.replace("gem__", ""), img))

    if not imgs:
        rprint("[red]No images found[/red]")
        return

    ncols = min(len(imgs), max_cols)
    nrows = (len(imgs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]
    for idx, (label, img) in enumerate(imgs):
        r, c = divmod(idx, ncols)
        axes[r][c].imshow(img)
        axes[r][c].set_title(label.replace("__", "\n"), fontsize=7)
        axes[r][c].axis("off")
    for idx in range(len(imgs), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")
    plt.tight_layout()
    plt.show()


# %%
#show_images(gen_results, filter_str="billboard")
show_images(gen_results, filter_str="serif")
# %%
