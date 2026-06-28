"""Replace PELE's hand-engineered priors with a TabICL foundation model.

[pele] is ELO loop over 50k matches, GDP priors, 12-region coefficients, home-field altitude model, tilt ratings, score matrix
vs Transfermarkt "Lineup" features + in context learning

    ┌──────────────────────────────────────────────────────────────────┐
    │  ┌─────────────────────┐   ┌──────────────────────────────────┐  │
    │  │ 50k match results   │──▶│ Elo (harmonic margin) + home adv │  │
    │  └─────────────────────┘   └───────────────┬──────────────────┘  │
    │  ┌─────────────────────┐   ┌───────────────▼──────────────────┐  │
    │  │ GDP PPP + ...       │──▶│ GDP prior / initial ratings      │  │
    │  └─────────────────────┘   └───────────────┬──────────────────┘  │
    │                            ┌───────────────▼──────────────────┐  │
    │                            │ PELE rating + Tilt               │  │
    │                            └───────────────┬──────────────────┘  │
    │                            ┌───────────────▼──────────────────┐  │
    │                            │ Score matrix (neg-binomial)      │  │
    │                            └───────────────┬──────────────────┘  │
    └────────────────────────────────────────────┼────────────────────-┘
    ┌────────────────────────────────────┐       │
    │  Transfermarkt features ---------- │       │
    │  ─ top11 / top23 market value      │       │
    │  ─ value-weighted age              │       │
    │  ─ positional value allocation     │       │
    │  ─ attack tilt, star concentration │       │
    │  ─ domestic/UEFA club share        │       │
    └────────────────────────────┬───────┘       │
                          ┌──────▼───────────────▼──┐
                          │ TabICL Foundation Model │
                          └────────┬────────────────┘
                            ┌──────▼───────┐
                            │  W/D/L proba │
                            └──────────────┘

The bet: TabICL can implicitly learn what PELE explicitly encodes (home
advantage, team momentum, regional strength) from the tabular examples
provided in-context — without needing 50k match histories or hand-tuned
regional coefficients.

[pele]: https://www.natesilver.net/p/pele-methodology
[tabicl]: https://arxiv.org/abs/2502.05977
"""
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportReturnType=false, reportUnusedExpression=false
# pyright: reportCallIssue=false
# %% Imports
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from tabicl import TabICLClassifier

from tk import datadir
from tk.nbs.footie import FEAT_COLS, load_match_dataset, load_tm_features

FIWC_DIR = datadir / "transfermarkt" / "fiwc"
WORLD_CUP_YEARS = (2006, 2010, 2014, 2018, 2022)
VALID_YEARS = (2018, 2022)

# Ground truth: top-4 finishing order per tournament (Portuguese names from TM)
WC_FINISH = {
    2006: {"Itália": 1, "França": 2, "Alemanha": 3, "Portugal": 4},
    2010: {"Espanha": 1, "Holanda": 2, "Alemanha": 3, "Uruguai": 4},
    2014: {"Alemanha": 1, "Argentina": 2, "Holanda": 3, "Brasil": 4},
    2018: {"França": 1, "Croácia": 2, "Bélgica": 3, "Inglaterra": 4},
    2022: {"Argentina": 1, "França": 2, "Croácia": 3, "Marrocos": 4},
}
# Binary target: did the team reach the semi-finals (top 4)?
FEATURE_COLS = [
    "logMarketValue",
    "logTop11MarketValue",
    "logTop23MarketValue",
    "benchShare",
    "starConcentration",
    "top3Share",
    "top5Share",
    "valueWeightedAge",
    "ageMean",
    "ageStd",
    "attackTilt",
    "attackDefenseRatio",
    "gkValueShare",
    "defValueShare",
    "midValueShare",
    "fwdValueShare",
    "squadCount",
    "currentClubDiversity",
    "u23ValueShare",
    "u25ValueShare",
    "over30ValueShare",
    "domesticClubShare",
    "domesticValueShare",
    "uefaClubValueShare",
]


def mkpath(name: str, exist: str = "ok") -> callable:
    """Create output dir, return a path joiner."""
    out = datadir / "out" / name
    out.mkdir(parents=True, exist_ok=(exist == "ok"))
    return lambda fname="": out / fname if fname else out


mkout = mkpath("2606_tabicl_pele")
print(out_csv := mkout("df_ours.csv"))

# %% Load team features into a single DataFrame
def load_team_features(years: tuple[int, ...] = WORLD_CUP_YEARS) -> pd.DataFrame:
    """Load team_features.json for each year into a flat DataFrame."""
    rows = []
    for year in years:
        path = FIWC_DIR / str(year) / "team_features.json"
        if not path.exists():
            continue
        teams = json.loads(path.read_text())
        finish = WC_FINISH.get(year, {})
        for club_id, feats in teams.items():
            name = feats.get("name", "")
            rank = finish.get(name)
            rows.append({
                "year": year,
                "club_id": club_id,
                "name": name,
                "top4": int(rank is not None),
                "finish_rank": rank,
                **{col: feats.get(col) for col in FEATURE_COLS},
            })
    return pd.DataFrame(rows)


# %% The 50k matches ARE the training rows — enriched with TM roster features (box C)
tm_db = load_tm_features()
print(f"Teams in TM database: {len(tm_db)}, snapshots: {sum(len(v) for v in tm_db.values())}")

df_post2005 = load_match_dataset(tm_db)
df_post2005 = df_post2005[df_post2005["outcome"].notna()].copy()
print(f"Matches with TM features for both teams: {len(df_post2005)}")
print(f"Features per match: {len(FEAT_COLS)}")

# Split: train on pre-2018, validate on WC 2018+2022
VALID_CUTOFF = "2018-01-01"
df_train_m = df_post2005[df_post2005["date"] < VALID_CUTOFF]
df_valid_m = df_post2005[
    (df_post2005["date"] >= VALID_CUTOFF) &
    (df_post2005["tournament"] == "FIFA World Cup")
]

X_train = df_train_m[FEAT_COLS].values.astype(np.float32)
y_train = df_train_m["outcome"].values.astype(np.int64)
X_valid = df_valid_m[FEAT_COLS].values.astype(np.float32)
y_valid = df_valid_m["outcome"].values.astype(np.int64)
print(f"train: {X_train.shape}, valid (WC 2018+2022): {X_valid.shape}")
print(f"train outcome dist: {np.bincount(y_train)}")
print(f"valid outcome dist: {np.bincount(y_valid)}")
df_post2005.to_csv(out_csv, index=False)

# %% Download TabICL checkpoint
print(HF_TABICL_CKPT := Path(hf_hub_download(
    repo_id="jingang/TabICL",
    filename="tabicl-classifier-v2-20260212.ckpt",
)))
# %%
clf = TabICLClassifier(
    model_path=HF_TABICL_CKPT,
    # TabICL ensemble / preprocessing
    norm_methods=None,  # -> ["none", "power"] (no-op + Yeo-Johnson, split across estimators)
    feat_shuffle_method="latin",  # latin hypercube sampling of feature permutations
    class_shuffle_method="shift",  # cyclic shift of class labels per estimator
    outlier_threshold=4,  # clip features beyond 4 std
    support_many_classes=True,  # mixed-radix + hierarchical for n_classes > model max
    batch_size=8,  # samples per forward-pass batch
    kv_cache=False,  # no KV cache (recompute train context each predict)
    # TabICL checkpoint / inference runtime
    allow_auto_download=False,  # we already dl above
    checkpoint_version=HF_TABICL_CKPT.name,
    use_amp="auto",  # AMP heuristic based on input size/device
    use_fa3="auto",  # FA3 heuristic; only effective when flash-attn 3 is available
    offload_mode="auto",  # auto-select GPU/CPU/disk for column embedding outputs
    disk_offload_dir=None,
    # inference_config=dict(  # NOTE(tk) can set explicitly
    #     device=None, use_amp=True, use_fa3=True, verbose=False,
    #     # Batching
    #     min_batch_size=1, safety_factor=0.8,
    #     # Offloading
    #     offload=False, auto_offload_threshold=0.5,
    #     cpu_safety_factor=0.85, max_pinned_memory_mb=32768.0,
    #     # Disk offloading
    #     disk_offload_dir=None, disk_min_free_mb=1024.0, disk_flush_mb=8192.0,
    #     disk_cleanup=True, disk_file_prefix="", disk_dtype=None, disk_safety_factor=0.95,
    #     # Async transfer
    #     use_async=True, async_depth=4,
    # )
)

# %% Fit and predict
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_valid)
preds = clf.predict(X_valid)

# %% Evaluate: accuracy on WC 2018+2022 match outcomes
from sklearn.metrics import accuracy_score, classification_report  # noqa: E402

acc = accuracy_score(y_valid, preds)
print(f"\n=== WC match outcome accuracy: {acc:.3f} ===")
print(f"  (baseline majority-class: {np.bincount(y_valid).max() / len(y_valid):.3f})")
print(classification_report(y_valid, preds, target_names=["away_win", "draw", "home_win"]))

# Per-match predictions
df_valid_m = df_valid_m.copy()
df_valid_m["pred"] = preds
df_valid_m["correct"] = (preds == y_valid)
print("\nSample predictions (WC matches):")
cols = ["date", "home_team", "away_team", "home_score", "away_score", "outcome", "pred", "correct"]
print(df_valid_m[cols].head(20).to_string())

# %% Save results
df_valid_m.to_csv(mkout("df_valid_match_preds.csv"), index=False)
print(f"\nSaved to {mkout('df_valid_match_preds.csv')}")
