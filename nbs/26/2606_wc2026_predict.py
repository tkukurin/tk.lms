"""WC 2026 match-outcome predictions using TabICL + TM roster features.

Train on all pre-2026 international matches (with TM data for both teams),
predict WC 2026 group-stage + knockout outcomes, evaluate against actuals.
"""
# pyright: reportArgumentType=false, reportReturnType=false
# %% Imports
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score, classification_report
from tabicl import TabICLClassifier

from tk import datadir
from tk.nbs.footie import FEAT_COLS, load_match_dataset, load_tm_features

mkout = lambda name: (p := datadir / "out" / name, p.mkdir(parents=True, exist_ok=True))[0]
OUT = mkout("2606_wc2026_preds")

# %% Load data
tm_db = load_tm_features()
print(f"TM database: {len(tm_db)} teams, {sum(len(v) for v in tm_db.values())} snapshots")

df = load_match_dataset(tm_db)
print(f"Total matched matches: {len(df)}")

# %% Split: train on everything before WC 2026, predict WC 2026
TRAIN_CUTOFF = "2026-06-01"
df_train = df[(df["date"] < TRAIN_CUTOFF) & df["outcome"].notna()].copy()
df_wc26 = df[
    (df["date"] >= TRAIN_CUTOFF) & (df["tournament"] == "FIFA World Cup")
].copy()

# Separate played (for evaluation) and unplayed (for pure prediction)
df_wc26_played = df_wc26[df_wc26["outcome"].notna()].copy()
df_wc26_unplayed = df_wc26[df_wc26["outcome"].isna()].copy()

X_train = df_train[FEAT_COLS].values.astype(np.float32)
y_train = df_train["outcome"].values.astype(np.int64)
print(f"Train: {X_train.shape}, outcome dist: {np.bincount(y_train)}")
print(f"WC 2026 played: {len(df_wc26_played)}, unplayed: {len(df_wc26_unplayed)}")

# %% TabICL
HF_TABICL_CKPT = Path(hf_hub_download(
    repo_id="jingang/TabICL",
    filename="tabicl-classifier-v2-20260212.ckpt",
))

clf = TabICLClassifier(
    model_path=HF_TABICL_CKPT,
    norm_methods=None,
    feat_shuffle_method="latin",
    class_shuffle_method="shift",
    outlier_threshold=4,
    support_many_classes=True,
    batch_size=8,
    kv_cache=False,
    allow_auto_download=False,
    checkpoint_version=HF_TABICL_CKPT.name,
    use_amp="auto",
    use_fa3="auto",
    offload_mode="auto",
    disk_offload_dir=None,
)

# %% Fit
clf.fit(X_train, y_train)

# %% Predict on played WC 2026 matches (evaluation)
if len(df_wc26_played) > 0:
    X_eval = df_wc26_played[FEAT_COLS].values.astype(np.float32)
    y_eval = df_wc26_played["outcome"].values.astype(np.int64)
    preds = clf.predict(X_eval)
    proba = clf.predict_proba(X_eval)

    acc = accuracy_score(y_eval, preds)
    baseline = np.bincount(y_eval).max() / len(y_eval)
    print(f"\n=== WC 2026 match accuracy: {acc:.3f} (baseline: {baseline:.3f}) ===")
    print(classification_report(
        y_eval, preds, target_names=["away_win", "draw", "home_win"],
        zero_division=0,
    ))

    df_wc26_played["pred"] = preds
    df_wc26_played["pred_label"] = np.where(preds == 2, "H", np.where(preds == 1, "D", "A"))
    df_wc26_played["actual_label"] = np.where(y_eval == 2, "H", np.where(y_eval == 1, "D", "A"))
    df_wc26_played["correct"] = preds == y_eval
    df_wc26_played["p_home"] = proba[:, 2]
    df_wc26_played["p_draw"] = proba[:, 1]
    df_wc26_played["p_away"] = proba[:, 0]

    show_cols = ["date", "home_team", "away_team", "home_score", "away_score",
                 "actual_label", "pred_label", "correct", "p_home", "p_draw", "p_away"]
    print("\nPredictions:")
    print(df_wc26_played[show_cols].to_string())

    df_wc26_played.to_csv(OUT / "wc2026_played_preds.csv", index=False)
    print(f"\nSaved: {OUT / 'wc2026_played_preds.csv'}")

# %% Predict unplayed matches
if len(df_wc26_unplayed) > 0:
    X_future = df_wc26_unplayed[FEAT_COLS].values.astype(np.float32)
    future_preds = clf.predict(X_future)
    future_proba = clf.predict_proba(X_future)

    df_wc26_unplayed["pred"] = future_preds
    df_wc26_unplayed["pred_label"] = np.where(future_preds == 2, "H", np.where(future_preds == 1, "D", "A"))
    df_wc26_unplayed["p_home"] = future_proba[:, 2]
    df_wc26_unplayed["p_draw"] = future_proba[:, 1]
    df_wc26_unplayed["p_away"] = future_proba[:, 0]

    show_cols = ["date", "home_team", "away_team", "pred_label", "p_home", "p_draw", "p_away"]
    print("\n=== Upcoming WC 2026 predictions ===")
    print(df_wc26_unplayed[show_cols].to_string())
    df_wc26_unplayed.to_csv(OUT / "wc2026_upcoming_preds.csv", index=False)
    print(f"\nSaved: {OUT / 'wc2026_upcoming_preds.csv'}")
