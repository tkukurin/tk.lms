"""Ablation: can TabICL learn team strength from match history via ICL?

(NB tentative results; need to double check LLM implementation of my spec)

A1: Seems NO. Pure ICL with team IDs + date (no roster features) performs at or below majority class (36-44%).

A2: TabICL > RF by 3-6pp when given roster features.
Date helps (+3pp over no-date from 2606_experiment.py).

Setup:
1. ids+date: team IDs only. n=11.5k. 44%/36% (worse than majority)
2. tm+date: TM roster feats, n=4.3k → 56%/65%
3. all: IDs + TM feats, n=4.3k → 56%/65%
"""
# pyright: reportArgumentType=false, reportReturnType=false
# %% Setup
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from tk import datadir
from tk.nbs.footie import (
    HOME_COLS, AWAY_COLS, MATCHES_CSV,
    load_match_dataset, load_tm_features,
)

OUT = datadir / "out" / "2606_icl_ablation"
OUT.mkdir(parents=True, exist_ok=True)

# %% Load full match data (all 49k, not just TM-covered)
df_all = pd.read_csv(MATCHES_CSV)
df_all = df_all[df_all["date"] >= "2006-01-01"].copy()
df_all = df_all.dropna(subset=["home_score", "away_score"])
df_all["home_score"] = df_all["home_score"].astype(int)
df_all["away_score"] = df_all["away_score"].astype(int)

all_teams = sorted(set(df_all["home_team"]) | set(df_all["away_team"]))
team2id = {t: i for i, t in enumerate(all_teams)}
df_all["home_id"] = df_all["home_team"].map(team2id)
df_all["away_id"] = df_all["away_team"].map(team2id)

df_all["date_ordinal"] = (
    pd.to_datetime(df_all["date"]) - pd.Timestamp("2006-01-01")
).dt.days
df_all["neutral"] = df_all["neutral"].astype(int)

from tk.nbs.footie import TOURNAMENT_IMPORTANCE
def _imp(t):
    for k, v in TOURNAMENT_IMPORTANCE.items():
        if k.lower() in t.lower():
            return v
    return 2
df_all["importance"] = df_all["tournament"].apply(_imp)
df_all["outcome"] = np.where(
    df_all["home_score"] > df_all["away_score"], 2,
    np.where(df_all["home_score"] == df_all["away_score"], 1, 0),
)

print(f"All matches 2006+: {len(df_all)}")

# %% Also load TM-enriched version (for conditions 2 and 3)
tm_db = load_tm_features()
df_tm = load_match_dataset(tm_db)
df_tm = df_tm[df_tm["outcome"].notna()].copy()
df_tm["outcome"] = df_tm["outcome"].astype(int)
df_tm["home_id"] = df_tm["home_team"].map(team2id)
df_tm["away_id"] = df_tm["away_team"].map(team2id)
df_tm["date_ordinal"] = (
    pd.to_datetime(df_tm["date"]) - pd.Timestamp("2006-01-01")
).dt.days
print(f"TM-covered matches: {len(df_tm)}")

# %% Define feature sets
FEATS_IDS_DATE = ["home_id", "away_id", "date_ordinal", "importance", "neutral"]
FEATS_TM_DATE = ["date_ordinal", "importance", "neutral"] + HOME_COLS + AWAY_COLS
FEATS_ALL = ["home_id", "away_id", "date_ordinal", "importance", "neutral"] + HOME_COLS + AWAY_COLS

# %% Splits
VALID_CUT = "2018-01-01"
TEST_CUT = "2026-06-01"
WC = "FIFA World Cup"

def split(df, feats):
    tr = df[df["date"] < VALID_CUT]
    va = df[(df["date"] >= VALID_CUT) & (df["date"] < TEST_CUT) & (df["tournament"] == WC)]
    te = df[(df["date"] >= TEST_CUT) & (df["tournament"] == WC)]
    X_tr, y_tr = tr[feats].values.astype(np.float32), tr["outcome"].values
    X_va, y_va = va[feats].values.astype(np.float32), va["outcome"].values
    X_te, y_te = te[feats].values.astype(np.float32), te["outcome"].values
    return X_tr, y_tr, X_va, y_va, X_te, y_te

# Condition 1: all matches, IDs + date (pure ICL test)
X1_tr, y1_tr, X1_va, y1_va, X1_te, y1_te = split(df_all, FEATS_IDS_DATE)
# Condition 2: TM matches, roster + date
X2_tr, y2_tr, X2_va, y2_va, X2_te, y2_te = split(df_tm, FEATS_TM_DATE)
# Condition 3: TM matches, everything
X3_tr, y3_tr, X3_va, y3_va, X3_te, y3_te = split(df_tm, FEATS_ALL)

print(f"\nCondition 1 (ids+date):  train={X1_tr.shape}, valid={X1_va.shape}, test={X1_te.shape}")
print(f"Condition 2 (tm+date):   train={X2_tr.shape}, valid={X2_va.shape}, test={X2_te.shape}")
print(f"Condition 3 (all):       train={X3_tr.shape}, valid={X3_va.shape}, test={X3_te.shape}")

# %% Run all models
from huggingface_hub import hf_hub_download  # noqa: E402
from tabicl import TabICLClassifier  # noqa: E402

ckpt = Path(hf_hub_download(
    repo_id="jingang/TabICL",
    filename="tabicl-classifier-v2-20260212.ckpt",
))

def make_tabicl():
    return TabICLClassifier(
        model_path=ckpt, norm_methods=None,
        feat_shuffle_method="latin", class_shuffle_method="shift",
        outlier_threshold=4, support_many_classes=True, batch_size=8,
        kv_cache=False, allow_auto_download=False,
        checkpoint_version=ckpt.name,
        use_amp="auto", use_fa3="auto", offload_mode="auto",
        disk_offload_dir=None,
    )

results = []

def run_condition(name, X_tr, y_tr, X_va, y_va, X_te, y_te):
    # Majority baseline (same training set)
    maj = np.bincount(y_tr).argmax()
    results.append({"cond": name, "model": "majority", "split": "valid",
        "acc": accuracy_score(y_va, np.full(len(y_va), maj))})
    results.append({"cond": name, "model": "majority", "split": "test",
        "acc": accuracy_score(y_te, np.full(len(y_te), maj))})

    # Random forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_tr, y_tr)
    results.append({"cond": name, "model": "rf", "split": "valid",
        "acc": accuracy_score(y_va, rf.predict(X_va))})
    results.append({"cond": name, "model": "rf", "split": "test",
        "acc": accuracy_score(y_te, rf.predict(X_te))})

    # TabICL
    tab = make_tabicl()
    tab.fit(X_tr, y_tr)
    results.append({"cond": name, "model": "tabicl", "split": "valid",
        "acc": accuracy_score(y_va, tab.predict(X_va))})
    results.append({"cond": name, "model": "tabicl", "split": "test",
        "acc": accuracy_score(y_te, tab.predict(X_te))})

print("\n--- Running condition 1: ids+date (pure ICL) ---")
run_condition("ids+date", X1_tr, y1_tr, X1_va, y1_va, X1_te, y1_te)
print("--- Running condition 2: tm+date ---")
run_condition("tm+date", X2_tr, y2_tr, X2_va, y2_va, X2_te, y2_te)
print("--- Running condition 3: all ---")
run_condition("all", X3_tr, y3_tr, X3_va, y3_va, X3_te, y3_te)

# %% Results
df_r = pd.DataFrame(results)
pivot = df_r.pivot_table(index=["cond", "model"], columns="split", values="acc")
pivot = pivot[["valid", "test"]].reset_index()
print("\n" + "=" * 60)
print("ABLATION RESULTS (accuracy)")
print("=" * 60)
print(pivot.to_string(index=False, float_format="%.3f"))
df_r.to_csv(OUT / "ablation_results.csv", index=False)
print(f"\nSaved: {OUT / 'ablation_results.csv'}")
