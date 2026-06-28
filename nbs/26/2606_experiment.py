"""Baseline comparison: TabICL vs sklearn on 3-way match outcome prediction.

Result: TabICL (53%) ≈ RF (52%) on valid, both ~65% on test. No model predicts
draws (F1=0 for all). TM roster features carry the signal; model choice barely
matters for direct outcome classification. See 2606_score_synth.py for
multi-output formulation that solves the draw problem.

Protocol:
- Train: 4326 post-2006 international matches (both teams have TM features)
- Valid: 128 FIFA World Cup 2018+2022 matches
- Test:  72 FIFA World Cup 2026 matches
- Features: 28 (importance, neutral, 13 home roster, 13 away roster)
- Baselines: majority (41-46%), logistic regression (50-60%), RF (52-65%)
- TabICL: 53-65%
"""
# pyright: reportArgumentType=false, reportReturnType=false
# %% Setup
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from tk import datadir
from tk.nbs.footie import FEAT_COLS, load_match_dataset, load_tm_features

OUT = datadir / "out" / "2606_experiment"
OUT.mkdir(parents=True, exist_ok=True)

# %% Load data
tm_db = load_tm_features()
df = load_match_dataset(tm_db)
df_played = df[df["outcome"].notna()].copy()
df_played["outcome"] = df_played["outcome"].astype(int)
print(f"Total matched (played): {len(df_played)}")
print(f"Outcome dist: {np.bincount(df_played['outcome'].values)}")
print(f"  (0=away, 1=draw, 2=home)")

# %% Splits
VALID_CUTOFF = "2018-01-01"
TEST_CUTOFF = "2026-06-01"

df_train = df_played[df_played["date"] < VALID_CUTOFF]
df_valid = df_played[
    (df_played["date"] >= VALID_CUTOFF)
    & (df_played["date"] < TEST_CUTOFF)
    & (df_played["tournament"] == "FIFA World Cup")
]
df_test = df_played[
    (df_played["date"] >= TEST_CUTOFF)
    & (df_played["tournament"] == "FIFA World Cup")
]

X_train = df_train[FEAT_COLS].values.astype(np.float32)
y_train = df_train["outcome"].values
X_valid = df_valid[FEAT_COLS].values.astype(np.float32)
y_valid = df_valid["outcome"].values
X_test = df_test[FEAT_COLS].values.astype(np.float32)
y_test = df_test["outcome"].values

print(f"\nTrain: {X_train.shape} | outcome: {np.bincount(y_train)}")
print(f"Valid (WC18+22): {X_valid.shape} | outcome: {np.bincount(y_valid)}")
print(f"Test  (WC26):    {X_test.shape} | outcome: {np.bincount(y_test)}")

# %% Baseline 1: majority class
majority = np.bincount(y_train).argmax()
pred_majority_v = np.full(len(y_valid), majority)
pred_majority_t = np.full(len(y_test), majority)

# %% Baseline 2: logistic regression
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_valid_s = scaler.transform(X_valid)
X_test_s = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, C=1.0)
lr.fit(X_train_s, y_train)
pred_lr_v = lr.predict(X_valid_s)
pred_lr_t = lr.predict(X_test_s)

# %% Baseline 3: random forest
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
pred_rf_v = rf.predict(X_valid)
pred_rf_t = rf.predict(X_test)

# %% TabICL
from huggingface_hub import hf_hub_download  # noqa: E402
from tabicl import TabICLClassifier  # noqa: E402

ckpt = Path(hf_hub_download(
    repo_id="jingang/TabICL",
    filename="tabicl-classifier-v2-20260212.ckpt",
))
clf = TabICLClassifier(
    model_path=ckpt,
    norm_methods=None,
    feat_shuffle_method="latin",
    class_shuffle_method="shift",
    outlier_threshold=4,
    support_many_classes=True,
    batch_size=8,
    kv_cache=False,
    allow_auto_download=False,
    checkpoint_version=ckpt.name,
    use_amp="auto",
    use_fa3="auto",
    offload_mode="auto",
    disk_offload_dir=None,
)
clf.fit(X_train, y_train)
pred_tab_v = clf.predict(X_valid)
pred_tab_t = clf.predict(X_test)

# %% Results table
LABELS = ["away_win", "draw", "home_win"]

def eval_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=LABELS,
        output_dict=True, zero_division=0,
    )
    return {
        "model": name,
        "accuracy": acc,
        "f1_away": report["away_win"]["f1-score"],
        "f1_draw": report["draw"]["f1-score"],
        "f1_home": report["home_win"]["f1-score"],
        "f1_macro": report["macro avg"]["f1-score"],
    }

results = []
for split_name, y_true, preds in [
    ("valid", y_valid, [
        ("majority", pred_majority_v),
        ("logreg", pred_lr_v),
        ("random_forest", pred_rf_v),
        ("tabicl", pred_tab_v),
    ]),
    ("test", y_test, [
        ("majority", pred_majority_t),
        ("logreg", pred_lr_t),
        ("random_forest", pred_rf_t),
        ("tabicl", pred_tab_t),
    ]),
]:
    for model_name, y_pred in preds:
        row = eval_model(model_name, y_true, y_pred)
        row["split"] = split_name
        results.append(row)

df_results = pd.DataFrame(results)
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
for split in ["valid", "test"]:
    print(f"\n--- {split.upper()} ---")
    sub = df_results[df_results["split"] == split]
    print(sub[["model", "accuracy", "f1_away", "f1_draw", "f1_home", "f1_macro"]].to_string(index=False))

df_results.to_csv(OUT / "results.csv", index=False)

# %% Per-example predictions (observability)
df_valid_out = df_valid[["date", "home_team", "away_team",
    "home_score", "away_score", "outcome"]].copy()
df_valid_out["pred_majority"] = pred_majority_v
df_valid_out["pred_logreg"] = pred_lr_v
df_valid_out["pred_rf"] = pred_rf_v
df_valid_out["pred_tabicl"] = pred_tab_v
df_valid_out.to_csv(OUT / "per_example_valid.csv", index=False)

df_test_out = df_test[["date", "home_team", "away_team",
    "home_score", "away_score", "outcome"]].copy()
df_test_out["pred_majority"] = pred_majority_t
df_test_out["pred_logreg"] = pred_lr_t
df_test_out["pred_rf"] = pred_rf_t
df_test_out["pred_tabicl"] = pred_tab_t
df_test_out.to_csv(OUT / "per_example_test.csv", index=False)

print(f"\nSaved to {OUT}")
