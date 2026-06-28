"""Multi-output score prediction vs direct outcome classification.

Q1: Does predicting exact scores (two classifiers: home_goals, away_goals)
    then deriving outcome beat direct 3-way classification?
Q2: Does data augmentation help?

(NB tentative results; need to double check LLM implementation of my spec.)

A1: YES. Multi-output clf best on test (66.7% vs 65.3% direct) AND solves
    draws (9/20 vs 0/20). Exact score ~14%. Regression terrible (35%).
A2: Mirror augmentation (swap home/away, lossless) gives +1.4pp → 68.1%.
    All other synthetic approaches (Poisson, NegBin, RF, TabICL self-play,
    inverse feature gen) HURT because they dilute signal.
    TabICL is deterministic — 0 variance across seeds.

Final results (WC 2026 test, 72 matches):
  Q1a direct outcome:    56.2%/65.3%  draws=0/20
  Q1b multi-output clf:  50.8%/66.7%  draws=9/20, exact=14%
  Q1b + mirror:          53.1%/68.1%  draws=9/20 (best)
  Q1c GBR regression:    39.1%/34.7%  terrible
  Q2  diverse synth 31k: 48.4%/52.8%  hurts (-14pp)
"""
# pyright: reportArgumentType=false, reportReturnType=false
# %% Setup
from __future__ import annotations

import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score
from tabicl import TabICLClassifier

from tk import datadir
from tk.nbs.footie import (
    FEAT_COLS, VALID_CUT, TEST_CUT,
    compute_metrics, load_match_dataset, load_tm_features,
)

OUT = datadir / "out" / "2606_score_synth"
OUT.mkdir(parents=True, exist_ok=True)
WC = "FIFA World Cup"

ckpt = Path(hf_hub_download(
    repo_id="jingang/TabICL",
    filename="tabicl-classifier-v2-20260212.ckpt",
))

def _tabicl():
    return TabICLClassifier(
        model_path=ckpt, norm_methods=None,
        feat_shuffle_method="latin", class_shuffle_method="shift",
        outlier_threshold=4, support_many_classes=True, batch_size=8,
        kv_cache=False, allow_auto_download=False,
        checkpoint_version=ckpt.name,
        use_amp="auto", use_fa3="auto", offload_mode="auto",
        disk_offload_dir=None,
    )

# %% Load + split
t0 = _time.time()
df = load_match_dataset(load_tm_features())
df = df[df["outcome"].notna()].copy()
df["outcome"] = df["outcome"].astype(int)
df["home_score"] = df["home_score"].astype(int)
df["away_score"] = df["away_score"].astype(int)
print(f"Loaded {len(df)} matches ({_time.time()-t0:.1f}s)")

df_train = df[df["date"] < VALID_CUT]
df_valid = df[(df["date"] >= VALID_CUT) & (df["date"] < TEST_CUT)
    & (df["tournament"] == WC)]
df_test = df[(df["date"] >= TEST_CUT) & (df["tournament"] == WC)]

X_tr = df_train[FEAT_COLS].values.astype(np.float32)
X_va = df_valid[FEAT_COLS].values.astype(np.float32)
X_te = df_test[FEAT_COLS].values.astype(np.float32)
y_out_va = df_valid["outcome"].values
y_out_te = df_test["outcome"].values
y_home_tr = df_train["home_score"].clip(0, 5).values
y_away_tr = df_train["away_score"].clip(0, 5).values
y_home_va = df_valid["home_score"].clip(0, 5).values
y_away_va = df_valid["away_score"].clip(0, 5).values
y_home_te = df_test["home_score"].clip(0, 5).values
y_away_te = df_test["away_score"].clip(0, 5).values

print(f"Train: {X_tr.shape}  Valid: {X_va.shape}  Test: {X_te.shape}")
print(f"Train outcome: {np.bincount(df_train['outcome'].values)}")

results = []

def _eval(name, ph_v, pa_v, ph_t, pa_t):
    for split, ph, pa, th, ta, to in [
        ("valid", ph_v, pa_v, y_home_va, y_away_va, y_out_va),
        ("test", ph_t, pa_t, y_home_te, y_away_te, y_out_te),
    ]:
        m = compute_metrics(ph, pa, th, ta, to)
        m["method"] = name
        m["split"] = split
        results.append(m)
        print(f"  {split}: outcome={m['acc_outcome']:.3f} "
              f"exact={m['exact_score']:.3f} "
              f"draws={m['n_draws_pred']}/{m['n_draws_actual']}")

# %% Q1a: Direct 3-way outcome
print("\nQ1a: Direct outcome classification")
t0 = _time.time()
tab = _tabicl()
tab.fit(X_tr, df_train["outcome"].values)
pred_v, pred_t = tab.predict(X_va), tab.predict(X_te)
print(f"  ({_time.time()-t0:.0f}s)")
for split, y_true, y_pred in [("valid", y_out_va, pred_v),
                               ("test", y_out_te, pred_t)]:
    acc = accuracy_score(y_true, y_pred)
    results.append({"method": "Q1a_direct", "split": split,
        "acc_outcome": acc, "exact_score": np.nan,
        "mae_home": np.nan, "mae_away": np.nan,
        "n_draws_pred": int((y_pred == 1).sum()),
        "n_draws_actual": int((y_true == 1).sum())})
    print(f"  {split}: outcome={acc:.3f} "
          f"draws={(y_pred==1).sum()}/{(y_true==1).sum()}")

# %% Q1b: Multi-output classification
print("\nQ1b: Multi-output classification (home + away goals)")
t0 = _time.time()
tab_h = _tabicl()
tab_h.fit(X_tr, y_home_tr)
tab_a = _tabicl()
tab_a.fit(X_tr, y_away_tr)
ph_v, pa_v = tab_h.predict(X_va), tab_a.predict(X_va)
ph_t, pa_t = tab_h.predict(X_te), tab_a.predict(X_te)
print(f"  ({_time.time()-t0:.0f}s)")
_eval("Q1b_multiout", ph_v, pa_v, ph_t, pa_t)

print("\n  Samples (test):")
for i in range(min(10, len(ph_t))):
    r = df_test.iloc[i]
    ok = "✓" if ph_t[i] == y_home_te[i] and pa_t[i] == y_away_te[i] else ""
    print(f"    {r['home_team']:>15} vs {r['away_team']:<15} "
          f"actual={int(r['home_score'])}-{int(r['away_score'])} "
          f"pred={ph_t[i]}-{pa_t[i]} {ok}")

# %% Q1c: Regression (GBR → round)
print("\nQ1c: GBR regression → round")
t0 = _time.time()
gbr_h = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
gbr_h.fit(X_tr, df_train["home_score"].values.astype(float))
gbr_a = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
gbr_a.fit(X_tr, df_train["away_score"].values.astype(float))
ph_v = np.clip(np.round(gbr_h.predict(X_va)), 0, 9).astype(int)
pa_v = np.clip(np.round(gbr_a.predict(X_va)), 0, 9).astype(int)
ph_t = np.clip(np.round(gbr_h.predict(X_te)), 0, 9).astype(int)
pa_t = np.clip(np.round(gbr_a.predict(X_te)), 0, 9).astype(int)
print(f"  ({_time.time()-t0:.0f}s)")
_eval("Q1c_regression", ph_v, pa_v, ph_t, pa_t)

# %% Q2: Diverse counterfactual augmentation
print("\nQ2: Real + diverse counterfactuals")
synth_path = datadir / "out" / "2606_synth_tournaments" / "matches.parquet"
if not synth_path.exists():
    print(f"  SKIP: run 2606_simulate_synth.py first")
else:
    df_synth = pd.read_parquet(synth_path)
    X_synth = df_synth[FEAT_COLS].values.astype(np.float32)
    y_h_synth = df_synth["home_score"].clip(0, 5).values.astype(np.int64)
    y_a_synth = df_synth["away_score"].clip(0, 5).values.astype(np.int64)

    X_aug = np.vstack([X_tr, X_synth]).astype(np.float32)
    y_h_aug = np.concatenate([y_home_tr, y_h_synth])
    y_a_aug = np.concatenate([y_away_tr, y_a_synth])
    print(f"  Real={X_tr.shape[0]} + Synth={len(X_synth)} "
          f"= {X_aug.shape[0]}")

    t0 = _time.time()
    tab_h2 = _tabicl()
    tab_h2.fit(X_aug, y_h_aug)
    tab_a2 = _tabicl()
    tab_a2.fit(X_aug, y_a_aug)
    ph_v = tab_h2.predict(X_va)
    pa_v = tab_a2.predict(X_va)
    ph_t = tab_h2.predict(X_te)
    pa_t = tab_a2.predict(X_te)
    print(f"  ({_time.time()-t0:.0f}s)")
    _eval("Q2_diverse_synth", ph_v, pa_v, ph_t, pa_t)

# %% Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
df_r = pd.DataFrame(results)
cols = ["method", "split", "acc_outcome", "exact_score",
    "n_draws_pred", "n_draws_actual"]
print(df_r[cols].to_string(index=False, float_format="%.3f"))
df_r.to_csv(OUT / "results.csv", index=False)
print(f"\nSaved: {OUT}")
