"""Multi-output score prediction vs direct outcome classification.

Q1: Does predicting exact scores (two classifiers: home_goals, away_goals)
    then deriving outcome beat direct 3-way classification?
Q2: Does data augmentation help?
Q3: Does fine-tuning on synthetic data help?

(NB tentative results; need to double check LLM implementation of my spec.)

A1: YES. Multi-output clf best on test (66.7% vs 65.3% direct) AND solves
    draws (9/20 vs 0/20). Exact score ~14%. Regression terrible (35%).
A2: Mirror augmentation (swap home/away, lossless) gives +1.4pp -> 68.1%.
    All other synthetic approaches (Poisson, NegBin, RF, TabICL self-play,
    inverse feature gen) HURT because they dilute signal.
    TabICL is deterministic -- 0 variance across seeds.
A3: TBD. Hypothesis: two-stage training (synth -> real fine-tune) or
    weighted mixtures preserve signal better than naive concatenation.

Final results (WC 2026 test, 72 matches):
  Q1a direct outcome:    56.2%/65.3%  draws=0/20
  Q1b multi-output clf:  50.8%/66.7%  draws=9/20, exact=14%
  Q1b + mirror:          53.1%/68.1%  draws=9/20 (best)
  Q1c GBR regression:    39.1%/34.7%  terrible
  Q2  diverse synth 31k: 48.4%/52.8%  hurts (-14pp)
  Q3a synth->GBR finetune:   TBD
  Q3b weighted mix (5:1):    TBD
  Q3c synth-only baseline:   TBD
  Q3d mirror+inverse synth:  TBD
"""
# pyright: reportArgumentType=false, reportReturnType=false
# %% Setup
from __future__ import annotations

import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score
from tabicl import TabICLClassifier

from tk import datadir
from tk.nbs.footie import (
    FEAT_COLS, HOME_COLS, VALID_CUT, TEST_CUT,
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

# %% Q1b + mirror: lossless augmentation (swap home/away)
print("\nQ1b+mirror: Multi-output + mirror augmentation")
t0 = _time.time()
nc = len(HOME_COLS)
X_mirror = X_tr.copy()
X_mirror[:, 3:3+nc] = X_tr[:, 3+nc:]
X_mirror[:, 3+nc:] = X_tr[:, 3:3+nc]
X_aug_mirror = np.vstack([X_tr, X_mirror]).astype(np.float32)
y_h_mirror = np.concatenate([y_home_tr, y_away_tr])
y_a_mirror = np.concatenate([y_away_tr, y_home_tr])
print(f"  Real={X_tr.shape[0]} + Mirror={X_mirror.shape[0]} = {X_aug_mirror.shape[0]}")

tab_hm = _tabicl()
tab_hm.fit(X_aug_mirror, y_h_mirror)
tab_am = _tabicl()
tab_am.fit(X_aug_mirror, y_a_mirror)
ph_v, pa_v = tab_hm.predict(X_va), tab_am.predict(X_va)
ph_t, pa_t = tab_hm.predict(X_te), tab_am.predict(X_te)
print(f"  ({_time.time()-t0:.0f}s)")
_eval("Q1b_mirror", ph_v, pa_v, ph_t, pa_t)

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

# %% Q3: Fine-tuning on synthetic data
print("\nQ3: Fine-tuning on synthetic data")
print("Hypothesis: two-stage training (synth pre-train -> real fine-tune)")
print("preserves signal better than naive concatenation.")

if not synth_path.exists():
    print("  SKIP: run 2606_simulate_synth.py first")
else:
    # Reload synth if not already loaded
    if "X_synth" not in dir():
        df_synth = pd.read_parquet(synth_path)
        X_synth = df_synth[FEAT_COLS].values.astype(np.float32)
        y_h_synth = df_synth["home_score"].clip(0, 5).values.astype(np.int64)
        y_a_synth = df_synth["away_score"].clip(0, 5).values.astype(np.int64)

    # --- Q3a: GBC pre-train on synth -> warm-start fine-tune on real ---
    print("\n  Q3a: GBC synth pre-train -> real fine-tune")
    t0 = _time.time()
    N_PRETRAIN = 300
    N_FINETUNE = 200

    # Stage 1: pre-train on synthetic data
    gbc_h_pre = GradientBoostingClassifier(
        n_estimators=N_PRETRAIN, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42, warm_start=True,
    )
    gbc_h_pre.fit(X_synth, y_h_synth)
    gbc_a_pre = GradientBoostingClassifier(
        n_estimators=N_PRETRAIN, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42, warm_start=True,
    )
    gbc_a_pre.fit(X_synth, y_a_synth)

    # Stage 2: fine-tune on real data (add more estimators fitted to real)
    gbc_h_pre.n_estimators = N_PRETRAIN + N_FINETUNE
    gbc_h_pre.fit(X_tr, y_home_tr)
    gbc_a_pre.n_estimators = N_PRETRAIN + N_FINETUNE
    gbc_a_pre.fit(X_tr, y_away_tr)

    ph_v = gbc_h_pre.predict(X_va)
    pa_v = gbc_a_pre.predict(X_va)
    ph_t = gbc_h_pre.predict(X_te)
    pa_t = gbc_a_pre.predict(X_te)
    print(f"    ({_time.time()-t0:.0f}s)")
    _eval("Q3a_gbc_finetune", ph_v, pa_v, ph_t, pa_t)

    # --- Q3a_baseline: GBC real-only (same budget, no synth pre-train) ---
    print("\n  Q3a_baseline: GBC real-only (same total estimators)")
    t0 = _time.time()
    gbc_h_base = GradientBoostingClassifier(
        n_estimators=N_PRETRAIN + N_FINETUNE, max_depth=4,
        learning_rate=0.05, subsample=0.8, random_state=42,
    )
    gbc_h_base.fit(X_tr, y_home_tr)
    gbc_a_base = GradientBoostingClassifier(
        n_estimators=N_PRETRAIN + N_FINETUNE, max_depth=4,
        learning_rate=0.05, subsample=0.8, random_state=42,
    )
    gbc_a_base.fit(X_tr, y_away_tr)
    ph_v = gbc_h_base.predict(X_va)
    pa_v = gbc_a_base.predict(X_va)
    ph_t = gbc_h_base.predict(X_te)
    pa_t = gbc_a_base.predict(X_te)
    print(f"    ({_time.time()-t0:.0f}s)")
    _eval("Q3a_gbc_realonly", ph_v, pa_v, ph_t, pa_t)

    # --- Q3b: Weighted mix (repeat real N times to upweight vs synth) ---
    print("\n  Q3b: TabICL weighted mix (real repeated 5x + synth 1x)")
    t0 = _time.time()
    REAL_WEIGHT = 5
    X_weighted = np.vstack(
        [X_tr] * REAL_WEIGHT + [X_synth]
    ).astype(np.float32)
    y_h_weighted = np.concatenate(
        [y_home_tr] * REAL_WEIGHT + [y_h_synth]
    )
    y_a_weighted = np.concatenate(
        [y_away_tr] * REAL_WEIGHT + [y_a_synth]
    )
    print(f"    Real x{REAL_WEIGHT}={X_tr.shape[0]*REAL_WEIGHT} + "
          f"Synth={len(X_synth)} = {len(X_weighted)}")

    tab_h3 = _tabicl()
    tab_h3.fit(X_weighted, y_h_weighted)
    tab_a3 = _tabicl()
    tab_a3.fit(X_weighted, y_a_weighted)
    ph_v = tab_h3.predict(X_va)
    pa_v = tab_a3.predict(X_va)
    ph_t = tab_h3.predict(X_te)
    pa_t = tab_a3.predict(X_te)
    print(f"    ({_time.time()-t0:.0f}s)")
    _eval("Q3b_weighted_5to1", ph_v, pa_v, ph_t, pa_t)

    # --- Q3c: Synth-only baseline (signal check) ---
    print("\n  Q3c: Synth-only (signal quality check)")
    t0 = _time.time()
    tab_h4 = _tabicl()
    tab_h4.fit(X_synth, y_h_synth)
    tab_a4 = _tabicl()
    tab_a4.fit(X_synth, y_a_synth)
    ph_v = tab_h4.predict(X_va)
    pa_v = tab_a4.predict(X_va)
    ph_t = tab_h4.predict(X_te)
    pa_t = tab_a4.predict(X_te)
    print(f"    ({_time.time()-t0:.0f}s)")
    _eval("Q3c_synth_only", ph_v, pa_v, ph_t, pa_t)

    # --- Q3d: Mirror + inverse synth (best aug + highest-signal synth) ---
    print("\n  Q3d: Mirror + filtered synth (inverse-only, higher signal)")
    t0 = _time.time()
    df_inv_only = df_synth[df_synth["facet"] == "inverse"]
    if len(df_inv_only) > 0:
        X_inv = df_inv_only[FEAT_COLS].values.astype(np.float32)
        y_h_inv = df_inv_only["home_score"].clip(0, 5).values.astype(np.int64)
        y_a_inv = df_inv_only["away_score"].clip(0, 5).values.astype(np.int64)
        X_best = np.vstack([X_aug_mirror, X_inv]).astype(np.float32)
        y_h_best = np.concatenate([y_h_mirror, y_h_inv])
        y_a_best = np.concatenate([y_a_mirror, y_a_inv])
        print(f"    Mirror={X_aug_mirror.shape[0]} + Inverse={len(X_inv)} "
              f"= {len(X_best)}")

        tab_h5 = _tabicl()
        tab_h5.fit(X_best, y_h_best)
        tab_a5 = _tabicl()
        tab_a5.fit(X_best, y_a_best)
        ph_v = tab_h5.predict(X_va)
        pa_v = tab_a5.predict(X_va)
        ph_t = tab_h5.predict(X_te)
        pa_t = tab_a5.predict(X_te)
        print(f"    ({_time.time()-t0:.0f}s)")
        _eval("Q3d_mirror_inv", ph_v, pa_v, ph_t, pa_t)
    else:
        print("    SKIP: no inverse facet in synth data")

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
