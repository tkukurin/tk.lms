"""Multi-output score prediction vs direct outcome classification.

Q1: Does predicting exact scores (two classifiers: home_goals, away_goals)
    then deriving outcome beat direct 3-way classification?
Q2: Does augmenting with Poisson-simulated matches (GLM-fitted λ, random
    team-year pairings) help TabICL?

(NB tentative results; need to double check LLM implementation of my spec. I
also suspect that including results in the input features better would achieve
similar outcomes. I also believe improving synthetic augmentation procedure
would help — current approach pairs random snapshots which doesn't reflect
real tournament scheduling structure.)

A1: YES on test. Multi-output clf gets 66.7% (vs 65.3% direct) AND solves
    draws: 9/20 vs 0/20. Exact score ~14%. Regression terrible (35%).
A2: NO. Poisson sim (10k synthetic, well-calibrated goals) still hurts
    (48%/56% vs 51%/67%). Over-predicts draws (27/20, 48/28). Hypothesis:
    random team pairings don't match tournament structure — real WC groups
    pair strong+weak teams, knockouts pair similar teams.

Results:
  Q1a direct outcome:    valid=56.2% test=65.3%  draws=0/20
  Q1b multi-output clf:  valid=50.8% test=66.7%  draws=9/20, 14% exact scores
  Q1c GBR regression:    valid=39.1% test=34.7%  terrible
  Q2  poisson sim 10k:   valid=48.4% test=55.6%  hurts (over-predicts draws)
"""
# pyright: reportArgumentType=false, reportReturnType=false
# %% Setup
from __future__ import annotations

import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error

from tk import datadir
from tk.nbs.footie import HOME_COLS, AWAY_COLS, load_match_dataset, load_tm_features

OUT = datadir / "out" / "2606_score_synth"
OUT.mkdir(parents=True, exist_ok=True)

def elapsed(t0):
    return f"{_time.time() - t0:.1f}s"

# %% Load
print("Loading data...")
t0 = _time.time()
tm_db = load_tm_features()
df = load_match_dataset(tm_db)
df = df[df["outcome"].notna()].copy()
df["outcome"] = df["outcome"].astype(int)
df["home_score"] = df["home_score"].astype(int)
df["away_score"] = df["away_score"].astype(int)
df["date_ordinal"] = (
    pd.to_datetime(df["date"]) - pd.Timestamp("2006-01-01")
).dt.days
print(f"  {elapsed(t0)}, shape={df.shape}")

FEATS = ["date_ordinal", "importance", "neutral"] + HOME_COLS + AWAY_COLS
VALID_CUT, TEST_CUT, WC = "2018-01-01", "2026-06-01", "FIFA World Cup"

df_train = df[df["date"] < VALID_CUT].copy()
df_valid = df[
    (df["date"] >= VALID_CUT) & (df["date"] < TEST_CUT)
    & (df["tournament"] == WC)
].copy()
df_test = df[(df["date"] >= TEST_CUT) & (df["tournament"] == WC)].copy()

X_tr = df_train[FEATS].values.astype(np.float32)
X_va = df_valid[FEATS].values.astype(np.float32)
X_te = df_test[FEATS].values.astype(np.float32)

y_outcome_tr = df_train["outcome"].values
y_outcome_va = df_valid["outcome"].values
y_outcome_te = df_test["outcome"].values

# Goal targets (clipped 0-5 for classification)
y_home_tr = df_train["home_score"].clip(0, 5).values
y_away_tr = df_train["away_score"].clip(0, 5).values
y_home_va = df_valid["home_score"].clip(0, 5).values
y_away_va = df_valid["away_score"].clip(0, 5).values
y_home_te = df_test["home_score"].clip(0, 5).values
y_away_te = df_test["away_score"].clip(0, 5).values

# Continuous targets for regression
y_home_tr_cont = df_train["home_score"].values.astype(np.float32)
y_away_tr_cont = df_train["away_score"].values.astype(np.float32)
y_home_va_cont = df_valid["home_score"].values.astype(np.float32)
y_away_va_cont = df_valid["away_score"].values.astype(np.float32)
y_home_te_cont = df_test["home_score"].values.astype(np.float32)
y_away_te_cont = df_test["away_score"].values.astype(np.float32)

print(f"\n{'='*60}")
print("DATA SUMMARY")
print(f"{'='*60}")
print(f"Features: {len(FEATS)}")
print(f"Train: {X_tr.shape}  outcome={np.bincount(y_outcome_tr)}")
print(f"  home_goals dist: {np.bincount(y_home_tr)}")
print(f"  away_goals dist: {np.bincount(y_away_tr)}")
print(f"Valid: {X_va.shape}  outcome={np.bincount(y_outcome_va)}")
print(f"Test:  {X_te.shape}  outcome={np.bincount(y_outcome_te)}")

# %% TabICL setup
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

# %% Metrics helper
def compute_metrics(
    name, split,
    pred_home, pred_away,
    true_home, true_away,
    true_outcome,
):
    pred_outcome = np.where(
        pred_home > pred_away, 2,
        np.where(pred_home == pred_away, 1, 0),
    )
    acc_outcome = accuracy_score(true_outcome, pred_outcome)
    exact_score = ((pred_home == true_home) & (pred_away == true_away)).mean()
    mae_home = mean_absolute_error(true_home, pred_home)
    mae_away = mean_absolute_error(true_away, pred_away)
    n_draws_pred = int((pred_outcome == 1).sum())
    n_draws_actual = int((true_outcome == 1).sum())
    return {
        "method": name, "split": split,
        "acc_outcome": acc_outcome,
        "exact_score": exact_score,
        "mae_home": mae_home,
        "mae_away": mae_away,
        "n_draws_pred": n_draws_pred,
        "n_draws_actual": n_draws_actual,
    }

results = []

# %% Q1a: Direct 3-way outcome (baseline)
print(f"\n{'='*60}")
print("Q1a: Direct 3-way outcome classification")
print(f"{'='*60}")
t0 = _time.time()
tab_out = make_tabicl()
tab_out.fit(X_tr, y_outcome_tr)
pred_out_v = tab_out.predict(X_va)
pred_out_t = tab_out.predict(X_te)
print(f"  time: {elapsed(t0)}")

# For Q1a we don't have score predictions, just outcome
for split, y_true, y_pred in [("valid", y_outcome_va, pred_out_v),
                               ("test", y_outcome_te, pred_out_t)]:
    acc = accuracy_score(y_true, y_pred)
    nd_pred = int((y_pred == 1).sum())
    nd_actual = int((y_true == 1).sum())
    results.append({
        "method": "Q1a_direct_outcome", "split": split,
        "acc_outcome": acc, "exact_score": np.nan,
        "mae_home": np.nan, "mae_away": np.nan,
        "n_draws_pred": nd_pred, "n_draws_actual": nd_actual,
    })
    print(f"  {split}: acc_outcome={acc:.3f}  draws={nd_pred}/{nd_actual}")

# %% Q1b: Multi-output classification (two separate TabICL classifiers)
print(f"\n{'='*60}")
print("Q1b: Multi-output classification (home_goals + away_goals, 0-5 each)")
print(f"{'='*60}")
t0 = _time.time()

print("  Fitting home_goals classifier...")
tab_home = make_tabicl()
tab_home.fit(X_tr, y_home_tr)
pred_home_v = tab_home.predict(X_va)
pred_home_t = tab_home.predict(X_te)
print(f"    done ({elapsed(t0)})")

t1 = _time.time()
print("  Fitting away_goals classifier...")
tab_away = make_tabicl()
tab_away.fit(X_tr, y_away_tr)
pred_away_v = tab_away.predict(X_va)
pred_away_t = tab_away.predict(X_te)
print(f"    done ({elapsed(t1)})")
print(f"  total time: {elapsed(t0)}")

for split, ph, pa, th, ta, to in [
    ("valid", pred_home_v, pred_away_v, y_home_va, y_away_va, y_outcome_va),
    ("test", pred_home_t, pred_away_t, y_home_te, y_away_te, y_outcome_te),
]:
    r = compute_metrics("Q1b_multioutput_clf", split, ph, pa, th, ta, to)
    results.append(r)
    print(f"  {split}: acc_outcome={r['acc_outcome']:.3f}  "
          f"exact_score={r['exact_score']:.3f}  "
          f"mae_h={r['mae_home']:.2f} mae_a={r['mae_away']:.2f}  "
          f"draws={r['n_draws_pred']}/{r['n_draws_actual']}")

# Show predictions
print("\n  Sample predictions (test, first 15):")
for i in range(min(15, len(pred_home_t))):
    row = df_test.iloc[i]
    print(f"    {row['home_team']:>15} vs {row['away_team']:<15} "
          f"actual={int(row['home_score'])}-{int(row['away_score'])} "
          f"pred={pred_home_t[i]}-{pred_away_t[i]} "
          f"{'✓' if pred_home_t[i]==y_home_te[i] and pred_away_t[i]==y_away_te[i] else ''}")

# %% Q1c: TabICL regression (predict continuous goals, round)
print(f"\n{'='*60}")
print("Q1c: Regression (continuous goals → round → derive outcome)")
print(f"{'='*60}")

# TabICL is a classifier. For regression, we use sklearn.
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402

t0 = _time.time()
print("  Fitting home_goals regressor (GBR)...")
gbr_home = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, random_state=42,
)
gbr_home.fit(X_tr, y_home_tr_cont)
pred_home_reg_v = np.clip(np.round(gbr_home.predict(X_va)), 0, 9).astype(int)
pred_home_reg_t = np.clip(np.round(gbr_home.predict(X_te)), 0, 9).astype(int)

print("  Fitting away_goals regressor (GBR)...")
gbr_away = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, random_state=42,
)
gbr_away.fit(X_tr, y_away_tr_cont)
pred_away_reg_v = np.clip(np.round(gbr_away.predict(X_va)), 0, 9).astype(int)
pred_away_reg_t = np.clip(np.round(gbr_away.predict(X_te)), 0, 9).astype(int)
print(f"  total time: {elapsed(t0)}")

for split, ph, pa, th, ta, to in [
    ("valid", pred_home_reg_v, pred_away_reg_v,
     y_home_va, y_away_va, y_outcome_va),
    ("test", pred_home_reg_t, pred_away_reg_t,
     y_home_te, y_away_te, y_outcome_te),
]:
    r = compute_metrics("Q1c_regression", split, ph, pa, th, ta, to)
    results.append(r)
    print(f"  {split}: acc_outcome={r['acc_outcome']:.3f}  "
          f"exact_score={r['exact_score']:.3f}  "
          f"mae_h={r['mae_home']:.2f} mae_a={r['mae_away']:.2f}  "
          f"draws={r['n_draws_pred']}/{r['n_draws_actual']}")

# %% Q2: Synthetic match simulation (Poisson score generation)
print(f"\n{'='*60}")
print("Q2: Simulated matches (Poisson goals from fitted GLM)")
print(f"{'='*60}")

from sklearn.linear_model import PoissonRegressor  # noqa: E402

# Fit Poisson GLM: features → expected goals (home and away separately)
# Use feature differentials + absolute features as predictors
feat_home_for_glm = X_tr  # same features that TabICL sees
feat_away_for_glm = X_tr.copy()
# Swap home/away columns to predict away goals
feat_away_for_glm[:, 3:3+len(HOME_COLS)] = X_tr[:, 3+len(HOME_COLS):]
feat_away_for_glm[:, 3+len(HOME_COLS):] = X_tr[:, 3:3+len(HOME_COLS)]

print("  Fitting Poisson GLM for home goals...")
glm_home = PoissonRegressor(alpha=0.01, max_iter=500)
glm_home.fit(X_tr, y_home_tr_cont)
print("  Fitting Poisson GLM for away goals...")
glm_away = PoissonRegressor(alpha=0.01, max_iter=500)
glm_away.fit(feat_away_for_glm, y_away_tr_cont)

# Validate GLM predictions on training data
lam_home_check = glm_home.predict(X_tr)
lam_away_check = glm_away.predict(feat_away_for_glm)
print(f"  GLM mean predicted home goals: {lam_home_check.mean():.2f} "
      f"(actual: {y_home_tr_cont.mean():.2f})")
print(f"  GLM mean predicted away goals: {lam_away_check.mean():.2f} "
      f"(actual: {y_away_tr_cont.mean():.2f})")

# Generate synthetic matches by pairing random team-year snapshots
rng = np.random.default_rng(42)

# Collect all available (team_name, year, feature_vec) from TM database
all_snapshots = []
for team, years_dict in tm_db.items():
    for year, vec in years_dict.items():
        if year < 2018:  # only use pre-validation snapshots
            all_snapshots.append((team, year, vec))
print(f"  Available team-year snapshots (pre-2018): {len(all_snapshots)}")

# Sample N_SYNTH synthetic matchups
N_SYNTH = 10000
synth_rows = []
for _ in range(N_SYNTH):
    i_home, i_away = rng.choice(len(all_snapshots), size=2, replace=False)
    _, yr_h, vec_h = all_snapshots[i_home]
    _, yr_a, vec_a = all_snapshots[i_away]
    # Use average year as date_ordinal
    avg_year = (yr_h + yr_a) / 2
    date_ord = (avg_year - 2006) * 365
    importance = rng.choice([1, 2, 3, 4, 5], p=[0.3, 0.2, 0.25, 0.15, 0.1])
    neutral = rng.choice([0, 1], p=[0.6, 0.4])
    synth_rows.append([date_ord, importance, neutral] + vec_h + vec_a)

X_synth = np.array(synth_rows, dtype=np.float32)

# Predict lambda from GLM, then sample Poisson goals
lam_h = glm_home.predict(X_synth)
# For away goals, need to swap home/away features
X_synth_swap = X_synth.copy()
X_synth_swap[:, 3:3+len(HOME_COLS)] = X_synth[:, 3+len(HOME_COLS):]
X_synth_swap[:, 3+len(HOME_COLS):] = X_synth[:, 3:3+len(HOME_COLS)]
lam_a = glm_away.predict(X_synth_swap)

y_home_synth = rng.poisson(lam_h).clip(0, 5).astype(np.int64)
y_away_synth = rng.poisson(lam_a).clip(0, 5).astype(np.int64)

# Validate synthetic score distribution
print(f"  Synthetic mean home goals: {y_home_synth.mean():.2f}")
print(f"  Synthetic mean away goals: {y_away_synth.mean():.2f}")
print(f"  Synthetic % draws: {(y_home_synth==y_away_synth).mean():.1%}")
print(f"  Synthetic % home wins: {(y_home_synth>y_away_synth).mean():.1%}")

# Combine real + synthetic
X_aug = np.vstack([X_tr, X_synth]).astype(np.float32)
y_home_aug = np.concatenate([y_home_tr, y_home_synth])
y_away_aug = np.concatenate([y_away_tr, y_away_synth])
print(f"  Real: {X_tr.shape[0]}, Synthetic: {N_SYNTH}, "
      f"Total: {X_aug.shape[0]}")

t0 = _time.time()
print("  Fitting home_goals (real + simulated)...")
tab_home_sim = make_tabicl()
tab_home_sim.fit(X_aug, y_home_aug)
pred_home_sim_v = tab_home_sim.predict(X_va)
pred_home_sim_t = tab_home_sim.predict(X_te)
print(f"    done ({elapsed(t0)})")

t1 = _time.time()
print("  Fitting away_goals (real + simulated)...")
tab_away_sim = make_tabicl()
tab_away_sim.fit(X_aug, y_away_aug)
pred_away_sim_v = tab_away_sim.predict(X_va)
pred_away_sim_t = tab_away_sim.predict(X_te)
print(f"    done ({elapsed(t1)})")
print(f"  total time: {elapsed(t0)}")

for split, ph, pa, th, ta, to in [
    ("valid", pred_home_sim_v, pred_away_sim_v,
     y_home_va, y_away_va, y_outcome_va),
    ("test", pred_home_sim_t, pred_away_sim_t,
     y_home_te, y_away_te, y_outcome_te),
]:
    r = compute_metrics("Q2_poisson_sim_10k", split, ph, pa, th, ta, to)
    results.append(r)
    print(f"  {split}: acc_outcome={r['acc_outcome']:.3f}  "
          f"exact_score={r['exact_score']:.3f}  "
          f"mae_h={r['mae_home']:.2f} mae_a={r['mae_away']:.2f}  "
          f"draws={r['n_draws_pred']}/{r['n_draws_actual']}")

# %% Final summary
print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
df_r = pd.DataFrame(results)
print(df_r[[
    "method", "split", "acc_outcome", "exact_score",
    "mae_home", "mae_away", "n_draws_pred", "n_draws_actual",
]].to_string(index=False, float_format="%.3f"))
df_r.to_csv(OUT / "results.csv", index=False)

# %% Per-example output (test set, all methods)
df_out = df_test[["date", "home_team", "away_team",
    "home_score", "away_score", "outcome"]].copy()
df_out["Q1a_outcome"] = pred_out_t
df_out["Q1b_home"] = pred_home_t
df_out["Q1b_away"] = pred_away_t
df_out["Q1c_home"] = pred_home_reg_t
df_out["Q1c_away"] = pred_away_reg_t
df_out["Q2_home"] = pred_home_sim_t
df_out["Q2_away"] = pred_away_sim_t
df_out.to_csv(OUT / "per_example_test.csv", index=False)
print(f"\nSaved: {OUT}")
