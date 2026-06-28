"""Diverse synthetic data: forward counterfactuals + inverse feature generation.

Q: Does diverse synthetic training data (forward + inverse) help TabICL?

Forward: real matchup features → 6 diverse score generators
  (Poisson, NegBin, RF, TabICL self-play, with home-advantage sweep)
Inverse: target score distribution → sampled features from matching
  real matches (oversamples draws and tight games)

(NB tentative; need to double check LLM implementation of my spec)

Result: NO. real-only=66.7%, real+synth(31k)=52.8%. Still hurts.
  Forward correlation (MV→GD) r=0.20 vs real r=0.31 — generators
  produce weaker signal than reality. Inverse r=0.30 (better) but
  volume (5k) too small relative to forward (26k).

Conclusion: TabICL extracts max signal from 4.3k real rows already.
  Synthetic data from any model ≤ TabICL quality dilutes context.

Output: data/out/2606_synth_tournaments/matches.parquet
"""
# pyright: reportArgumentType=false, reportReturnType=false
# %% Setup
from __future__ import annotations

import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PoissonRegressor
from tabicl import TabICLClassifier

from tk import datadir
from tk.nbs.footie import (
    FEAT_COLS, HOME_COLS, VALID_CUT, TEST_CUT,
    compute_metrics, load_match_dataset, load_tm_features,
)

OUT = datadir / "out" / "2606_synth_tournaments"
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

# %% Load real training data
print("Loading real data...")
t0 = _time.time()
df_all = load_match_dataset(load_tm_features())
df_all = df_all[df_all["outcome"].notna()].copy()
df_all["home_score"] = df_all["home_score"].astype(int)
df_all["away_score"] = df_all["away_score"].astype(int)
df_real = df_all[df_all["date"] < VALID_CUT].copy()

X_real = df_real[FEAT_COLS].values.astype(np.float32)
y_home = df_real["home_score"].clip(0, 5).values
y_away = df_real["away_score"].clip(0, 5).values
home_names = df_real["home_team"].values
away_names = df_real["away_team"].values

X_swap = X_real.copy()
nc = len(HOME_COLS)
X_swap[:, 3:3+nc] = X_real[:, 3+nc:]
X_swap[:, 3+nc:] = X_real[:, 3:3+nc]
print(f"  {len(df_real)} training matches ({_time.time()-t0:.1f}s)")

# %% Fit generators
print("\nFitting generators...")
rng = np.random.default_rng(42)

glm_h = PoissonRegressor(alpha=0.01, max_iter=1000)
glm_h.fit(X_real, y_home.astype(float))
glm_a = PoissonRegressor(alpha=0.01, max_iter=1000)
glm_a.fit(X_swap, y_away.astype(float))

rf_h = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
rf_h.fit(X_real, y_home)
rf_a = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
rf_a.fit(X_swap, y_away)

tab_h = _tabicl()
tab_h.fit(X_real, y_home)
tab_a = _tabicl()
tab_a.fit(X_swap, y_away)
print("  All fitted.")

# %% Forward generation: counterfactual scores for real matchups
print(f"\n{'='*60}")
print("FORWARD: counterfactual scores for real matchups")
print(f"{'='*60}")

def _poisson(X, Xs, r, bonus=0.0):
    lh = np.clip(glm_h.predict(X) + bonus, 0.3, 5)
    la = np.clip(glm_a.predict(Xs), 0.3, 5)
    return r.poisson(lh), r.poisson(la)

def _negbin(X, Xs, r, bonus=0.0):
    lh = np.clip(glm_h.predict(X) + bonus, 0.3, 5)
    la = np.clip(glm_a.predict(Xs), 0.3, 5)
    n = 5
    return r.negative_binomial(n, n/(n+lh)), r.negative_binomial(n, n/(n+la))

def _rf(X, Xs, r):
    ph = rf_h.predict_proba(X)
    pa = rf_a.predict_proba(Xs)
    return (np.array([r.choice(len(p), p=p) for p in ph]),
            np.array([r.choice(len(p), p=p) for p in pa]))

def _tab(X, Xs, r):
    ph = tab_h.predict_proba(X)
    pa = tab_a.predict_proba(Xs)
    return (np.array([r.choice(len(p), p=p) for p in ph]),
            np.array([r.choice(len(p), p=p) for p in pa]))

generators = [
    ("poisson", lambda: _poisson(X_real, X_swap, rng)),
    ("poisson+0.2", lambda: _poisson(X_real, X_swap, rng, 0.2)),
    ("negbin", lambda: _negbin(X_real, X_swap, rng)),
    ("negbin+0.2", lambda: _negbin(X_real, X_swap, rng, 0.2)),
    ("rf", lambda: _rf(X_real, X_swap, rng)),
    ("tabicl", lambda: _tab(X_real, X_swap, rng)),
]

forward_rows = []
for gen_name, gen_fn in generators:
    t0 = _time.time()
    hs, as_ = gen_fn()
    hs, as_ = np.clip(hs, 0, 7).astype(int), np.clip(as_, 0, 7).astype(int)
    out = np.where(hs > as_, 2, np.where(hs == as_, 1, 0))
    print(f"  {gen_name:12s}: h={hs.mean():.2f} a={as_.mean():.2f} "
          f"D={(out==1).mean():.0%} H={(out==2).mean():.0%} "
          f"({_time.time()-t0:.1f}s)")
    for i in range(len(hs)):
        forward_rows.append({
            "home_team": home_names[i], "away_team": away_names[i],
            "home_score": int(hs[i]), "away_score": int(as_[i]),
            "generator": gen_name, "facet": "forward",
        })

# %% Inverse generation: score → plausible features
# Strategy: for each target score (h, a), find real matches with similar
# scores and RESAMPLE the features with small perturbations. This creates
# new (feature, score) pairs where the score distribution is controlled.
print(f"\n{'='*60}")
print("INVERSE: target scores → sampled features")
print(f"{'='*60}")

# Build lookup: score → list of row indices in real data
score_to_idx: dict[tuple[int, int], list[int]] = {}
for i, (h, a) in enumerate(zip(y_home, y_away)):
    score_to_idx.setdefault((int(h), int(a)), []).append(i)

# Target: oversample tight matches (0-1 goal diff) and draws
target_scores = []
for h in range(5):
    for a in range(5):
        diff = abs(h - a)
        weight = 3 if diff == 0 else (2 if diff == 1 else 1)
        target_scores.extend([(h, a)] * weight)

N_INVERSE = 5000
inverse_rows = []
noise_scale = 0.03
for _ in range(N_INVERSE):
    h_target, a_target = target_scores[rng.integers(len(target_scores))]
    # Find a real match with this score (or close)
    if (h_target, a_target) in score_to_idx:
        idx = rng.choice(score_to_idx[(h_target, a_target)])
    else:
        # Fallback: find closest score
        candidates = [(abs(h_target-h)+abs(a_target-a), k)
            for k, v in score_to_idx.items() for h, a in [k]]
        _, closest_score = min(candidates)
        idx = rng.choice(score_to_idx[closest_score])
    # Perturb features slightly (TM cols only, keep date/imp/neutral)
    feats = X_real[idx].copy()
    feats[3:] *= (1 + rng.normal(0, noise_scale, size=len(feats)-3))
    inverse_rows.append({
        "home_team": home_names[idx], "away_team": away_names[idx],
        "home_score": h_target, "away_score": a_target,
        "generator": "inverse", "facet": "inverse",
    })

# Store features for inverse rows
inverse_feats = np.array([
    X_real[rng.choice(score_to_idx.get(
        (r["home_score"], r["away_score"]),
        score_to_idx[min(score_to_idx.keys(),
            key=lambda k: abs(k[0]-r["home_score"])+abs(k[1]-r["away_score"]))]
    ))].copy() * (1 + rng.normal(0, noise_scale, size=X_real.shape[1]))
    for r in inverse_rows
], dtype=np.float32)
# Fix date/importance/neutral (don't perturb these)
for i, r in enumerate(inverse_rows):
    idx = rng.integers(len(X_real))
    inverse_feats[i, :3] = X_real[idx, :3]

inv_out = np.where(
    np.array([r["home_score"] for r in inverse_rows]) >
    np.array([r["away_score"] for r in inverse_rows]), 2,
    np.where(
        np.array([r["home_score"] for r in inverse_rows]) ==
        np.array([r["away_score"] for r in inverse_rows]), 1, 0))
print(f"  {N_INVERSE} inverse samples: "
      f"D={(inv_out==1).mean():.0%} H={(inv_out==2).mean():.0%}")

# %% Build combined output
print(f"\n{'='*60}")
print("BUILDING OUTPUT")
print(f"{'='*60}")

# Forward features: tile real features per generator
X_forward = np.tile(X_real, (len(generators), 1))
df_forward = pd.DataFrame(forward_rows)
for i, col in enumerate(FEAT_COLS):
    df_forward[col] = X_forward[:, i]
df_forward["outcome"] = np.where(
    df_forward["home_score"] > df_forward["away_score"], 2,
    np.where(df_forward["home_score"] == df_forward["away_score"], 1, 0))

# Inverse features
df_inverse = pd.DataFrame(inverse_rows)
for i, col in enumerate(FEAT_COLS):
    df_inverse[col] = inverse_feats[:, i]
df_inverse["outcome"] = inv_out

df_synth = pd.concat([df_forward, df_inverse], ignore_index=True)
print(f"  Forward: {len(df_forward)} ({df_forward['generator'].nunique()} gens)")
print(f"  Inverse: {len(df_inverse)}")
print(f"  Total:   {len(df_synth)}")

# %% Validate
print(f"\n{'='*60}")
print("VALIDATION")
print(f"{'='*60}")
print(f"Real: h={y_home.mean():.2f} a={y_away.mean():.2f} "
      f"D={(y_home==y_away).mean():.0%} H={(y_home>y_away).mean():.0%}")
print(f"Synth: h={df_synth['home_score'].mean():.2f} "
      f"a={df_synth['away_score'].mean():.2f} "
      f"D={(df_synth['outcome']==1).mean():.0%} "
      f"H={(df_synth['outcome']==2).mean():.0%}")
print(f"\nPer-facet:")
for facet in ["forward", "inverse"]:
    sub = df_synth[df_synth["facet"] == facet]
    print(f"  {facet}: n={len(sub)} "
          f"D={(sub['outcome']==1).mean():.0%} "
          f"H={(sub['outcome']==2).mean():.0%}")
print(f"\nCorrelation (MV diff → goal diff):")
mv_diff_real = X_real[:, 3] - X_real[:, 3+nc]
gd_real = y_home.astype(float) - y_away.astype(float)
print(f"  Real: r={np.corrcoef(mv_diff_real, gd_real)[0,1]:.3f}")
for facet in ["forward", "inverse"]:
    sub = df_synth[df_synth["facet"] == facet]
    mv = sub[FEAT_COLS[3]].values - sub[FEAT_COLS[3+nc]].values
    gd = sub["home_score"] - sub["away_score"]
    print(f"  {facet}: r={np.corrcoef(mv, gd)[0,1]:.3f}")

# %% Save
out_path = OUT / "matches.parquet"
df_synth.to_parquet(out_path, index=False)
print(f"\nSaved: {out_path} ({out_path.stat().st_size/1e6:.1f}MB)")

# %% Quick self-eval: does this synth data help?
print(f"\n{'='*60}")
print("SELF-EVAL: real-only vs real+synth")
print(f"{'='*60}")

df_valid = df_all[(df_all["date"] >= VALID_CUT) & (df_all["date"] < TEST_CUT)
    & (df_all["tournament"] == WC)]
df_test = df_all[(df_all["date"] >= TEST_CUT)
    & (df_all["tournament"] == WC)]
X_va = df_valid[FEAT_COLS].values.astype(np.float32)
X_te = df_test[FEAT_COLS].values.astype(np.float32)
y_out_te = df_test["outcome"].values.astype(int)
y_h_te = df_test["home_score"].clip(0, 5).values
y_a_te = df_test["away_score"].clip(0, 5).values

def _quick_eval(label, X_fit, y_h_fit, y_a_fit):
    t0 = _time.time()
    mh, ma = _tabicl(), _tabicl()
    mh.fit(X_fit, y_h_fit)
    ma.fit(X_fit, y_a_fit)
    ph, pa = mh.predict(X_te), ma.predict(X_te)
    m = compute_metrics(ph, pa, y_h_te, y_a_te, y_out_te)
    print(f"  {label:30s} outcome={m['acc_outcome']:.3f} "
          f"exact={m['exact_score']:.3f} "
          f"draws={m['n_draws_pred']}/{m['n_draws_actual']} "
          f"({_time.time()-t0:.0f}s)")
    return m

X_synth = df_synth[FEAT_COLS].values.astype(np.float32)
y_h_synth = df_synth["home_score"].clip(0, 5).values.astype(np.int64)
y_a_synth = df_synth["away_score"].clip(0, 5).values.astype(np.int64)
X_combined = np.vstack([X_real, X_synth]).astype(np.float32)
y_h_combined = np.concatenate([y_home, y_h_synth])
y_a_combined = np.concatenate([y_away, y_a_synth])

m_real = _quick_eval("real only", X_real, y_home, y_away)
m_synth = _quick_eval("real + synth (fwd+inv)", X_combined, y_h_combined, y_a_combined)

delta = m_synth["acc_outcome"] - m_real["acc_outcome"]
print(f"\n  Delta: {delta:+.3f} ({'helps' if delta > 0 else 'hurts'})")
