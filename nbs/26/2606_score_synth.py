"""Multi-output score prediction vs direct outcome classification.

Q1: Does predicting exact scores (two classifiers: home_goals, away_goals)
    then deriving outcome beat direct 3-way classification?
Q2: Does mirror augmentation (swap home/away, lossless) help ICL?
Q3: Does fine-tuning TabICL on synthetic data help ICL?
    Setup: fine-tune on synth, validate on 2014 WC, then ICL with real
    data, evaluate on 2018/2022 (valid) and 2026 (test).

A1: YES. Multi-output clf best on test (66.7% vs 65.3% direct) AND solves
    draws (9/20 vs 0/20). Exact score ~14%. Regression terrible (35%).
A2: Mirror augmentation gives +1.4pp -> 68.1%.
A3: TBD.

Usage:
  uv run python nbs/26/2606_score_synth.py q1
  uv run python nbs/26/2606_score_synth.py q2
  uv run python nbs/26/2606_score_synth.py q3
  uv run python nbs/26/2606_score_synth.py all
"""
# pyright: reportArgumentType=false, reportReturnType=false
from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score
from tabicl import TabICLClassifier
from tabicl._finetune.classifier import FinetunedTabICLClassifier

from tk import datadir
from tk.nbs.footie import (
    FEAT_COLS, HOME_COLS, VALID_CUT, TEST_CUT,
    compute_metrics, load_match_dataset, load_tm_features,
)

OUT = datadir / "out" / "2606_score_synth"
OUT.mkdir(parents=True, exist_ok=True)
WC = "FIFA World Cup"
FT_OUT = OUT / "ft_ckpt"

ckpt = Path(hf_hub_download(
    repo_id="jingang/TabICL",
    filename="tabicl-classifier-v2-20260212.ckpt",
))


def _tabicl(model_path=None):
    return TabICLClassifier(
        model_path=model_path or ckpt, norm_methods=None,
        feat_shuffle_method="latin", class_shuffle_method="shift",
        outlier_threshold=4, support_many_classes=True, batch_size=8,
        kv_cache=False, allow_auto_download=False,
        checkpoint_version=ckpt.name,
        use_amp="auto", use_fa3="auto", offload_mode="auto",
        disk_offload_dir=None,
    )


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_data():
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

    print(f"Train: {len(df_train)}  Valid(WC 2018+2022): {len(df_valid)}"
          f"  Test(WC 2026): {len(df_test)}")
    return df, df_train, df_valid, df_test


def get_arrays(df_train, df_valid, df_test):
    X_tr = df_train[FEAT_COLS].values.astype(np.float32)
    X_va = df_valid[FEAT_COLS].values.astype(np.float32)
    X_te = df_test[FEAT_COLS].values.astype(np.float32)
    return dict(
        X_tr=X_tr, X_va=X_va, X_te=X_te,
        y_out_va=df_valid["outcome"].values,
        y_out_te=df_test["outcome"].values,
        y_home_tr=df_train["home_score"].clip(0, 5).values,
        y_away_tr=df_train["away_score"].clip(0, 5).values,
        y_home_va=df_valid["home_score"].clip(0, 5).values,
        y_away_va=df_valid["away_score"].clip(0, 5).values,
        y_home_te=df_test["home_score"].clip(0, 5).values,
        y_away_te=df_test["away_score"].clip(0, 5).values,
    )


def eval_model(name, ph_v, pa_v, ph_t, pa_t, d):
    results = []
    for split, ph, pa, th, ta, to in [
        ("valid", ph_v, pa_v, d["y_home_va"], d["y_away_va"], d["y_out_va"]),
        ("test", ph_t, pa_t, d["y_home_te"], d["y_away_te"], d["y_out_te"]),
    ]:
        m = compute_metrics(ph, pa, th, ta, to)
        m["method"] = name
        m["split"] = split
        results.append(m)
        print(f"  {split}: outcome={m['acc_outcome']:.3f} "
              f"exact={m['exact_score']:.3f} "
              f"draws={m['n_draws_pred']}/{m['n_draws_actual']}")
    return results


# --------------------------------------------------------------------------- #
# Q1: Multi-output vs direct vs regression
# --------------------------------------------------------------------------- #

def run_q1(df_train, d):
    results = []

    print("\nQ1a: Direct outcome classification")
    t0 = _time.time()
    tab = _tabicl()
    tab.fit(d["X_tr"], df_train["outcome"].values)
    pred_v, pred_t = tab.predict(d["X_va"]), tab.predict(d["X_te"])
    print(f"  ({_time.time()-t0:.0f}s)")
    for split, y_true, y_pred in [("valid", d["y_out_va"], pred_v),
                                   ("test", d["y_out_te"], pred_t)]:
        acc = accuracy_score(y_true, y_pred)
        results.append({"method": "Q1a_direct", "split": split,
            "acc_outcome": acc, "exact_score": np.nan,
            "mae_home": np.nan, "mae_away": np.nan,
            "n_draws_pred": int((y_pred == 1).sum()),
            "n_draws_actual": int((y_true == 1).sum())})
        print(f"  {split}: outcome={acc:.3f} "
              f"draws={(y_pred==1).sum()}/{(y_true==1).sum()}")

    print("\nQ1b: Multi-output classification (home + away goals)")
    t0 = _time.time()
    tab_h = _tabicl()
    tab_h.fit(d["X_tr"], d["y_home_tr"])
    tab_a = _tabicl()
    tab_a.fit(d["X_tr"], d["y_away_tr"])
    ph_v = tab_h.predict(d["X_va"])
    pa_v = tab_a.predict(d["X_va"])
    ph_t = tab_h.predict(d["X_te"])
    pa_t = tab_a.predict(d["X_te"])
    print(f"  ({_time.time()-t0:.0f}s)")
    results += eval_model("Q1b_multiout", ph_v, pa_v, ph_t, pa_t, d)

    print("\nQ1c: GBR regression -> round")
    t0 = _time.time()
    gbr_h = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
    gbr_h.fit(d["X_tr"], df_train["home_score"].values.astype(float))
    gbr_a = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
    gbr_a.fit(d["X_tr"], df_train["away_score"].values.astype(float))
    ph_v = np.clip(np.round(gbr_h.predict(d["X_va"])), 0, 9).astype(int)
    pa_v = np.clip(np.round(gbr_a.predict(d["X_va"])), 0, 9).astype(int)
    ph_t = np.clip(np.round(gbr_h.predict(d["X_te"])), 0, 9).astype(int)
    pa_t = np.clip(np.round(gbr_a.predict(d["X_te"])), 0, 9).astype(int)
    print(f"  ({_time.time()-t0:.0f}s)")
    results += eval_model("Q1c_regression", ph_v, pa_v, ph_t, pa_t, d)

    return results


# --------------------------------------------------------------------------- #
# Q2: Mirror augmentation
# --------------------------------------------------------------------------- #

def run_q2(d):
    print("\nQ2: Multi-output + mirror augmentation")
    t0 = _time.time()
    nc = len(HOME_COLS)
    X_tr = d["X_tr"]
    X_mirror = X_tr.copy()
    X_mirror[:, 3:3+nc] = X_tr[:, 3+nc:]
    X_mirror[:, 3+nc:] = X_tr[:, 3:3+nc]
    X_aug = np.vstack([X_tr, X_mirror]).astype(np.float32)
    y_h_aug = np.concatenate([d["y_home_tr"], d["y_away_tr"]])
    y_a_aug = np.concatenate([d["y_away_tr"], d["y_home_tr"]])
    print(f"  Real={X_tr.shape[0]} + Mirror={X_mirror.shape[0]} = {X_aug.shape[0]}")

    tab_h = _tabicl()
    tab_h.fit(X_aug, y_h_aug)
    tab_a = _tabicl()
    tab_a.fit(X_aug, y_a_aug)
    ph_v = tab_h.predict(d["X_va"])
    pa_v = tab_a.predict(d["X_va"])
    ph_t = tab_h.predict(d["X_te"])
    pa_t = tab_a.predict(d["X_te"])
    print(f"  ({_time.time()-t0:.0f}s)")
    return eval_model("Q2_mirror", ph_v, pa_v, ph_t, pa_t, d)


# --------------------------------------------------------------------------- #
# Q3: Fine-tune on synthetic, then ICL on real
# --------------------------------------------------------------------------- #

SYNTH_CONFIGS = {
    "tabicl_fwd": {"facet": "forward", "generator": "tabicl"},
    "inverse": {"facet": "inverse"},
    "tabicl_fwd+inverse": [
        {"facet": "forward", "generator": "tabicl"},
        {"facet": "inverse"},
    ],
    "all_fwd": {"facet": "forward"},
}


def _load_synth(config):
    """Load synth subset based on config."""
    synth_path = datadir / "out" / "2606_synth_tournaments" / "matches.parquet"
    df_synth = pd.read_parquet(synth_path)
    if isinstance(config, list):
        parts = []
        for c in config:
            mask = df_synth["facet"] == c["facet"]
            if "generator" in c:
                mask &= df_synth["generator"] == c["generator"]
            parts.append(df_synth[mask])
        return pd.concat(parts, ignore_index=True)
    mask = df_synth["facet"] == config["facet"]
    if "generator" in config:
        mask &= df_synth["generator"] == config["generator"]
    return df_synth[mask].copy()


def run_q3(df, df_train, d):
    """Fine-tune TabICL on synth, then ICL with real data.

    Tries multiple synth sources (tabicl forward, inverse, combined, all forward).
    For each: fine-tune -> save ckpt -> ICL with real -> evaluate.
    Logs to wandb project 'tabicl'.
    """
    synth_path = datadir / "out" / "2606_synth_tournaments" / "matches.parquet"
    if not synth_path.exists():
        print("  SKIP: run 2606_simulate_synth.py first")
        return []

    # Fine-tune validation: 2014 WC
    df_ft_val = df[(df["tournament"] == WC)
        & (df["date"] >= "2014-06-01") & (df["date"] < "2015-01-01")]
    X_ft_val = df_ft_val[FEAT_COLS].values.astype(np.float32)
    y_h_ft_val = df_ft_val["home_score"].clip(0, 5).values.astype(np.int64)
    y_a_ft_val = df_ft_val["away_score"].clip(0, 5).values.astype(np.int64)
    print(f"  Fine-tune validation: {len(df_ft_val)} matches (2014 WC)")

    results = []

    for synth_name, synth_cfg in SYNTH_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Q3 [{synth_name}]: Fine-tune on synth -> ICL on real")
        print(f"{'='*60}")

        df_s = _load_synth(synth_cfg)
        X_synth = df_s[FEAT_COLS].values.astype(np.float32)
        y_h_synth = df_s["home_score"].clip(0, 5).values.astype(np.int64)
        y_a_synth = df_s["away_score"].clip(0, 5).values.astype(np.int64)
        print(f"  Synth: {len(df_s)} samples ({synth_name})")

        # Clip ft_val to match synth label range
        synth_max = max(y_h_synth.max(), y_a_synth.max())
        y_h_val = np.clip(y_h_ft_val, 0, synth_max)
        y_a_val = np.clip(y_a_ft_val, 0, synth_max)

        run_dir = FT_OUT / synth_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Fine-tune home model
        print(f"\n  Fine-tuning home_score model...")
        t0 = _time.time()
        ft_h = FinetunedTabICLClassifier(
            model_path=ckpt, epochs=30, learning_rate=1e-5,
            n_estimators_finetune=2, n_estimators_inference=8,
            norm_methods=None, feat_shuffle_method="latin",
            class_shuffle_method="shift", outlier_threshold=4,
            support_many_classes=True, early_stopping=True,
            patience=8, eval_metric="accuracy",
            allow_auto_download=False, checkpoint_version=ckpt.name,
            verbose=True,
            wandb_kwargs={"project": "tabicl",
                          "name": f"ft_{synth_name}_home",
                          "tags": ["q3", synth_name, "home"]},
        )
        ft_h.fit(X_synth, y_h_synth,
                 X_val=X_ft_val, y_val=y_h_val,
                 output_dir=run_dir / "home")

        # Fine-tune away model
        print(f"\n  Fine-tuning away_score model...")
        ft_a = FinetunedTabICLClassifier(
            model_path=ckpt, epochs=30, learning_rate=1e-5,
            n_estimators_finetune=2, n_estimators_inference=8,
            norm_methods=None, feat_shuffle_method="latin",
            class_shuffle_method="shift", outlier_threshold=4,
            support_many_classes=True, early_stopping=True,
            patience=8, eval_metric="accuracy",
            allow_auto_download=False, checkpoint_version=ckpt.name,
            verbose=True,
            wandb_kwargs={"project": "tabicl",
                          "name": f"ft_{synth_name}_away",
                          "tags": ["q3", synth_name, "away"]},
        )
        ft_a.fit(X_synth, y_a_synth,
                 X_val=X_ft_val, y_val=y_a_val,
                 output_dir=run_dir / "away")
        print(f"  Fine-tune done ({_time.time()-t0:.0f}s)")

        # ICL with fine-tuned model on real data
        print(f"\n  ICL with fine-tuned model (real context)...")
        t0 = _time.time()
        ft_ckpt_h = run_dir / "home" / "best.ckpt"
        ft_ckpt_a = run_dir / "away" / "best.ckpt"

        tab_h = _tabicl(model_path=ft_ckpt_h)
        tab_h.fit(d["X_tr"], d["y_home_tr"])
        tab_a = _tabicl(model_path=ft_ckpt_a)
        tab_a.fit(d["X_tr"], d["y_away_tr"])

        ph_v = tab_h.predict(d["X_va"])
        pa_v = tab_a.predict(d["X_va"])
        ph_t = tab_h.predict(d["X_te"])
        pa_t = tab_a.predict(d["X_te"])
        print(f"  ICL done ({_time.time()-t0:.0f}s)")
        results += eval_model(f"Q3_ft_{synth_name}", ph_v, pa_v, ph_t, pa_t, d)

    return results


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    stages = sys.argv[1:] if len(sys.argv) > 1 else ["all"]
    run_all = "all" in stages

    df, df_train, df_valid, df_test = load_data()
    d = get_arrays(df_train, df_valid, df_test)
    results = []

    if run_all or "q1" in stages:
        results += run_q1(df_train, d)

    if run_all or "q2" in stages:
        results += run_q2(d)

    if run_all or "q3" in stages:
        results += run_q3(df, df_train, d)

    if results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        df_r = pd.DataFrame(results)
        cols = ["method", "split", "acc_outcome", "exact_score",
            "n_draws_pred", "n_draws_actual"]
        avail = [c for c in cols if c in df_r.columns]
        print(df_r[avail].to_string(index=False, float_format="%.3f"))
        df_r.to_csv(OUT / "results.csv", index=False)
        print(f"\nSaved: {OUT / 'results.csv'}")


if __name__ == "__main__":
    main()
