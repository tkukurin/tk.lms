"""Multi-output score prediction vs direct outcome classification.

Q1: Does predicting exact scores (two classifiers: home_goals, away_goals)
    then deriving outcome beat direct 3-way classification?
Q2: Does mirror augmentation (swap home/away, lossless) help ICL?
Q3: Does synthetic adaptation from a football task prior help TabICL ICL?
    Setup: sample task-prior scores from TM+ranking contexts, adapt TabICL,
    validate on 2014 WC, then ICL with real context and evaluate WC splits.

A1: YES. Multi-output clf best on test (66.7% vs 65.3% direct) AND solves
    draws (9/20 vs 0/20). Exact score ~14%. Regression terrible (35%).
A2: Mirror augmentation gives +1.4pp -> 68.1%.
A3: Full TabICL fine-tune on the task prior recovers 66.7% test / 9 draws.
    Retouche input adapter was worse (45.8% test, 0 draws).

Usage:
  uv run python nbs/26/2606_score_synth.py q1
  uv run python nbs/26/2606_score_synth.py q2
  uv run python nbs/26/2606_score_synth.py q3       # Retouche adapter
  uv run python nbs/26/2606_score_synth.py q3_ft    # full TabICL fine-tune
  uv run python nbs/26/2606_score_synth.py all

Remote via SkyPilot (uses .skyignore-filtered workdir sync):
  uv run python nbs/26/2606_score_synth.py q3_ft --sky
  uv run python nbs/26/2606_score_synth.py q3_ft --sky cluster=tk-footie-retouche-l4,gpus=L4:1,idle_mins=30
"""
# pyright: reportArgumentType=false, reportReturnType=false
from __future__ import annotations

import os
import argparse
import shlex
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score
from tabicl import TabICLClassifier
from tabicl._finetune.classifier import FinetunedTabICLClassifier
from tabicl._finetune.data import iter_epoch_meta_batches, move_meta_batch

from tk import datadir
from tk.nbs.footie import (
    FEAT_COLS,
    HOME_COLS,
    TEST_CUT,
    VALID_CUT,
    compute_metrics,
    load_match_dataset,
    load_tm_features,
)
from tk.nbs.footie.priors import FootiePriorConfig, FootiePriorSampler
from tk.utils import sky as sky_utils

OUT = datadir / "out" / "2606_score_synth"
OUT.mkdir(parents=True, exist_ok=True)
WC = "FIFA World Cup"
FT_OUT = OUT / "ft_ckpt"

ckpt = Path(hf_hub_download(
    repo_id="jingang/TabICL",
    filename="tabicl-classifier-v2-20260212.ckpt",
))
CHECKPOINT_VERSION = ckpt.name


def _tabicl(model_path=ckpt):
    return TabICLClassifier(
        model_path=model_path,
        checkpoint_version=CHECKPOINT_VERSION,
        norm_methods=None,
        feat_shuffle_method="latin",
        class_shuffle_method="shift",
        outlier_threshold=4, support_many_classes=True,
        batch_size=8, kv_cache=False,
        allow_auto_download=False,
        use_amp="auto", use_fa3="auto",
        offload_mode="auto", disk_offload_dir=None,
    )


def load_data():
    t0 = _time.time()
    df = load_match_dataset(load_tm_features())
    df = df[df["outcome"].notna()].copy()
    df["outcome"] = df["outcome"].astype(int)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    print(f"Loaded {len(df)} matches ({_time.time()-t0:.1f}s)")

    df_train = df[df["date"] < VALID_CUT]
    df_valid = df[
        (df["date"] >= VALID_CUT)
        & (df["date"] < TEST_CUT)
        & (df["tournament"] == WC)
    ]
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
        ("valid", ph_v, pa_v,
            d["y_home_va"], d["y_away_va"], d["y_out_va"]),
        ("test", ph_t, pa_t,
            d["y_home_te"], d["y_away_te"], d["y_out_te"]),
    ]:
        m = compute_metrics(ph, pa, th, ta, to)
        m["method"] = name
        m["split"] = split
        results.append(m)
        print(f"  {split}: outcome={m['acc_outcome']:.3f} "
              f"exact={m['exact_score']:.3f} "
              f"draws={m['n_draws_pred']}/{m['n_draws_actual']}")
    return results


def run_q1(df_train, d):
    results = []
    print("\nQ1a: Direct outcome classification")
    t0 = _time.time()
    tab = _tabicl()
    tab.fit(d["X_tr"], df_train["outcome"].values)
    pred_v = tab.predict(d["X_va"])
    pred_t = tab.predict(d["X_te"])
    print(f"  ({_time.time()-t0:.0f}s)")
    for split, y_true, y_pred in [
        ("valid", d["y_out_va"], pred_v),
        ("test", d["y_out_te"], pred_t),
    ]:
        acc = accuracy_score(y_true, y_pred)
        results.append({
            "method": "Q1a_direct", "split": split,
            "acc_outcome": acc, "exact_score": np.nan,
            "mae_home": np.nan, "mae_away": np.nan,
            "n_draws_pred": int((y_pred == 1).sum()),
            "n_draws_actual": int((y_true == 1).sum()),
        })
        print(f"  {split}: outcome={acc:.3f} "
              f"draws={(y_pred==1).sum()}/{(y_true==1).sum()}")

    print("\nQ1b: Multi-output classification (home + away)")
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
    gbr_h = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, random_state=42,
    )
    gbr_h.fit(d["X_tr"], df_train["home_score"].values.astype(float))
    gbr_a = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, random_state=42,
    )
    gbr_a.fit(d["X_tr"], df_train["away_score"].values.astype(float))
    ph_v = np.clip(np.round(gbr_h.predict(d["X_va"])), 0, 9).astype(int)
    pa_v = np.clip(np.round(gbr_a.predict(d["X_va"])), 0, 9).astype(int)
    ph_t = np.clip(np.round(gbr_h.predict(d["X_te"])), 0, 9).astype(int)
    pa_t = np.clip(np.round(gbr_a.predict(d["X_te"])), 0, 9).astype(int)
    print(f"  ({_time.time()-t0:.0f}s)")
    results += eval_model("Q1c_regression", ph_v, pa_v, ph_t, pa_t, d)

    return results


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


FOOTIE_PRIOR_CONFIG = FootiePriorConfig(n_tasks=64)
FOOTIE_PRIOR_SEED = 42


class InputResidualAdapter(nn.Module):
    """Residual MLP in input-feature space, identity-initialized (arXiv:2605.06047)."""

    def __init__(self, n_features: int, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_features))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def _retouche_train(
    name: str,
    target: str,
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    out_dir: Path,
    model_path: Path = ckpt,
    epochs: int = 20,
    lr: float = 1e-3,
    n_estimators: int = 2,
) -> InputResidualAdapter | None:
    """Train Retouche input-space adapter through frozen TabICL.

    Returns trained adapter, or None if identity guard fires.
    """
    adapter_path = out_dir / f"adapter_{target}.pt"
    guard_path = out_dir / f"guard_{target}.txt"

    if guard_path.exists():
        print(f"  [{target}] Identity guard fired (cached) → using frozen base")
        return None
    if adapter_path.exists():
        print(f"  [{target}] Reusing {adapter_path}")
        adapter = InputResidualAdapter(X.shape[1])
        adapter.load_state_dict(torch.load(adapter_path, weights_only=True))
        return adapter

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from tabicl._sklearn.classifier import TabICLClassifier as _Clf
    loader = _Clf(
        model_path=model_path, norm_methods=None,
        feat_shuffle_method="latin", class_shuffle_method="shift",
        outlier_threshold=4, support_many_classes=True,
        batch_size=8, kv_cache=False, allow_auto_download=False,
        checkpoint_version=CHECKPOINT_VERSION,
        use_amp="auto", use_fa3="auto",
        offload_mode="auto", disk_offload_dir=None,
    )
    loader._resolve_device()
    loader._load_model()
    frozen_model = loader.model_.to(device)
    # Keep TabICL weights frozen, but use training forward so logits retain a
    # gradient path to adapted_X. eval() dispatches through the inference path,
    # which may detach/offload tensors and breaks input-space Retouche training.
    frozen_model.train()
    for p in frozen_model.parameters():
        p.requires_grad = False

    n_features = X.shape[1]
    adapter = InputResidualAdapter(n_features).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=0.01)
    n_classes = int(y.max()) + 1
    n_params = sum(p.numel() for p in adapter.parameters())

    import wandb
    run = wandb.init(
        project="tabicl",
        name=f"retouche_{name}_{target}",
        tags=["q3", "retouche", name, target],
        config={
            "method": "retouche", "target": target, "epochs": epochs,
            "lr": lr, "n_estimators": n_estimators, "n_features": n_features,
            "n_params": n_params, "n_classes": n_classes,
            "n_train": len(y), "n_val": len(y_val),
        },
    )
    print(f"  [{target}] Training Retouche adapter "
          f"({n_params} params, {epochs} epochs, {n_classes} classes)")

    best_loss = float("inf")
    for epoch in range(epochs):
        adapter.train()
        epoch_loss, n_batches = 0.0, 0
        for batch in iter_epoch_meta_batches(
            X, y,
            classification=True,
            n_estimators=n_estimators,
            max_chunk_size=min(len(y), 10_000),
            query_ratio=0.2,
            epoch_seed=epoch * 7 + 42,
            preprocessing_seed=123,
            norm_methods=None,
            feat_shuffle_method="latin",
            class_shuffle_method="shift",
            outlier_threshold=4,
        ):
            batch = move_meta_batch(batch, device)
            adapted_X = adapter(batch.X)
            logits = frozen_model(adapted_X, batch.y_train.float())
            logits_used = logits[..., :n_classes].reshape(-1, n_classes)
            loss = F.cross_entropy(logits_used, batch.y_query.long().reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        best_loss = min(best_loss, avg)
        wandb.log({"epoch": epoch + 1, "train/loss": avg, "train/best_loss": best_loss})
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    epoch {epoch+1:>2}/{epochs}  loss={avg:.4f}")

    # identity guard: adapted vs base on held-out validation split
    adapter.eval()
    mid = len(X_val) // 2

    base_tab = _tabicl(model_path)
    base_tab.fit(X_val[:mid], y_val[:mid])
    base_acc = accuracy_score(y_val[mid:], base_tab.predict(X_val[mid:]))

    with torch.no_grad():
        X_val_adapted = adapter(
            torch.from_numpy(X_val.astype(np.float32)).to(device)
        ).cpu().numpy()
    adapted_tab = _tabicl(model_path)
    adapted_tab.fit(X_val_adapted[:mid], y_val[:mid])
    adapted_acc = accuracy_score(y_val[mid:], adapted_tab.predict(X_val_adapted[mid:]))

    wandb.log({"guard/base_acc": base_acc, "guard/adapted_acc": adapted_acc})
    print(f"  [{target}] Identity guard: base={base_acc:.3f} adapted={adapted_acc:.3f}")

    if adapted_acc <= base_acc:
        print(f"  [{target}] Guard FIRED → falling back to frozen base")
        guard_path.write_text(f"base={base_acc:.4f} adapted={adapted_acc:.4f}\n")
        wandb.log({"guard/fired": True})
        run.finish()
        return None

    torch.save(adapter.state_dict(), adapter_path)
    wandb.log({"guard/fired": False})
    run.finish()
    print(f"  [{target}] Adapter saved to {adapter_path}")
    return adapter.cpu()


def _sample_ranking_goal_synth(
    df_train: pd.DataFrame,
    config: FootiePriorConfig = FOOTIE_PRIOR_CONFIG,
    seed: int = FOOTIE_PRIOR_SEED,
) -> pd.DataFrame:
    """Generate task-prior synthetic score labels over real football contexts."""
    sampler = FootiePriorSampler.from_matches(df_train, config=config)
    return sampler.sample_rows(random_state=seed)


def _apply_adapter(adapter: InputResidualAdapter | None, X: np.ndarray) -> np.ndarray:
    if adapter is None:
        return X
    adapter.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X.astype(np.float32)).to(
            next(adapter.parameters()).device
        )
        return adapter(X_t).cpu().numpy()


def _finetune_full(
    name: str,
    target: str,
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    out_dir: Path,
    model_path: Path = ckpt,
) -> Path:
    """Run memory-capped full TabICL fine-tuning for one score target."""
    best = out_dir / "best.ckpt"
    if best.exists():
        print(f"  [{target}] Reusing {best}")
        return best

    ft = FinetunedTabICLClassifier(
        model_path=model_path,
        epochs=30,
        learning_rate=1e-5,
        n_estimators_finetune=1,
        n_estimators_validation=1,
        n_estimators_inference=8,
        max_data_size=5_000,
        norm_methods=None,
        feat_shuffle_method="latin",
        class_shuffle_method="shift",
        outlier_threshold=4,
        support_many_classes=True,
        early_stopping=True,
        patience=8,
        eval_metric="accuracy",
        allow_auto_download=False,
        checkpoint_version=CHECKPOINT_VERSION,
        verbose=True,
        wandb_kwargs={
            "project": "tabicl",
            "name": f"ft_{name}_{target}",
            "tags": ["q3_ft", name, target],
        },
    )
    ft.fit(X, y, X_val=X_val, y_val=y_val, output_dir=out_dir)
    return best


def _fine_tune_validation(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df_ft_val = df[
        (df["tournament"] == WC)
        & (df["date"] >= "2014-06-01")
        & (df["date"] < "2015-01-01")
    ]
    X_ft_val = np.asarray(df_ft_val[FEAT_COLS], dtype=np.float32)
    y_h_ft_val = np.asarray(df_ft_val["home_score"].clip(0, 5), dtype=np.int64)
    y_a_ft_val = np.asarray(df_ft_val["away_score"].clip(0, 5), dtype=np.int64)
    print(f"  Fine-tune validation: {len(df_ft_val)} matches (2014 WC)")
    return X_ft_val, y_h_ft_val, y_a_ft_val


def run_q3_ft(df, df_train, d):
    """Full TabICL fine-tuning on task-prior synth, then ICL with real data."""
    X_ft_val, y_h_ft_val, y_a_ft_val = _fine_tune_validation(df)

    synth_name = "footie_task_prior_fullft"
    print(f"\nQ3_FT [{synth_name}]: full fine-tune on synth -> ICL on real")

    df_s = _sample_ranking_goal_synth(df_train)
    X_synth = df_s[FEAT_COLS].values.astype(np.float32)
    y_h_synth = df_s["home_score"].clip(0, 5).values.astype(np.int64)
    y_a_synth = df_s["away_score"].clip(0, 5).values.astype(np.int64)
    print(f"  Synth: {len(df_s)} samples ({synth_name})")

    synth_max = max(y_h_synth.max(), y_a_synth.max())
    y_h_val = np.clip(y_h_ft_val, 0, synth_max)
    y_a_val = np.clip(y_a_ft_val, 0, synth_max)

    run_dir = FT_OUT / synth_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n  Full fine-tuning home + away models...")
    t0 = _time.time()
    ckpt_h = _finetune_full(
        synth_name, "home", X_synth, y_h_synth,
        X_ft_val, y_h_val, run_dir / "home",
    )
    ckpt_a = _finetune_full(
        synth_name, "away", X_synth, y_a_synth,
        X_ft_val, y_a_val, run_dir / "away",
    )
    print(f"  Full fine-tune done ({_time.time()-t0:.0f}s)")

    print("\n  ICL with fine-tuned model (real context)...")
    t0 = _time.time()
    tab_h = _tabicl(model_path=ckpt_h)
    tab_h.fit(d["X_tr"], d["y_home_tr"])
    tab_a = _tabicl(model_path=ckpt_a)
    tab_a.fit(d["X_tr"], d["y_away_tr"])

    ph_v = tab_h.predict(d["X_va"])
    pa_v = tab_a.predict(d["X_va"])
    ph_t = tab_h.predict(d["X_te"])
    pa_t = tab_a.predict(d["X_te"])
    print(f"  ICL done ({_time.time()-t0:.0f}s)")
    return eval_model(
        f"Q3_ft_{synth_name}", ph_v, pa_v, ph_t, pa_t, d,
    )


def run_q3(df, df_train, d):
    """Retouche-style input adapter on task-prior synth, then ICL with real data."""

    # 2014 WC used as adapter validation (identity guard)
    df_ft_val = df[
        (df["tournament"] == WC)
        & (df["date"] >= "2014-06-01")
        & (df["date"] < "2015-01-01")
    ]
    X_ft_val = np.asarray(df_ft_val[FEAT_COLS], dtype=np.float32)
    y_h_ft_val = np.asarray(df_ft_val["home_score"].clip(0, 5), dtype=np.int64)
    y_a_ft_val = np.asarray(df_ft_val["away_score"].clip(0, 5), dtype=np.int64)
    print(f"  Retouche validation: {len(df_ft_val)} matches (2014 WC)")

    synth_name = "footie_task_prior"
    print(f"\nQ3 [{synth_name}]: Retouche adapter on synth -> ICL on real")

    df_s = _sample_ranking_goal_synth(df_train)
    X_synth = df_s[FEAT_COLS].values.astype(np.float32)
    y_h_synth = df_s["home_score"].clip(0, 5).values.astype(np.int64)
    y_a_synth = df_s["away_score"].clip(0, 5).values.astype(np.int64)
    print(f"  Synth: {len(df_s)} samples ({synth_name})")

    synth_max = max(y_h_synth.max(), y_a_synth.max())
    y_h_val = np.clip(y_h_ft_val, 0, synth_max)
    y_a_val = np.clip(y_a_ft_val, 0, synth_max)

    run_dir = FT_OUT / synth_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n  Training Retouche adapters (home + away)...")
    t0 = _time.time()
    adapter_h = _retouche_train(
        synth_name, "home", X_synth, y_h_synth,
        X_ft_val, y_h_val, run_dir,
    )
    adapter_a = _retouche_train(
        synth_name, "away", X_synth, y_a_synth,
        X_ft_val, y_a_val, run_dir,
    )
    print(f"  Retouche training done ({_time.time()-t0:.0f}s)")

    # ICL with adapted inputs (frozen base model)
    print("\n  ICL with Retouche-adapted inputs (real context)...")
    t0 = _time.time()
    X_tr_h = _apply_adapter(adapter_h, d["X_tr"])
    X_tr_a = _apply_adapter(adapter_a, d["X_tr"])
    X_va_h = _apply_adapter(adapter_h, d["X_va"])
    X_va_a = _apply_adapter(adapter_a, d["X_va"])
    X_te_h = _apply_adapter(adapter_h, d["X_te"])
    X_te_a = _apply_adapter(adapter_a, d["X_te"])

    tab_h = _tabicl()
    tab_h.fit(X_tr_h, d["y_home_tr"])
    tab_a = _tabicl()
    tab_a.fit(X_tr_a, d["y_away_tr"])

    ph_v = tab_h.predict(X_va_h)
    pa_v = tab_a.predict(X_va_a)
    ph_t = tab_h.predict(X_te_h)
    pa_t = tab_a.predict(X_te_a)
    print(f"  ICL done ({_time.time()-t0:.0f}s)")
    return eval_model(
        f"Q3_retouche_{synth_name}", ph_v, pa_v, ph_t, pa_t, d,
    )


def runsky(args=("q3_ft", ), cluster="tk-footie-retouche-l4", gpus="l4:1", bucket="sky-tabpfn-bench", idle_mins: int=30, disk_size: int=256):
    """Launch a notebook stage on SkyPilot using repo-root .skyignore sync."""
    cmd = shlex.join(["nbs/26/2606_score_synth.py", *map(str, args)])
    idle_mins_val = None if idle_mins < 0 else idle_mins
    gpus_val = "" if gpus.lower() in {"", "none", "null"} else gpus
    py = ".venv/bin/python"
    setup_lines = [
        "python -m pip install -U uv",
        "uv venv .venv --python 3.10 --clear",
        f"uv pip install --python {py} --no-deps -e .",
        f"uv pip install --python {py} --reinstall torch --index-url https://download.pytorch.org/whl/cu126",
        f"uv pip install --python {py} 'tabicl[finetune]' huggingface_hub wandb scikit-learn pandas pyarrow numpy scipy tqdm",
    ]
    print(f"Launching SkyPilot task: {cmd}")
    print(f"  cluster={cluster} gpus={gpus_val or 'None'} bucket={bucket}")
    return sky_utils.exec(
        cmd,
        cluster=cluster,
        gpus=gpus_val,
        output=(bucket, ("data/out/2606_score_synth/.", "wandb/.")),
        setup_lines=setup_lines,
        run_prefix_lines=("export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",),
        idle_mins=idle_mins_val,
        secrets={k: os.environ[k] for k in ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")},
        disk_size=disk_size,
    )


def main():
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("stages", nargs="*", default=["all"])
    p.add_argument("--sky", nargs="?", const="1", default="")
    args = p.parse_args()
    if args.sky:  # expect --sky a=b,c=d
        cfg = dict(x.split("=", 1) for x in args.sky.split(",") if "=" in x)
        return runsky(next((s for s in args.stages if s != "all"), "q3"), **cfg)

    df, df_train, df_valid, df_test = load_data()
    d = get_arrays(df_train, df_valid, df_test)
    results = []
    _a = "all" in args.stages
    if _a or "q1" in args.stages: results += run_q1(df_train, d)
    if _a or "q2" in args.stages: results += run_q2(d)
    if _a or "q3" in args.stages: results += run_q3(df, df_train, d)
    if _a or "q3_ft" in args.stages: results += run_q3_ft(df, df_train, d)
    if (df_r := pd.DataFrame(results)).empty: return print("df empty")
    avail = df_r.columns.intersection([
        "method", "split", "acc_outcome", "exact_score",
        "n_draws_pred", "n_draws_actual",
    ])
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(df_r[avail].to_string(index=False, float_format="%.3f"))
    df_r.to_csv(OUT / "results.csv", index=False)
    print(f"\nSaved: {OUT / 'results.csv'}")


if __name__ == "__main__":
    main()
