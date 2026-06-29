"""Score priors using Transfermarkt features plus cached FIFA and World Elo rankings."""
# pyright: reportArgumentType=false, reportReturnType=false
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss

from tk.nbs.footie import (
    ELO_RATINGS_ROOT,
    FIFA_RANKINGS_ROOT,
    TEST_CUT,
    TOURNAMENT_IMPORTANCE,
    VALID_CUT,
    load_elo_ratings,
    load_fifa_rankings,
    load_match_dataset,
    load_tm_features,
    normalize_team_name,
)

GOAL_FEATURES = (
    "log_mv_diff",
    "home_attack_tilt",
    "away_attack_tilt",
    "home_scorer_power",
    "away_scorer_power",
    "fifa_rating_diff_100",
    "fifa_rank_diff_100",
    "world_elo_diff_400",
    "world_elo_rank_diff_100",
    "neutral",
    "importance",
)


@dataclass(frozen=True)
class RankingRating:
    team: str
    date: pd.Timestamp
    rank: float
    rating: float
    source: str
    country_code: str = ""


@dataclass
class RankingProvider:
    rankings: pd.DataFrame
    source: str
    by_team: dict[str, pd.DataFrame] = field(init=False)

    def __post_init__(self) -> None:
        df = self.rankings.copy()
        if df.empty:
            raise ValueError(f"no {self.source} rankings loaded")
        df["team"] = df["team"].map(normalize_team_name)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notna()].sort_values(["team", "date"], kind="mergesort")
        self.rankings = df
        self.by_team = {
            team: rows.reset_index(drop=True)
            for team, rows in df.groupby("team", sort=False)
        }

    def at(self, team: str, date: str | pd.Timestamp) -> RankingRating | None:
        rows = self.by_team.get(normalize_team_name(team))
        if rows is None:
            return None
        idx = int(rows["date"].searchsorted(_timestamp(date), side="left")) - 1
        if idx < 0:
            return None
        row = rows.iloc[idx]
        return RankingRating(
            team=str(row["team"]),
            date=pd.Timestamp(row["date"]),
            rank=float(row["rank"]),
            rating=float(row["rating"]),
            source=self.source,
            country_code=str(row.get("country_code", "") or ""),
        )

    def attach(self, matches: pd.DataFrame) -> pd.DataFrame:
        out = matches.copy()
        prefix = self.source
        home_rank, away_rank, home_rating, away_rating = [], [], [], []
        for row in out.itertuples(index=False):
            home = self.at(getattr(row, "home_team"), getattr(row, "date"))
            away = self.at(getattr(row, "away_team"), getattr(row, "date"))
            home_rank.append(np.nan if home is None else home.rank)
            away_rank.append(np.nan if away is None else away.rank)
            home_rating.append(np.nan if home is None else home.rating)
            away_rating.append(np.nan if away is None else away.rating)
        out[f"home_{prefix}_rank"] = home_rank
        out[f"away_{prefix}_rank"] = away_rank
        out[f"home_{prefix}_rating"] = home_rating
        out[f"away_{prefix}_rating"] = away_rating
        out[f"{prefix}_rating_diff"] = (
            out[f"home_{prefix}_rating"] - out[f"away_{prefix}_rating"]
        )
        out[f"{prefix}_rank_diff"] = (
            out[f"away_{prefix}_rank"] - out[f"home_{prefix}_rank"]
        )
        return out


class FifaRankingProvider(RankingProvider):
    def __init__(self, rankings: pd.DataFrame):
        super().__init__(rankings=rankings, source="fifa")

    @classmethod
    def from_cache(cls, root: Path = FIFA_RANKINGS_ROOT) -> FifaRankingProvider:
        return cls(load_fifa_rankings(root=root))


class EloRankingProvider(RankingProvider):
    def __init__(self, rankings: pd.DataFrame):
        super().__init__(rankings=rankings, source="world_elo")

    @classmethod
    def from_cache(cls, root: Path = ELO_RATINGS_ROOT) -> EloRankingProvider:
        return cls(load_elo_ratings(root=root))


def _timestamp(date: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(date)
    return ts.tz_convert(None) if ts.tzinfo is not None else ts


def get_importance(tournament: str) -> int:
    t = str(tournament).lower()
    return next((v for k, v in TOURNAMENT_IMPORTANCE.items() if k.lower() in t), 2)


@dataclass(frozen=True)
class RankingGoalSamplerConfig:
    dispersion: float = 7.5
    draw_coupling: float = 0.10
    max_goals: int = 9
    poisson_alpha: float = 0.02


@dataclass
class RankingGoalSampler:
    config: RankingGoalSamplerConfig = field(default_factory=RankingGoalSamplerConfig)
    home_model: PoissonRegressor | None = None
    away_model: PoissonRegressor | None = None

    def fit(self, matches: pd.DataFrame) -> RankingGoalSampler:
        df = matches[matches["home_score"].notna() & matches["away_score"].notna()]
        X = self.design_matrix(df)
        self.home_model = PoissonRegressor(
            alpha=self.config.poisson_alpha,
            max_iter=1000,
        ).fit(X, np.asarray(df["home_score"], dtype=float))
        self.away_model = PoissonRegressor(
            alpha=self.config.poisson_alpha,
            max_iter=1000,
        ).fit(X, np.asarray(df["away_score"], dtype=float))
        return self

    def design_matrix(self, matches: pd.DataFrame) -> np.ndarray:
        df = matches.copy()
        feats = pd.DataFrame({
            "log_mv_diff": (
                _col(df, "h_logTop11MarketValue")
                - _col(df, "a_logTop11MarketValue")
            ),
            "home_attack_tilt": _col(df, "h_attackTilt"),
            "away_attack_tilt": _col(df, "a_attackTilt"),
            "home_scorer_power": _scorer_power(df, "h"),
            "away_scorer_power": _scorer_power(df, "a"),
            "fifa_rating_diff_100": _col(df, "fifa_rating_diff") / 100.0,
            "fifa_rank_diff_100": _col(df, "fifa_rank_diff") / 100.0,
            "world_elo_diff_400": _col(df, "world_elo_rating_diff") / 400.0,
            "world_elo_rank_diff_100": _col(df, "world_elo_rank_diff") / 100.0,
            "neutral": _col(df, "neutral"),
            "importance": _col(df, "importance"),
        })
        return (
            feats[list(GOAL_FEATURES)]
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
            .values.astype(np.float32)
        )

    def predict_lambdas(self, matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.home_model is None or self.away_model is None:
            raise RuntimeError("fit RankingGoalSampler before predicting")
        X = self.design_matrix(matches)
        home = self.home_model.predict(X)
        away = self.away_model.predict(X)
        return np.clip(home, 0.05, 6.5), np.clip(away, 0.05, 6.5)

    def sample_rows(
        self,
        matches: pd.DataFrame,
        n_sims: int = 1,
        random_state: int | np.random.Generator | None = None,
    ) -> pd.DataFrame:
        rng = _rng(random_state)
        home_lambda, away_lambda = self.predict_lambdas(matches)
        home_score, away_score = self.sample_score_arrays(
            home_lambda, away_lambda, n_sims, rng,
        )
        rows = []
        for sim in range(n_sims):
            part = matches[["date", "home_team", "away_team"]].copy()
            part["sim"] = sim
            part["home_score"] = home_score[sim].astype(int)
            part["away_score"] = away_score[sim].astype(int)
            part["outcome"] = np.where(
                home_score[sim] > away_score[sim],
                2,
                np.where(home_score[sim] == away_score[sim], 1, 0),
            )
            part["home_lambda"] = home_lambda
            part["away_lambda"] = away_lambda
            rows.append(part)
        return pd.concat(rows, ignore_index=True)

    def sample_score_arrays(
        self,
        home_lambda: np.ndarray,
        away_lambda: np.ndarray,
        n_sims: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        c = self.config
        home = _negbin(home_lambda, c.dispersion, n_sims, rng)
        away = _negbin(away_lambda, c.dispersion, n_sims, rng)
        closeness = np.exp(-np.abs(home_lambda - away_lambda))
        couple_p = np.clip(c.draw_coupling * closeness, 0.0, 0.35)
        mask = rng.random(size=home.shape) < couple_p[None, :]
        draw_score = np.rint((home + away) / 2.0).astype(int)
        home = np.where(mask, draw_score, home)
        away = np.where(mask, draw_score, away)
        return np.clip(home, 0, c.max_goals), np.clip(away, 0, c.max_goals)


def _col(df: pd.DataFrame, col: str) -> pd.Series:
    values = pd.Series(df[col], index=df.index)
    numeric = pd.Series(pd.to_numeric(values, errors="coerce"), index=df.index)
    return numeric.fillna(0).astype(float)


def _scorer_power(df: pd.DataFrame, prefix: str) -> pd.Series:
    top11 = _col(df, f"{prefix}_logTop11MarketValue")
    tilt = _col(df, f"{prefix}_attackTilt")
    return (top11 * (1.0 + tilt)).clip(lower=0.0)


def _negbin(
    mean: np.ndarray,
    dispersion: float,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    p = dispersion / (dispersion + np.asarray(mean, dtype=float))
    return rng.negative_binomial(dispersion, p, size=(n_sims, len(mean)))


def _rng(random_state: int | np.random.Generator | None) -> np.random.Generator:
    return (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )


def setup_backtest_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matches = load_match_dataset(load_tm_features())
    matches = matches[matches["outcome"].notna()].copy()
    matches["date"] = pd.to_datetime(matches["date"])
    matches["home_score"] = matches["home_score"].astype(int)
    matches["away_score"] = matches["away_score"].astype(int)
    matches = FifaRankingProvider.from_cache().attach(matches)
    matches = EloRankingProvider.from_cache().attach(matches)
    train = matches[matches["date"] < VALID_CUT].copy()
    test = matches[(matches["date"] >= VALID_CUT) & (matches["date"] < TEST_CUT)].copy()
    return matches, train, test


def backtest_sampler(
    sampler: RankingGoalSampler,
    test: pd.DataFrame,
    n_sims: int = 500,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(random_state)
    lh, la = sampler.predict_lambdas(test)
    hs, aw = sampler.sample_score_arrays(lh, la, n_sims, rng)
    outcomes = np.where(hs > aw, 2, np.where(hs == aw, 1, 0))
    probs = np.column_stack([
        (outcomes == 0).mean(axis=0),
        (outcomes == 1).mean(axis=0),
        (outcomes == 2).mean(axis=0),
    ])
    safe_probs = np.clip(probs, 1e-6, 1.0)
    safe_probs = safe_probs / safe_probs.sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1)
    y = test["outcome"].astype(int).values
    detail = test[["date", "home_team", "away_team", "home_score", "away_score", "outcome"]].copy()
    detail["pred_outcome"] = pred
    detail["p_away"] = probs[:, 0]
    detail["p_draw"] = probs[:, 1]
    detail["p_home"] = probs[:, 2]
    detail["lambda_home"] = lh
    detail["lambda_away"] = la
    detail["sim_home_mean"] = hs.mean(axis=0)
    detail["sim_away_mean"] = aw.mean(axis=0)
    metrics = {
        "n": float(len(test)),
        "acc_outcome": float(accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, safe_probs, labels=[0, 1, 2])),
        "actual_home_goals": float(test["home_score"].mean()),
        "sim_home_goals": float(hs.mean()),
        "actual_away_goals": float(test["away_score"].mean()),
        "sim_away_goals": float(aw.mean()),
        "actual_draw_rate": float((y == 1).mean()),
        "sim_draw_rate": float((outcomes == 1).mean()),
        "actual_home_win_rate": float((y == 2).mean()),
        "sim_home_win_rate": float((outcomes == 2).mean()),
    }
    return detail, metrics


def cmd_backtest(args: argparse.Namespace) -> None:
    matches, train, test = setup_backtest_data()
    if args.tournament:
        test = test[test["tournament"] == args.tournament].copy()
    sampler = RankingGoalSampler().fit(train)
    detail, metrics = backtest_sampler(
        sampler,
        test,
        n_sims=args.n_sims,
        random_state=args.seed,
    )
    print("DATA")
    print(f"  matches={len(matches)} train={len(train)} test={len(test)}")
    print("METRICS")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print("\nEYES")
    cols = [
        "date", "home_team", "away_team", "home_score", "away_score",
        "outcome", "pred_outcome", "p_home", "p_draw", "p_away",
        "lambda_home", "lambda_away",
    ]
    print(detail[cols].head(args.head).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "Football prior samplers").splitlines()[0],
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("backtest", help="Backtest sampler against ground truth")
    p.add_argument("--n-sims", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--head", type=int, default=16)
    p.add_argument("--tournament", default="FIFA World Cup")
    p.set_defaults(func=cmd_backtest)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
