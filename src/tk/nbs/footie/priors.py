"""Prior samplers for international football score simulation.

The Transfermarkt API/cache is used for roster/player state.
Elo provider below computes pre-match Elo from historical match results.
TODO: Ideally get historical FIFA rankings.
"""
# pyright: reportArgumentType=false, reportReturnType=false
from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss

from tk import datadir
from tk.nbs.footie import (
    NAME_MAP,
    TEST_CUT,
    TM_ROOT,
    TOURNAMENT_IMPORTANCE,
    VALID_CUT,
    iter_tm_year_dirs,
    load_match_dataset,
    load_tm_data,
    load_tm_features,
)

RESULTS_CSV = datadir / "international_football" / "results.csv"
BASE_URL = "https://tmapi-alpha.transfermarkt.technology/"


@dataclass(frozen=True)
class PlayerState:
    player_id: str
    name: str
    position_group: str
    market_value: float
    age: float
    national_caps: int = 0
    national_goals: int = 0
    goal_affinity: float = 0.0


@dataclass(frozen=True)
class TeamState:
    name: str
    year: int
    features: Mapping[str, float]
    players: tuple[PlayerState, ...] = ()

    @property
    def scorer_power(self) -> float:
        if self.players:
            weights = [p.goal_affinity for p in self.players]
            return float(np.sum(sorted(weights)[-5:]))
        top11 = float(self.features.get("logTop11MarketValue", 0.0) or 0.0)
        tilt = float(self.features.get("attackTilt", 0.0) or 0.0)
        return max(0.0, top11 * (1.0 + tilt))


@dataclass(frozen=True)
class MatchContext:
    date: pd.Timestamp
    home_team: str
    away_team: str
    neutral: int
    importance: float
    home_elo: float
    away_elo: float
    home_state: TeamState | None = None
    away_state: TeamState | None = None
    features: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoreSample:
    home_score: int
    away_score: int
    outcome: int
    home_lambda: float
    away_lambda: float
    home_scorers: tuple[str, ...] = ()
    away_scorers: tuple[str, ...] = ()


@dataclass
class TmJsonClient:
    base_url: str = BASE_URL
    timeout: float = 10.0
    headers: Mapping[str, str] = field(default_factory=lambda: {
        "Accept": "application/json",
        "Accept-Language": "en-US",
        "User-Agent": "Mozilla/5.0",
    })

    def fetch(self, endpoint: str) -> Any:
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        request = Request(url, headers=dict(self.headers))
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)["data"]

    def player_national_career(self, player_id: str) -> Any:
        return self.fetch(f"player/{player_id}/national-career-history")


@dataclass
class TransfermarktCareerCache:
    root: Path = TM_ROOT / "_player_national_career"
    client: TmJsonClient = field(default_factory=TmJsonClient)
    sleep_s: float = 0.0

    def get(self, player_id: str, fetch_missing: bool = False) -> dict[str, Any] | None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{player_id}.json"
        if path.exists():
            return json.loads(path.read_text())
        if not fetch_missing:
            return None
        data = self.client.player_national_career(player_id)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        if self.sleep_s:
            time.sleep(self.sleep_s)
        return data


def _mv(player: Mapping[str, Any]) -> float:
    return float(
        ((player.get("marketValueDetails") or {}).get("current") or {}).get("value")
        or 0.0
    )


def _age(player: Mapping[str, Any]) -> float:
    return float((player.get("lifeDates") or {}).get("age") or 0.0)


def _position_group(player: Mapping[str, Any]) -> str:
    return str((player.get("attributes") or {}).get("positionGroup") or "UNKNOWN")


def _senior_national_stats(career: Mapping[str, Any] | None) -> tuple[int, int]:
    if not career:
        return 0, 0
    rows = list(career.get("history", []))
    if not rows:
        return 0, 0
    current = [r for r in rows if r.get("careerState") == "CURRENT_NATIONAL_PLAYER"]
    row = max(current or rows, key=lambda r: int(r.get("gamesPlayed") or 0))
    return int(row.get("gamesPlayed") or 0), int(row.get("goalsScored") or 0)


def get_goal_affinity(
    player: Mapping[str, Any],
    career: Mapping[str, Any] | None = None,
) -> float:
    caps, goals = _senior_national_stats(career)
    value = _mv(player)
    age = _age(player)
    pos = _position_group(player)
    pos_bonus = {
        "FORWARD": 1.15,
        "MIDFIELDER": 0.45,
        "DEFENDER": -0.15,
        "GOALKEEPER": -3.0,
    }.get(pos, 0.0)
    gpc = goals / max(caps, 1)
    age_penalty = max(age - 31.0, 0.0) * 0.04
    return (
        math.log1p(goals)
        + 1.5 * gpc
        + 0.12 * math.log1p(value / 1_000_000.0)
        + pos_bonus
        - age_penalty
    )


def get_player_state(
    player: Mapping[str, Any],
    career: Mapping[str, Any] | None = None,
) -> PlayerState:
    caps, goals = _senior_national_stats(career)
    return PlayerState(
        player_id=str(player.get("id") or player.get("playerId") or ""),
        name=str(player.get("name") or player.get("displayName") or ""),
        position_group=_position_group(player),
        market_value=_mv(player),
        age=_age(player),
        national_caps=caps,
        national_goals=goals,
        goal_affinity=get_goal_affinity(player, career),
    )


@dataclass
class TransfermarktTeamStateProvider:
    states: dict[str, dict[int, TeamState]]

    @classmethod
    def from_cache(
        cls,
        root: Path = TM_ROOT,
        career_cache: TransfermarktCareerCache | None = None,
        fetch_player_careers: bool = False,
    ) -> "TransfermarktTeamStateProvider":
        states: dict[str, dict[int, TeamState]] = {}
        for year, year_dir in iter_tm_year_dirs(root):
            features_by_club, meta = load_tm_data(year_dir)
            squads, snapshots = meta["squads"], meta["snapshots"]
            for club_id, features in features_by_club.items():
                name = NAME_MAP.get(features.get("name", ""), features.get("name", ""))
                players = _get_roster_states(
                    club_id, squads, snapshots, career_cache, fetch_player_careers,
                )
                numeric = {
                    key: float(value)
                    for key, value in features.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
                states.setdefault(name, {})[year] = TeamState(
                    name=name, year=year, features=numeric, players=tuple(players),
                )
        return cls(states=states)

    def at(self, team: str, date: str | pd.Timestamp | int) -> TeamState | None:
        year = int(date) if isinstance(date, int) else pd.Timestamp(date).year
        team_states = self.states.get(team)
        if not team_states:
            return None
        candidates = [y for y in team_states if y <= year]
        if not candidates:
            return None
        return team_states[max(candidates)]


def _get_roster_states(
    club_id: str,
    squads: Mapping[str, Any],
    snapshots: Mapping[str, Any],
    career_cache: TransfermarktCareerCache | None,
    fetch_player_careers: bool,
) -> list[PlayerState]:
    squad = squads.get(str(club_id), {})
    player_ids = squad.get("playerIds", [])
    rows = []
    for player_id in player_ids:
        player = snapshots.get(str(player_id))
        if not player:
            continue
        career = None if career_cache is None else career_cache.get(
            str(player_id), fetch_missing=fetch_player_careers)
        rows.append(get_player_state(player, career))
    return rows


@dataclass(frozen=True)
class EloConfig:
    initial: float = 1500.0
    k: float = 28.0
    home_advantage: float = 65.0
    margin_scale: float = 0.55
    use_importance: bool = True


@dataclass
class HistoricalEloProvider:
    config: EloConfig = field(default_factory=EloConfig)
    ratings: dict[str, float] = field(default_factory=dict)
    history: dict[str, list[tuple[pd.Timestamp, float]]] = field(default_factory=dict)
    snapshots: pd.DataFrame = field(default_factory=pd.DataFrame)

    @classmethod
    def from_results(
        cls,
        results: pd.DataFrame,
        config: EloConfig | None = None,
    ) -> "HistoricalEloProvider":
        provider = cls(config=config or EloConfig())
        provider.fit(results)
        return provider

    def fit(self, results: pd.DataFrame) -> "HistoricalEloProvider":
        rows = []
        df = results.copy()
        df = df[df["home_score"].notna() & df["away_score"].notna()].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date", kind="mergesort")
        for idx, row in df.iterrows():
            home, away = str(row["home_team"]), str(row["away_team"])
            date = pd.Timestamp(row["date"])
            neutral = int(row.get("neutral", 0) or 0)
            rh, ra = self.rating(home), self.rating(away)
            rows.append({
                "source_index": idx,
                "date": date,
                "home_team": home,
                "away_team": away,
                "home_elo": rh,
                "away_elo": ra,
                "elo_diff": rh - ra,
                "elo_diff_home_adv": rh - ra + self.home_adv(neutral),
            })
            self.update(
                home=home,
                away=away,
                home_score=int(row["home_score"]),
                away_score=int(row["away_score"]),
                date=date,
                neutral=neutral,
                tournament=str(row.get("tournament", "")),
            )
        self.snapshots = pd.DataFrame(rows).set_index("source_index", drop=False)
        return self

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.config.initial)

    def rating_at(self, team: str, date: str | pd.Timestamp) -> float:
        date_ts = pd.Timestamp(date)
        rows = self.history.get(team, [])
        rating = self.config.initial
        for row_date, row_rating in rows:
            if row_date >= date_ts:
                break
            rating = row_rating
        return rating

    def attach(self, matches: pd.DataFrame) -> pd.DataFrame:
        out = matches.copy()
        home_elo, away_elo = [], []
        for row in out.itertuples(index=False):
            date = getattr(row, "date")
            home_elo.append(self.rating_at(getattr(row, "home_team"), date))
            away_elo.append(self.rating_at(getattr(row, "away_team"), date))
        out["home_elo"] = home_elo
        out["away_elo"] = away_elo
        out["elo_diff"] = out["home_elo"] - out["away_elo"]
        neutral = _series(out, "neutral").astype(int).values
        out["elo_diff_home_adv"] = out["elo_diff"] + np.where(
            neutral == 1,
            0.0,
            self.config.home_advantage,
        )
        return out

    def update(
        self,
        home: str,
        away: str,
        home_score: int,
        away_score: int,
        date: pd.Timestamp,
        neutral: int,
        tournament: str = "",
    ) -> None:
        rh, ra = self.rating(home), self.rating(away)
        expected = _logistic_elo(rh + self.home_adv(neutral) - ra)
        actual = 1.0 if home_score > away_score else 0.5 if home_score == away_score else 0.0
        gd = abs(home_score - away_score)
        margin = 1.0 + self.config.margin_scale * math.log1p(gd)
        k = self.config.k * margin * self.importance_multiplier(tournament)
        delta = k * (actual - expected)
        self.ratings[home] = rh + delta
        self.ratings[away] = ra - delta
        self.history.setdefault(home, []).append((date, self.ratings[home]))
        self.history.setdefault(away, []).append((date, self.ratings[away]))

    def home_adv(self, neutral: int) -> float:
        return 0.0 if neutral else self.config.home_advantage

    def importance_multiplier(self, tournament: str) -> float:
        if not self.config.use_importance:
            return 1.0
        importance = get_importance(tournament)
        return 0.55 + 0.16 * importance


def _logistic_elo(diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def get_importance(tournament: str) -> int:
    t = str(tournament).lower()
    return next((v for k, v in TOURNAMENT_IMPORTANCE.items() if k.lower() in t), 2)


@dataclass(frozen=True)
class EloPlayerGoalSamplerConfig:
    base_goal: float = 1.22
    elo_coef: float = 0.55
    market_coef: float = 0.035
    attack_coef: float = 0.22
    scorer_coef: float = 0.018
    home_goal_advantage: float = 0.08
    dispersion: float = 7.5
    draw_coupling: float = 0.10
    max_goals: int = 9
    poisson_alpha: float = 0.02


@dataclass
class EloPlayerGoalSampler:
    config: EloPlayerGoalSamplerConfig = field(default_factory=EloPlayerGoalSamplerConfig)
    team_states: TransfermarktTeamStateProvider | None = None
    home_model: PoissonRegressor | None = None
    away_model: PoissonRegressor | None = None
    feature_names: tuple[str, ...] = (
        "elo_diff_400",
        "log_mv_diff",
        "home_attack_tilt",
        "away_attack_tilt",
        "home_scorer_power",
        "away_scorer_power",
        "neutral",
        "importance",
    )

    def fit(self, matches: pd.DataFrame) -> "EloPlayerGoalSampler":
        df = matches[matches["home_score"].notna() & matches["away_score"].notna()].copy()
        X = self.design_matrix(df)
        self.home_model = PoissonRegressor(
            alpha=self.config.poisson_alpha,
            max_iter=1000,
        ).fit(X, df["home_score"].astype(float).values)
        self.away_model = PoissonRegressor(
            alpha=self.config.poisson_alpha,
            max_iter=1000,
        ).fit(X, df["away_score"].astype(float).values)
        return self

    def design_matrix(self, matches: pd.DataFrame) -> np.ndarray:
        df = matches.copy()
        if "importance" not in df:
            df["importance"] = df["tournament"].map(get_importance)
        if "home_elo" not in df or "away_elo" not in df:
            df["home_elo"] = 1500.0
            df["away_elo"] = 1500.0
        feats = pd.DataFrame({
            "elo_diff_400": (df["home_elo"] - df["away_elo"]) / 400.0,
            "log_mv_diff": _series(df, "h_logTop11MarketValue") - _series(df, "a_logTop11MarketValue"),
            "home_attack_tilt": _series(df, "h_attackTilt"),
            "away_attack_tilt": _series(df, "a_attackTilt"),
            "home_scorer_power": self.scorer_power_series(df, "home"),
            "away_scorer_power": self.scorer_power_series(df, "away"),
            "neutral": _series(df, "neutral"),
            "importance": _series(df, "importance"),
        })
        return feats[list(self.feature_names)].replace([np.inf, -np.inf], 0).fillna(0).values.astype(np.float32)

    def scorer_power_series(self, df: pd.DataFrame, side: str) -> pd.Series:
        if self.team_states is None:
            prefix = "h" if side == "home" else "a"
            top11 = _series(df, f"{prefix}_logTop11MarketValue")
            tilt = _series(df, f"{prefix}_attackTilt")
            return (top11 * (1.0 + tilt)).clip(lower=0.0)
        vals = []
        team_col = f"{side}_team"
        for row in df.itertuples(index=False):
            state = self.team_states.at(getattr(row, team_col), getattr(row, "date"))
            vals.append(0.0 if state is None else state.scorer_power)
        return pd.Series(vals, index=df.index, dtype=float)

    def predict_lambdas(self, matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = self.design_matrix(matches)
        if self.home_model is not None and self.away_model is not None:
            lh = self.home_model.predict(X)
            la = self.away_model.predict(X)
        else:
            lh, la = self._hand_lambdas(matches)
        return np.clip(lh, 0.05, 6.5), np.clip(la, 0.05, 6.5)

    def sample_rows(
        self,
        matches: pd.DataFrame,
        n_sims: int = 1,
        random_state: int | np.random.Generator | None = None,
    ) -> pd.DataFrame:
        rng = _rng(random_state)
        lh, la = self.predict_lambdas(matches)
        hs, aw = self._sample_score_arrays(lh, la, n_sims, rng)
        rows = []
        for sim in range(n_sims):
            out = np.where(hs[sim] > aw[sim], 2, np.where(hs[sim] == aw[sim], 1, 0))
            part = matches[["date", "home_team", "away_team"]].copy()
            part["sim"] = sim
            part["home_score"] = hs[sim].astype(int)
            part["away_score"] = aw[sim].astype(int)
            part["outcome"] = out.astype(int)
            part["home_lambda"] = lh
            part["away_lambda"] = la
            rows.append(part)
        return pd.concat(rows, ignore_index=True)

    def sample_context(
        self,
        context: MatchContext,
        random_state: int | np.random.Generator | None = None,
    ) -> ScoreSample:
        rng = _rng(random_state)
        df = pd.DataFrame([context_to_row(context)])
        lh, la = self.predict_lambdas(df)
        h, a = self._sample_score_arrays(lh, la, n_sims=1, rng=rng)
        hs, aw = int(h[0, 0]), int(a[0, 0])
        return ScoreSample(
            home_score=hs,
            away_score=aw,
            outcome=2 if hs > aw else 1 if hs == aw else 0,
            home_lambda=float(lh[0]),
            away_lambda=float(la[0]),
            home_scorers=self.sample_scorers(context.home_state, hs, rng),
            away_scorers=self.sample_scorers(context.away_state, aw, rng),
        )

    def sample_scorers(
        self,
        state: TeamState | None,
        goals: int,
        rng: np.random.Generator,
    ) -> tuple[str, ...]:
        if state is None or goals <= 0 or not state.players:
            return ()
        weights = np.array([max(p.goal_affinity, 0.01) for p in state.players], dtype=float)
        weights = weights / weights.sum()
        idx = rng.choice(len(state.players), size=goals, replace=True, p=weights)
        return tuple(state.players[i].name for i in idx)

    def _hand_lambdas(self, matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = pd.DataFrame(self.design_matrix(matches), columns=self.feature_names)
        c = self.config
        elo = X["elo_diff_400"].to_numpy(dtype=float)
        mv = X["log_mv_diff"].to_numpy(dtype=float)
        ht = X["home_attack_tilt"].to_numpy(dtype=float)
        at = X["away_attack_tilt"].to_numpy(dtype=float)
        hs = X["home_scorer_power"].to_numpy(dtype=float)
        aws = X["away_scorer_power"].to_numpy(dtype=float)
        neutral = X["neutral"].to_numpy(dtype=float)
        log_base = math.log(c.base_goal)
        log_h = (
            log_base
            + c.home_goal_advantage * (1.0 - neutral)
            + c.elo_coef * elo
            + c.market_coef * mv
            + c.attack_coef * (ht - at)
            + c.scorer_coef * (hs - aws)
        )
        log_a = (
            log_base
            - c.elo_coef * elo
            - c.market_coef * mv
            + c.attack_coef * (at - ht)
            + c.scorer_coef * (aws - hs)
        )
        return np.exp(log_h), np.exp(log_a)

    def _sample_score_arrays(
        self,
        home_lambda: np.ndarray,
        away_lambda: np.ndarray,
        n_sims: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        c = self.config
        h = _negbin(home_lambda, c.dispersion, n_sims, rng)
        a = _negbin(away_lambda, c.dispersion, n_sims, rng)
        closeness = np.exp(-np.abs(home_lambda - away_lambda))
        couple_p = np.clip(c.draw_coupling * closeness, 0.0, 0.35)
        mask = rng.random(size=h.shape) < couple_p[None, :]
        low = np.rint((h + a) / 2.0).astype(int)
        h = np.where(mask, low, h)
        a = np.where(mask, low, a)
        return np.clip(h, 0, c.max_goals), np.clip(a, 0, c.max_goals)


def _series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df:
        values = pd.Series(df[col], index=df.index)
        numeric = pd.Series(pd.to_numeric(values, errors="coerce"), index=df.index)
        return numeric.fillna(default).astype(float)
    return pd.Series(default, index=df.index, dtype=float)


def _negbin(
    mean: np.ndarray,
    dispersion: float,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    mean = np.asarray(mean, dtype=float)
    if dispersion <= 0:
        return rng.poisson(mean, size=(n_sims, len(mean)))
    p = dispersion / (dispersion + mean)
    return rng.negative_binomial(dispersion, p, size=(n_sims, len(mean)))


def _rng(random_state: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def context_to_row(context: MatchContext) -> dict[str, Any]:
    row: dict[str, Any] = dict(context.features)
    row.update({
        "date": context.date,
        "home_team": context.home_team,
        "away_team": context.away_team,
        "neutral": context.neutral,
        "importance": context.importance,
        "home_elo": context.home_elo,
        "away_elo": context.away_elo,
    })
    if context.home_state is not None:
        row.update({f"h_{k}": v for k, v in context.home_state.features.items()})
    if context.away_state is not None:
        row.update({f"a_{k}": v for k, v in context.away_state.features.items()})
    return row


def load_results(path: Path = RESULTS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = df["neutral"].astype(int)
    return df


def setup_backtest_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = load_results()
    elo = HistoricalEloProvider.from_results(results)
    matches = load_match_dataset(load_tm_features())
    matches = matches[matches["outcome"].notna()].copy()
    matches["date"] = pd.to_datetime(matches["date"])
    matches["home_score"] = matches["home_score"].astype(int)
    matches["away_score"] = matches["away_score"].astype(int)
    matches = elo.attach(matches)
    train = matches[matches["date"] < VALID_CUT].copy()
    test = matches[(matches["date"] >= VALID_CUT) & (matches["date"] < TEST_CUT)].copy()
    return matches, train, test


def backtest_sampler(
    sampler: EloPlayerGoalSampler,
    test: pd.DataFrame,
    n_sims: int = 500,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(random_state)
    lh, la = sampler.predict_lambdas(test)
    hs, aw = sampler._sample_score_arrays(lh, la, n_sims, rng)
    outcomes = np.where(hs > aw, 2, np.where(hs == aw, 1, 0))
    probs = np.column_stack([
        (outcomes == 0).mean(axis=0),
        (outcomes == 1).mean(axis=0),
        (outcomes == 2).mean(axis=0),
    ])
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
        "log_loss": float(log_loss(y, np.clip(probs, 1e-6, 1.0), labels=[0, 1, 2])),
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
    sampler = EloPlayerGoalSampler()
    if args.fit:
        sampler.fit(train)
    detail, metrics = backtest_sampler(
        sampler,
        test,
        n_sims=args.n_sims,
        random_state=args.seed,
    )
    print("DATA")
    print(f"  matches={len(matches)} train={len(train)} test={len(test)} fit={args.fit}")
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
    description = (__doc__ or "Football prior samplers").splitlines()[0]
    parser = argparse.ArgumentParser(description=description)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("backtest", help="Backtest sampler against ground truth")
    p.add_argument("--n-sims", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--head", type=int, default=16)
    p.add_argument("--tournament", default="FIFA World Cup")
    p.add_argument("--fit", action=argparse.BooleanOptionalAction, default=True)
    p.set_defaults(func=cmd_backtest)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
