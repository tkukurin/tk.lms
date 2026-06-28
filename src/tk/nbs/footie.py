"""Minimal match-outcome prediction using TM roster features + TabICL.

Usage as library:
    from tk.nbs.footie import load_match_dataset, FEAT_COLS, NAME_MAP, ROSTER_COLS

Usage standalone (prints train/valid shapes):
    python -m tk.nbs.footie
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tk import datadir

MATCHES_CSV = datadir / "international_football" / "results.csv"
TM_ROOT = datadir / "transfermarkt"

ROSTER_COLS = [
    "logMarketValue", "logTop11MarketValue", "benchShare",
    "starConcentration", "top3Share", "valueWeightedAge", "ageMean",
    "attackTilt", "attackDefenseRatio", "currentClubDiversity",
    "u23ValueShare", "domesticClubShare", "uefaClubValueShare",
]

HOME_COLS = [f"h_{c}" for c in ROSTER_COLS]
AWAY_COLS = [f"a_{c}" for c in ROSTER_COLS]
FEAT_COLS = ["importance", "neutral"] + HOME_COLS + AWAY_COLS

TOURNAMENT_IMPORTANCE = {
    "Friendly": 1, "FIFA World Cup qualification": 3,
    "FIFA World Cup": 5, "UEFA Euro": 4, "UEFA Euro qualification": 3,
    "Copa América": 4, "AFC Asian Cup": 3, "African Cup of Nations": 3,
    "CONCACAF Gold Cup": 3, "Confederations Cup": 3,
}

NAME_MAP = {
    "Brasil": "Brazil", "Alemanha": "Germany", "França": "France",
    "Espanha": "Spain", "Inglaterra": "England", "Holanda": "Netherlands",
    "Itália": "Italy", "Argentina": "Argentina", "Portugal": "Portugal",
    "Bélgica": "Belgium", "Croácia": "Croatia", "Uruguai": "Uruguay",
    "Colômbia": "Colombia", "México": "Mexico", "Suíça": "Switzerland",
    "Dinamarca": "Denmark", "Suécia": "Sweden", "Polônia": "Poland",
    "Sérvia": "Serbia", "Senegal": "Senegal", "Marrocos": "Morocco",
    "Japão": "Japan", "Coreia do Sul": "South Korea",
    "Austrália": "Australia", "Irã": "Iran", "Arábia Saudita": "Saudi Arabia",
    "Tunísia": "Tunisia", "Gana": "Ghana", "Camarões": "Cameroon",
    "Nigéria": "Nigeria", "Costa do Marfim": "Ivory Coast",
    "Argélia": "Algeria", "Egito": "Egypt", "EUA": "United States",
    "Costa Rica": "Costa Rica", "Canadá": "Canada", "Equador": "Ecuador",
    "Paraguai": "Paraguay", "Chile": "Chile", "Peru": "Peru",
    "Honduras": "Honduras", "Panamá": "Panama", "Catar": "Qatar",
    "País de Gales": "Wales", "Escócia": "Scotland",
    "República Tcheca": "Czech Republic", "Ucrânia": "Ukraine",
    "Rússia": "Russia", "Grécia": "Greece", "Eslováquia": "Slovakia",
    "Eslovênia": "Slovenia", "Bósnia e Herzegovina": "Bosnia and Herzegovina",
    "Togo": "Togo", "Angola": "Angola", "Trinidad e Tobago": "Trinidad and Tobago",
    "Sérvia e Montenegro": "Serbia and Montenegro",
    "República da Irlanda": "Republic of Ireland",
    "África do Sul": "South Africa", "Nova Zelândia": "New Zealand",
    "Coreia do Norte": "North Korea",
}


def _get_importance(t: str) -> int:
    for key, val in TOURNAMENT_IMPORTANCE.items():
        if key.lower() in t.lower():
            return val
    return 2


def load_tm_features() -> dict[str, dict[int, list[float]]]:
    """Load all TM team features into {english_name: {year: vector}}."""
    db: dict[str, dict[int, list[float]]] = {}
    for comp_dir in TM_ROOT.iterdir():
        if not comp_dir.is_dir() or comp_dir.name.startswith("_"):
            continue
        for year_dir in comp_dir.iterdir():
            if not year_dir.is_dir():
                continue
            tf_path = year_dir / "team_features.json"
            if not tf_path.exists():
                continue
            year_val = int(year_dir.name)
            teams = json.loads(tf_path.read_text())
            for _cid, feats in teams.items():
                name_pt = feats.get("name", "")
                name_en = NAME_MAP.get(name_pt, name_pt)
                vec = [feats.get(c, 0) or 0 for c in ROSTER_COLS]
                db.setdefault(name_en, {})[year_val] = vec
    return db


def get_team_vec(
    db: dict[str, dict[int, list[float]]], team: str, year: int,
) -> list[float] | None:
    """Get closest snapshot <= year for a team."""
    team_db = db.get(team)
    if team_db is None:
        return None
    candidates = [y for y in team_db if y <= year]
    if not candidates:
        return None
    return team_db[max(candidates)]


def load_match_dataset(
    tm_db: dict[str, dict[int, list[float]]] | None = None,
) -> pd.DataFrame:
    """Load international matches enriched with TM features for both teams.

    Returns a DataFrame with FEAT_COLS + outcome + metadata columns.
    Only includes post-2006 matches where both teams have TM data.
    """
    if tm_db is None:
        tm_db = load_tm_features()

    df = pd.read_csv(MATCHES_CSV)
    df = df[df["date"] >= "2006-01-01"].copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["importance"] = df["tournament"].apply(_get_importance)
    df["neutral"] = df["neutral"].astype(int)

    # Outcome (NaN for unplayed matches)
    has_score = df["home_score"].notna()
    df.loc[has_score, "home_score"] = df.loc[has_score, "home_score"].astype(int)
    df.loc[has_score, "away_score"] = df.loc[has_score, "away_score"].astype(int)
    df["outcome"] = np.where(
        ~has_score, np.nan,
        np.where(
            df["home_score"] > df["away_score"], 2,
            np.where(df["home_score"] == df["away_score"], 1, 0),
        ),
    )

    # Join TM features
    home_vecs, away_vecs, keep = [], [], []
    for _, row in df.iterrows():
        hv = get_team_vec(tm_db, row["home_team"], row["year"])
        av = get_team_vec(tm_db, row["away_team"], row["year"])
        if hv is not None and av is not None:
            home_vecs.append(hv)
            away_vecs.append(av)
            keep.append(True)
        else:
            home_vecs.append([0] * len(ROSTER_COLS))
            away_vecs.append([0] * len(ROSTER_COLS))
            keep.append(False)

    df = df[keep].copy()
    home_vecs = [v for v, k in zip(home_vecs, keep) if k]
    away_vecs = [v for v, k in zip(away_vecs, keep) if k]
    df[HOME_COLS] = pd.DataFrame(home_vecs, index=df.index)
    df[AWAY_COLS] = pd.DataFrame(away_vecs, index=df.index)
    return df


if __name__ == "__main__":
    db = load_tm_features()
    print(f"Teams: {len(db)}, snapshots: {sum(len(v) for v in db.values())}")
    df = load_match_dataset(db)
    print(f"Matched rows: {len(df)}")
    played = df[df["outcome"].notna()]
    print(f"With outcomes: {len(played)}")
