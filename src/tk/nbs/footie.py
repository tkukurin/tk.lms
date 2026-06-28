"""TM API client, roster feature extraction, and match dataset for TabICL.

See 538's [pele](https://www.natesilver.net/p/pele-methodology) for infos on chosen features.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partialmethod
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from tk import datadir

CacheItem = tuple[float, Any]

BASE_URL = "https://tmapi-alpha.transfermarkt.technology/"
DEFAULT_TIMEOUT = 5.0
DEFAULT_TTL = 300.0
TMAPI_BATCH_SIZE = 100
DEFAULT_HEADERS = {
    "Accept": "application/json",
    "Accept-Language": "pt-BR",
    "User-Agent": "Mozilla/5.0",
}

WORLD_CUP_YEARS = (2006, 2010, 2014, 2018, 2022, 2026)
WORLD_CUP_START_DATES = {
    2006: "2006-06-09", 2010: "2010-06-11", 2014: "2014-06-12",
    2018: "2018-06-14", 2022: "2022-11-20", 2026: "2026-06-11",
}

COMPETITIONS = {
    "FIWC": WORLD_CUP_START_DATES,
    "EURO": {
        2008: "2008-06-07", 2012: "2012-06-08", 2016: "2016-06-10",
        2021: "2021-06-11", 2024: "2024-06-14",
    },
    "COPA": {
        2007: "2007-06-26", 2011: "2011-07-01", 2015: "2015-06-11",
        2016: "2016-06-03", 2019: "2019-06-14", 2021: "2021-06-13",
        2024: "2024-06-20",
    },
    "AFCN": {
        2008: "2008-01-20", 2010: "2010-01-10", 2012: "2012-01-21",
        2013: "2013-01-19", 2015: "2015-01-17", 2017: "2017-01-14",
        2019: "2019-06-21", 2022: "2022-01-09", 2024: "2024-01-13",
    },
    "AFAC": {
        2007: "2007-07-07", 2011: "2011-01-07", 2015: "2015-01-09",
        2019: "2019-01-05", 2023: "2023-01-12",
    },
    "GOCU": {
        2007: "2007-06-06", 2009: "2009-07-03", 2011: "2011-06-05",
        2013: "2013-07-07", 2015: "2015-07-07", 2017: "2017-07-07",
        2019: "2019-06-15", 2021: "2021-07-10", 2023: "2023-06-24",
    },
}

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
FEAT_COLS = ["date_ordinal", "importance", "neutral"] + HOME_COLS + AWAY_COLS

VALID_CUT = "2018-01-01"
TEST_CUT = "2026-06-01"

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
    "Albânia": "Albania",
    "Bahrein": "Bahrain",
    "Benim": "Benin",
    "Bermudas": "Bermuda",
    "Bolívia": "Bolivia",
    "Burquina Faso": "Burkina Faso",
    "Burúndi": "Burundi",
    "Cabo Verde": "Cape Verde",
    "Comores": "Comoros",
    "Emirados Árabes Unidos": "United Arab Emirates",
    "Estados Unidos": "United States",
    "Etiópia": "Ethiopia",
    "Filipinas": "Philippines",
    "Finlândia": "Finland",
    "Gabão": "Gabon",
    "Geórgia": "Georgia",
    "Granada": "Grenada",
    "Guadalupe": "Guadeloupe",
    "Guiana": "Guyana",
    "Guiana Francesa": "French Guiana",
    "Guiné": "Guinea",
    "Guiné Equatorial": "Equatorial Guinea",
    "Guiné-Bissau": "Guinea-Bissau",
    "Gâmbia": "Gambia",
    "Hungria": "Hungary",
    "Indonésia": "Indonesia",
    "Iraque": "Iraq",
    "Irlanda do Norte": "Northern Ireland",
    "Islândia": "Iceland",
    "Iémen": "Yemen",
    "Jordânia": "Jordan",
    "Líbano": "Lebanon",
    "Líbia": "Libya",
    "Macedônia do Norte": "North Macedonia",
    "Madagáscar": "Madagascar",
    "Malaui": "Malawi",
    "Malásia": "Malaysia",
    "Martinica": "Martinique",
    "Mauritânia": "Mauritania",
    "Moçambique": "Mozambique",
    "Máli": "Mali",
    "Namíbia": "Namibia",
    "Nicarágua": "Nicaragua",
    "Noruega": "Norway",
    "Níger": "Niger",
    "Omã": "Oman",
    "Palestina": "Palestine",
    "Quirguistão": "Kyrgyzstan",
    "Quénia": "Kenya",
    "República Democrática do Congo": "DR Congo",
    "República do Congo": "Congo",
    "Romênia": "Romania",
    "Serra Leoa": "Sierra Leone",
    "Sudão": "Sudan",
    "São Cristóvão e Névis": "Saint Kitts and Nevis",
    "Síria": "Syria",
    "Tailândia": "Thailand",
    "Tajiquistão": "Tajikistan",
    "Tanzânia": "Tanzania",
    "Tchéquia": "Czech Republic",
    "Trindade e Tobago": "Trinidad and Tobago",
    "Turquemenistão": "Turkmenistan",
    "Turquia": "Turkey",
    "Uzbequistão": "Uzbekistan",
    "Vietnã": "Vietnam",
    "Zimbábue": "Zimbabwe",
    "Zâmbia": "Zambia",
    "Áustria": "Austria",
    "Índia": "India",
    "Curaçau": "Curaçao",
}

WC_FINAL_RANK = {
    2006: ["Itália", "França", "Alemanha", "Portugal"],
    2010: ["Espanha", "Holanda", "Alemanha", "Uruguai"],
    2014: ["Alemanha", "Argentina", "Holanda", "Brasil"],
    2018: ["França", "Croácia", "Bélgica", "Inglaterra"],
    2022: ["Argentina", "França", "Croácia", "Marrocos"],
}


@dataclass(frozen=True)
class Endpoint:
    path: str
    path_args: tuple[str, ...] = ()
    required_query: tuple[str, ...] = ()
    optional_query: tuple[str, ...] = ()
    ids_query: bool = False


ENDPOINTS = {
    "attributes": Endpoint("attributes"),
    "tmsearch": Endpoint("quick-search", required_query=("term",)),
    "get_player_profile": Endpoint("player/{player_id}", ("player_id",)),
    "get_players_info": Endpoint("players", ids_query=True),
    "get_player_gallery": Endpoint("player/{player_id}/gallery", ("player_id",)),
    "get_player_performance": Endpoint("player/{player_id}/performance", ("player_id",), optional_query=("season",)),
    "get_player_injuries": Endpoint("player/{player_id}/injury", ("player_id",)),
    "get_player_market_value_history": Endpoint("player/{player_id}/market-value-history", ("player_id",)),
    "get_player_national_career": Endpoint("player/{player_id}/national-career-history", ("player_id",)),
    "get_player_transfer_history": Endpoint("transfer/history/player/{player_id}", ("player_id",)),
    "get_players_performance": Endpoint("players/performance"),
    "get_club_info": Endpoint("club/{club_id}", ("club_id",)),
    "get_clubs_info": Endpoint("clubs", ids_query=True),
    "get_club_squad": Endpoint("club/{club_id}/squad", ("club_id",)),
    "get_club_stadium": Endpoint("club/{club_id}/stadium", ("club_id",)),
    "get_club_transfer_history": Endpoint("transfer/history/club/{club_id}", ("club_id",)),
    "get_competition_info": Endpoint("competition/{code}", ("code",)),
    "get_competition_table": Endpoint("competition/{code}/table", ("code",)),
    "get_competitions_info": Endpoint("competitions", ids_query=True),
    "get_game": Endpoint("game/{game_id}", ("game_id",)),
    "get_game_live_detail": Endpoint("game/{game_id}/live-detail", ("game_id",)),
    "get_referee_profile": Endpoint("referee/{referee_id}", ("referee_id",)),
    "get_referees_info": Endpoint("referees", ids_query=True),
    "get_coach_profile": Endpoint("coach/{coach_id}", ("coach_id",)),
    "get_coaches_info": Endpoint("coaches", ids_query=True),
    "get_stadium_info": Endpoint("stadium/{stadium_id}", ("stadium_id",)),
}


@dataclass
class TmClient:
    base_url: str = BASE_URL
    timeout: float = DEFAULT_TIMEOUT
    ttl: float = DEFAULT_TTL
    use_cache: bool = True
    headers: dict[str, str] = field(default_factory=lambda: DEFAULT_HEADERS.copy())
    cache: dict[str, CacheItem] = field(default_factory=dict)

    def fetch(self, endpoint: str) -> Any:
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        now = time.monotonic()
        if self.use_cache and (hit := self.cache.get(url)):
            expires_at, data = hit
            if now < expires_at:
                return data
        request = Request(url, headers=self.headers)
        with urlopen(request, timeout=self.timeout) as response:
            data = json.load(response)
        if self.use_cache:
            self.cache[url] = (now + self.ttl, data)
        return data

    def call(self, name: str, *values: Any, **params: Any) -> Any:
        assert name in ENDPOINTS, f"{name=}\n{sorted(ENDPOINTS)=}"
        spec = ENDPOINTS[name]
        if values:
            names = [*spec.path_args, *spec.required_query, *(["ids"] if spec.ids_query else []), *spec.optional_query]
            params = dict(zip(names, values, strict=True)) | params
        path_args = {k: quote(str(params[k]), safe="") for k in spec.path_args}
        endpoint = spec.path.format(**path_args)
        query_items: list[tuple[str, Any]] = []
        if spec.ids_query:
            query_items.extend(("ids[]", value) for value in params["ids"])
        for key in (*spec.required_query, *spec.optional_query):
            if (value := params.get(key)) is not None:
                query_items.append((key, value))
        if query_items:
            endpoint = f"{endpoint}?{urlencode(query_items, doseq=True)}"
        return self.fetch(endpoint)

    attributes = partialmethod(call, "attributes")
    tmsearch = partialmethod(call, "tmsearch")
    get_player_profile = partialmethod(call, "get_player_profile")
    get_players_info = partialmethod(call, "get_players_info")
    get_player_market_value_history = partialmethod(call, "get_player_market_value_history")
    get_club_info = partialmethod(call, "get_club_info")
    get_clubs_info = partialmethod(call, "get_clubs_info")
    get_club_squad = partialmethod(call, "get_club_squad")
    get_competition_info = partialmethod(call, "get_competition_info")
    get_competition_table = partialmethod(call, "get_competition_table")


def _ratio(n, d): return n / d if d else None

def _club_id(player):
    a = player.get("clubAssignments", [])
    c = next((x for x in a if x.get("type") == "current"), None)
    return str(c["clubId"]) if c else None

def _mv(player):
    return (player.get("marketValueDetails") or {}).get(
        "current", {}).get("value") or 0

def _mv_at(history, cutoff):
    rows = [r for r in history.get("history", [])
        if r["marketValue"]["determined"] <= cutoff]
    return max(rows, key=lambda r: r["marketValue"]["determined"],
        default=None)


def get_player_snapshot(profile: dict[str, Any], history: dict[str, Any], cutoff: str) -> dict[str, Any]:
    row = _mv_at(history, cutoff)
    value = row["marketValue"]["value"] if row else 0
    age = row["age"] if row else profile.get("lifeDates", {}).get("age")
    club_id = str(row["clubId"]) if row else None
    determined = None if not row else row["marketValue"]["determined"]
    return {
        "id": profile["id"],
        "name": profile.get("name"),
        "displayName": profile.get("displayName"),
        "lifeDates": {"age": age},
        "attributes": profile.get("attributes", {}),
        "marketValueDetails": {"current": {"value": value, "determined": determined}},
        "clubAssignments": ([] if not club_id else [{"type": "current", "clubId": club_id}]),
        "missingMarketValueAtCutoff": row is None,
    }


def get_club_flags(
    current_id: str | None, country_id: int | None,
    current_clubs: dict[str, dict[str, Any]], country_confed: dict[int, int],
) -> tuple[bool, bool, bool]:
    if current_id is None:
        return False, False, False
    club = current_clubs.get(current_id, {})
    club_country = club.get("baseDetails", {}).get("countryId")
    is_domestic = club_country == country_id
    is_uefa = country_confed.get(club_country) == 6
    return True, is_domestic, is_uefa


def get_value_features(
    values: Sequence[int], ages: Sequence[int | float],
    age_value: int | float, position_values: dict[str, int],
) -> dict[str, Any]:
    values = sorted(values)
    total = sum(values)
    top11 = sum(values[-11:])
    top23 = sum(values[-23:])
    age_mean = ratio(sum(ages), len(ages)) if ages else None
    age_std = None
    if age_mean is not None:
        age_var = sum((age - age_mean) ** 2 for age in ages)
        age_std = math.sqrt(age_var / len(ages))
    mid = position_values.get("MIDFIELDER", 0)
    attack = position_values.get("FORWARD", 0) + 0.5 * mid
    defense = position_values.get("DEFENDER", 0) + position_values.get("GOALKEEPER", 0) + 0.5 * mid
    return {
        "marketValueEur": total,
        "logMarketValue": math.log1p(total),
        "top11MarketValueEur": top11,
        "logTop11MarketValue": math.log1p(top11),
        "top23MarketValueEur": top23,
        "logTop23MarketValue": math.log1p(top23),
        "benchValueEur": top23 - top11,
        "benchShare": ratio(top23 - top11, top23),
        "starConcentration": ratio(values[-1], total) if values else None,
        "top3Share": ratio(sum(values[-3:]), total),
        "top5Share": ratio(sum(values[-5:]), total),
        "valueWeightedAge": ratio(age_value, total),
        "ageMean": age_mean,
        "ageStd": age_std,
        "positionValueEur": dict(sorted(position_values.items())),
        "attackValueEur": attack,
        "defenseValueEur": defense,
        "attackTilt": ratio(attack - defense, total),
        "attackDefenseRatio": ratio(attack, defense),
        "gkValueShare": ratio(position_values.get("GOALKEEPER", 0), total),
        "defValueShare": ratio(position_values.get("DEFENDER", 0), total),
        "midValueShare": ratio(mid, total),
        "fwdValueShare": ratio(position_values.get("FORWARD", 0), total),
    }


def get_roster_features(
    roster: Sequence[dict[str, Any]], squad: dict[str, Any],
    country_id: int | None, current_clubs: dict[str, dict[str, Any]],
    country_confed: dict[int, int],
) -> dict[str, Any]:
    values, ages, age_value = [], [], 0
    position_values: dict[str, int] = {}
    current_ids = set()
    domestic_count = current_count = 0
    domestic_value = uefa_value = u23_value = u25_value = over30_value = 0
    zero_count = missing_count = 0

    for player in roster:
        value = _mv(player)
        age = player.get("lifeDates", {}).get("age") or 0
        group = player.get("attributes", {}).get("positionGroup") or "UNKNOWN"
        current_id = _club_id(player)
        has_club, domestic, uefa = get_club_flags(current_id, country_id, current_clubs, country_confed)

        values.append(value)
        ages.append(age)
        age_value += age * value
        position_values[group] = position_values.get(group, 0) + value
        current_ids.add(current_id or "")
        current_count += int(has_club)
        domestic_count += int(domestic)
        domestic_value += value * int(domestic)
        uefa_value += value * int(uefa)
        missing_count += int(player.get("missingMarketValueAtCutoff", False))
        zero_count += int(not value)
        u23_value += value * int(age < 23)
        u25_value += value * int(age < 25)
        over30_value += value * int(age > 30)

    total = sum(values)
    roster_share_feats = {
        "u23ValueShare": ratio(u23_value, total),
        "u25ValueShare": ratio(u25_value, total),
        "over30ValueShare": ratio(over30_value, total),
        "domesticClubShare": ratio(domestic_count, current_count),
        "domesticValueShare": ratio(domestic_value, total),
        "uefaClubValueShare": ratio(uefa_value, total),
    }

    captain_feats = _get_captain_features(roster, squad)
    return get_value_features(values, ages, age_value, position_values) | {
        "squadCount": len(roster),
        "currentClubDiversity": len(current_ids - {""}),
        "squadMissingValueCount": missing_count,
        "squadZeroValueCount": zero_count,
    } | roster_share_feats | captain_feats


def _get_captain_features(roster: Sequence[dict[str, Any]], squad: dict[str, Any]) -> dict[str, Any]:
    captain_ids = {p["playerId"] for p in squad.get("squad", []) if p.get("isCaptain")}
    players_by_id = {player["id"]: player for player in roster}
    captain = next((players_by_id.get(pid) for pid in captain_ids), None)
    if not captain:
        return {"captainMarketValueEur": None, "captainAge": None, "captainPosition": None}
    captain_value = (captain.get("marketValueDetails") or {}).get("current", {}).get("value")
    return {
        "captainMarketValueEur": captain_value,
        "captainAge": captain.get("lifeDates", {}).get("age"),
        "captainPosition": captain.get("attributes", {}).get("positionGroup"),
    }


def get_batched_rows(ids: Sequence[str], fetcher: Callable[..., Any]) -> dict[str, Any]:
    result = {}
    for i in range(0, len(ids), TMAPI_BATCH_SIZE):
        batch = ids[i : i + TMAPI_BATCH_SIZE]
        data = fetcher(ids=batch)["data"]
        for item in data:
            result[str(item["id"])] = item
    return result


def fetch_player_histories(client: TmClient, player_ids: Sequence[str], cache_dir: Path) -> dict[str, Any]:
    histories = {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    for player_id in player_ids:
        path = cache_dir / f"{player_id}.json"
        if path.exists():
            histories[player_id] = json.loads(path.read_text())
            continue
        data = client.get_player_market_value_history(player_id)["data"]
        histories[player_id] = data
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return histories


def fetch_squads(client: TmClient, club_ids: Sequence[str], season: int, out: Path) -> dict[str, Any]:
    squads = {}
    (out / "squads").mkdir(parents=True, exist_ok=True)
    for club_id in club_ids:
        squad = client.fetch(f"club/{club_id}/squad?season={season}")["data"]
        squads[club_id] = squad
        (out / "squads" / f"{club_id}.json").write_text(json.dumps(squad, ensure_ascii=False, indent=2))
    (out / "squads.json").write_text(json.dumps(squads, ensure_ascii=False, indent=2))
    return squads


def fetch_comp_year(
    client: TmClient, comp_id: str, year: int, cutoff: str,
    root: Path = TM_ROOT,
) -> dict[str, Any]:
    """Fetch roster features for any international competition/year."""
    season = year - 1
    out = root / comp_id.lower() / str(year)
    out.mkdir(parents=True, exist_ok=True)

    if (out / "team_features.json").exists():
        return json.loads((out / "summary.json").read_text())

    table = client.fetch(f"competition/{comp_id}/table?season={season}")["data"]
    (out / "table.json").write_text(json.dumps(table, ensure_ascii=False, indent=2))
    rows = [club for group in table.get("tables", []) for club in group.get("clubs", [])]
    club_ids = [row["clubId"] for row in rows]
    club_rows = {row["clubId"]: row for row in rows}

    clubs = get_batched_rows(club_ids, client.get_clubs_info)
    (out / "clubs.json").write_text(json.dumps(clubs, ensure_ascii=False, indent=2))

    squads = fetch_squads(client, club_ids, season, out)
    player_ids = sorted({pid for squad in squads.values() for pid in squad["playerIds"]}, key=int)
    players = get_batched_rows(player_ids, client.get_players_info)
    player_ids = sorted(players, key=int)
    (out / "player_profiles_current.json").write_text(json.dumps(players, ensure_ascii=False, indent=2))

    histories = fetch_player_histories(client, player_ids, root / "_player_market_value_history")
    snapshots = {
        pid: get_player_snapshot(players[pid], histories[pid], cutoff) for pid in player_ids
    }
    (out / "player_snapshots.json").write_text(json.dumps(snapshots, ensure_ascii=False, indent=2))

    snap_club_ids = sorted({cid for p in snapshots.values() if (cid := _club_id(p))}, key=int)
    current_clubs = get_batched_rows(snap_club_ids, client.get_clubs_info)
    (out / "clubs_at_value.json").write_text(json.dumps(current_clubs, ensure_ascii=False, indent=2))

    attributes = client.attributes()["data"]
    country_confed = {c["id"]: c["confederationId"] for c in attributes["countries"]}

    teams = {}
    for club_id, squad in squads.items():
        roster = [snapshots[pid] for pid in squad["playerIds"]]
        country_id = clubs[club_id].get("baseDetails", {}).get("countryId")
        teams[club_id] = get_roster_features(
            roster, squad, country_id, current_clubs, country_confed,
        ) | {"name": clubs[club_id].get("name"), "table": club_rows.get(club_id)}
    (out / "team_features.json").write_text(json.dumps(teams, ensure_ascii=False, indent=2))

    summary = {
        "comp": comp_id, "year": year, "season": season, "cutoff": cutoff,
        "teams": len(teams), "players": len(players),
        "missingMarketValues": sum(p["missingMarketValueAtCutoff"] for p in snapshots.values()),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def fetch_all(client: TmClient) -> None:
    for comp_id, years in COMPETITIONS.items():
        for year, cutoff in years.items():
            t0 = time.time()
            try:
                summary = fetch_comp_year(client, comp_id, year, cutoff)
                summary["seconds"] = round(time.time() - t0, 1)
                print(json.dumps(summary, ensure_ascii=False), flush=True)
            except Exception as e:
                print(f"ERROR {comp_id}/{year}: {e}", flush=True)


def analyze(root: Path = TM_ROOT / "fiwc") -> None:
    """Check if top11MarketValueEur correlates with WC finishing position."""
    rows = []
    for year in WORLD_CUP_YEARS:
        if not (path := root / str(year) / "team_features.json").exists():
            continue
        if not (podium := WC_FINAL_RANK.get(year)):
            continue
        teams = json.loads(path.read_text())
        ranked = sorted(teams.values(), key=lambda t: -t["top11MarketValueEur"])
        names = [t["name"] for t in ranked]
        winner = podium[0]
        winner_rank = names.index(winner) + 1 if winner in names else None
        top4_in_top8 = sum(1 for p in podium if p in names[:8])
        top4_in_top16 = sum(1 for p in podium if p in names[:16])
        rows.append({"year": year, "winnerValueRank": winner_rank, "top4inTop8value": top4_in_top8})
        print(f"\n{year}")
        print(f"  Winner: {winner} (value rank #{winner_rank}/{len(teams)})")
        print(f"  Podium in top-8 by value: {top4_in_top8}/4")
        print(f"  Podium in top-16 by value: {top4_in_top16}/4")
        print(f"  Top 5 by value: {names[:5]}")
        print(f"  Actual podium:  {podium}")
    if rows:
        print(f"\nSummary across {len(rows)} tournaments")
        print(f"  Avg winner value rank: {sum(r['winnerValueRank'] for r in rows) / len(rows):.1f}")
        print(f"  Avg podium in top-8: {sum(r['top4inTop8value'] for r in rows) / len(rows):.1f}/4")


def _get_importance(t: str) -> int:
    for key, val in TOURNAMENT_IMPORTANCE.items():
        if key.lower() in t.lower():
            return val
    return 2


def load_tm_features() -> dict[str, dict[int, list[float]]]:
    db: dict[str, dict[int, list[float]]] = {}
    paths = [
        (int(yd.name), tf)
        for cd in TM_ROOT.iterdir()
        if cd.is_dir() and not cd.name.startswith("_")
        for yd in cd.iterdir()
        if yd.is_dir() and (tf := yd / "team_features.json").exists()
    ]
    for year_val, tf_path in paths:
        teams = json.loads(tf_path.read_text())
        for _cid, feats in teams.items():
            name_en = NAME_MAP.get(feats.get("name", ""),
                feats.get("name", ""))
            vec = [feats.get(c, 0) or 0 for c in ROSTER_COLS]
            db.setdefault(name_en, {})[year_val] = vec
    return db


def get_team_vec(db: dict[str, dict[int, list[float]]], team: str, year: int) -> list[float] | None:
    """Get closest snapshot <= year for a team."""
    team_db = db.get(team)
    if team_db is None:
        return None
    candidates = [y for y in team_db if y <= year]
    if not candidates:
        return None
    return team_db[max(candidates)]


def load_match_dataset(tm_db: dict[str, dict[int, list[float]]]) -> pd.DataFrame:
    df = pd.read_csv(MATCHES_CSV)
    df = df[df["date"] >= "2006-01-01"].copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["importance"] = df["tournament"].apply(_get_importance)
    df["neutral"] = df["neutral"].astype(int)

    has_score = df["home_score"].notna()
    df.loc[has_score, "home_score"] = df.loc[has_score, "home_score"].astype(int)
    df.loc[has_score, "away_score"] = df.loc[has_score, "away_score"].astype(int)
    df["outcome"] = np.where(
        ~has_score, np.nan,
        np.where(df["home_score"] > df["away_score"], 2,
                 np.where(df["home_score"] == df["away_score"], 1, 0)),
    )

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
    df["date_ordinal"] = pd.to_datetime(df["date"]).map(lambda d: d.toordinal())
    return df


def compute_metrics(
    pred_home: np.ndarray, pred_away: np.ndarray,
    true_home: np.ndarray, true_away: np.ndarray,
    true_outcome: np.ndarray,
) -> dict[str, float]:
    """Compute outcome acc, exact score acc, MAE, draw counts."""
    pred_outcome = np.where(
        pred_home > pred_away, 2,
        np.where(pred_home == pred_away, 1, 0),
    )
    return {
        "acc_outcome": (pred_outcome == true_outcome).mean(),
        "exact_score": ((pred_home == true_home) & (pred_away == true_away)).mean(),
        "mae_home": np.abs(pred_home - true_home).mean(),
        "mae_away": np.abs(pred_away - true_away).mean(),
        "n_draws_pred": int((pred_outcome == 1).sum()),
        "n_draws_actual": int((true_outcome == 1).sum()),
    }


def make_tabicl():
    from huggingface_hub import hf_hub_download
    from tabicl import TabICLClassifier
    ckpt = Path(hf_hub_download(
        repo_id="jingang/TabICL",
        filename="tabicl-classifier-v2-20260212.ckpt",
    ))
    return TabICLClassifier(
        model_path=ckpt, norm_methods=None,
        feat_shuffle_method="latin", class_shuffle_method="shift",
        outlier_threshold=4, support_many_classes=True, batch_size=8,
        kv_cache=False, allow_auto_download=False,
        checkpoint_version=ckpt.name,
        use_amp="auto", use_fa3="auto", offload_mode="auto",
        disk_offload_dir=None,
    )


COMMANDS: dict[str, Any] = {name.replace("_", "-"): name for name in ENDPOINTS}
COMMANDS |= {"analyze": analyze, "fetch-all": fetch_all}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("values", nargs="*")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--ttl", type=float, default=DEFAULT_TTL)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)
    client = TmClient(timeout=args.timeout, ttl=args.ttl, use_cache=not args.no_cache)
    if spec := ENDPOINTS.get(name := COMMANDS[args.command]):
        names = [*spec.path_args, *spec.required_query]
        if spec.ids_query:
            names.append("ids")
        params = dict(zip(names, args.values, strict=True))
        if spec.ids_query:
            params["ids"] = [part.strip() for part in args.values[-1].split(",")]
        result = client.call(name, **params)  # type: ignore[arg-type]
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        sig = inspect.signature(name)  # type: ignore[arg-type]
        avail = {"client": client, **{v: int(v) for v in args.values}}
        kwargs = {k: avail[k] for k in sig.parameters if k in avail}
        name(**kwargs)
    return 0
