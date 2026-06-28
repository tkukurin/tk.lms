"""Query Transfermarkt's JSON API from notebooks or uv.

Examples:
  uv run python nbs/26/2606_tmapi_client.py search-transfermarkt "world cup"
  uv run python nbs/26/2606_tmapi_client.py get-competition-info FIWC

See 538's [pele] (Predictive Elo with Lineup Equilibria) for infos on chosen features.

Apparently coding agent's favorite way to run:
    import importlib.util, json, sys, time
    spec=importlib.util.spec_from_file_location('tmc','nbs/26/2606_tmapi_client.py')
    mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod)
    client=mod.TmClient(timeout=20)
    rows=[]
    t0=time.time()
    for year in mod.WORLD_CUP_YEARS:
        y0=time.time()
        row=mod.fetch_fiwc_year(client, year)
        row['seconds']=round(time.time()-y0,1)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    print(json.dumps({'seconds': round(time.time()-t0,1), 'years': rows}, ensure_ascii=False, indent=2))


[wc26silver]: : https://www.natesilver.net/p/world-cup-2026-odds-predictions
[pele]: https://www.natesilver.net/p/pele-methodology
[538meth]: https://www.natesilver.net/p/pele-international-football-rankings-soccer-ratings-projections
"""
# %% Imports
from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import partialmethod
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

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
    2006: "2006-06-09",
    2010: "2010-06-11",
    2014: "2014-06-12",
    2018: "2018-06-14",
    2022: "2022-11-20",
    2026: "2026-06-11",
}

# %% Endpoint specs
@dataclass(frozen=True)
class Endpoint:
    path: str
    path_args: tuple[str, ...] = ()
    required_query: tuple[str, ...] = ()
    optional_query: tuple[str, ...] = ()
    ids_query: bool = False


ENDPOINTS: dict[str, Endpoint] = {
    "attributes": Endpoint("attributes"),
    "search_transfermarkt": Endpoint(
        "quick-search",
        required_query=("term",),
    ),
}

# %% Player endpoints
ENDPOINTS.update(
    {
        "get_player_profile": Endpoint("player/{player_id}", ("player_id",)),
        "get_players_info": Endpoint("players", ids_query=True),
        "get_player_gallery": Endpoint(
            "player/{player_id}/gallery",
            ("player_id",),
        ),
        "get_player_performance": Endpoint(
            "player/{player_id}/performance",
            ("player_id",),
            optional_query=("season",),
        ),
        "get_player_injuries": Endpoint(
            "player/{player_id}/injury",
            ("player_id",),
        ),
        "get_player_market_value_history": Endpoint(
            "player/{player_id}/market-value-history",
            ("player_id",),
        ),
        "get_player_national_career": Endpoint(
            "player/{player_id}/national-career-history",
            ("player_id",),
        ),
        "get_player_transfer_history": Endpoint(
            "transfer/history/player/{player_id}",
            ("player_id",),
        ),
        "get_players_performance": Endpoint("players/performance"),
    }
)

# %% Club and competition endpoints
ENDPOINTS.update(
    {
        "get_club_info": Endpoint("club/{club_id}", ("club_id",)),
        "get_clubs_info": Endpoint("clubs", ids_query=True),
        "get_club_squad": Endpoint("club/{club_id}/squad", ("club_id",)),
        "get_club_stadium": Endpoint("club/{club_id}/stadium", ("club_id",)),
        "get_club_transfer_history": Endpoint(
            "transfer/history/club/{club_id}",
            ("club_id",),
        ),
        "get_competition_info": Endpoint("competition/{code}", ("code",)),
        "get_competition_table": Endpoint(
            "competition/{code}/table",
            ("code",),
        ),
        "get_competitions_info": Endpoint("competitions", ids_query=True),
    }
)

# %% Game and people endpoints
ENDPOINTS.update(
    {
        "get_game": Endpoint("game/{game_id}", ("game_id",)),
        "get_game_live_detail": Endpoint(
            "game/{game_id}/live-detail",
            ("game_id",),
        ),
        "get_referee_profile": Endpoint(
            "referee/{referee_id}",
            ("referee_id",),
        ),
        "get_referees_info": Endpoint("referees", ids_query=True),
        "get_coach_profile": Endpoint("coach/{coach_id}", ("coach_id",)),
        "get_coaches_info": Endpoint("coaches", ids_query=True),
        "get_stadium_info": Endpoint("stadium/{stadium_id}", ("stadium_id",)),
    }
)

COMMANDS = {name.replace("_", "-"): name for name in ENDPOINTS}
COMMANDS |= {"pele": fetch_fiwc_year}

# %% Endpoint construction
def encode_query(items: Iterable[tuple[str, object]]) -> str:
    return "&".join(
        f"{quote(key, safe='[]')}={quote(str(value), safe='')}"
        for key, value in items
    )


def get_endpoint(spec: Endpoint, params: dict[str, Any]) -> str:
    path_args = {
        key: quote(str(params[key]), safe="")
        for key in spec.path_args
    }
    endpoint = spec.path.format(**path_args)
    query_items: list[tuple[str, object]] = []

    if spec.ids_query:
        query_items.extend(("ids[]", value) for value in params["ids"])

    for key in spec.required_query:
        query_items.append((key, params[key]))

    for key in spec.optional_query:
        if (value := params.get(key)) is not None:
            query_items.append((key, value))

    if not query_items:
        return endpoint
    return f"{endpoint}?{encode_query(query_items)}"


# %% Client core
@dataclass
class TmClient:
    base_url: str = BASE_URL
    timeout: float = DEFAULT_TIMEOUT
    ttl: float = DEFAULT_TTL
    use_cache: bool = True
    headers: dict[str, str] = field(
        default_factory=lambda: DEFAULT_HEADERS.copy(),
    )
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
        if name not in ENDPOINTS:
            choices = ", ".join(sorted(ENDPOINTS))
            raise KeyError(
                f"Unknown endpoint {name!r}; choose one of: {choices}"
            )

        spec = ENDPOINTS[name]
        if values:
            names = [
                *spec.path_args,
                *spec.required_query,
                *(["ids"] if spec.ids_query else []),
                *spec.optional_query,
            ]
            params = dict(zip(names, values, strict=True)) | params

        return self.fetch(get_endpoint(spec, params))

    attributes = partialmethod(call, "attributes")
    search_transfermarkt = partialmethod(call, "search_transfermarkt")

    get_player_profile = partialmethod(call, "get_player_profile")
    get_players_info = partialmethod(call, "get_players_info")
    get_player_gallery = partialmethod(call, "get_player_gallery")
    get_player_performance = partialmethod(call, "get_player_performance")
    get_player_injuries = partialmethod(call, "get_player_injuries")
    get_player_market_value_history = partialmethod(
        call,
        "get_player_market_value_history",
    )
    get_player_national_career = partialmethod(
        call,
        "get_player_national_career",
    )
    get_player_transfer_history = partialmethod(
        call,
        "get_player_transfer_history",
    )
    get_players_performance = partialmethod(call, "get_players_performance")

    get_club_info = partialmethod(call, "get_club_info")
    get_clubs_info = partialmethod(call, "get_clubs_info")
    get_club_squad = partialmethod(call, "get_club_squad")
    get_club_stadium = partialmethod(call, "get_club_stadium")
    get_club_transfer_history = partialmethod(call, "get_club_transfer_history")
    get_competition_info = partialmethod(call, "get_competition_info")
    get_competition_table = partialmethod(call, "get_competition_table")
    get_competitions_info = partialmethod(call, "get_competitions_info")

    get_game = partialmethod(call, "get_game")
    get_game_live_detail = partialmethod(call, "get_game_live_detail")
    get_referee_profile = partialmethod(call, "get_referee_profile")
    get_referees_info = partialmethod(call, "get_referees_info")
    get_coach_profile = partialmethod(call, "get_coach_profile")
    get_coaches_info = partialmethod(call, "get_coaches_info")
    get_stadium_info = partialmethod(call, "get_stadium_info")


# %% World Cup player snapshot helpers
def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def get_current_club_id(player: dict[str, Any]) -> str | None:
    for assignment in player.get("clubAssignments", []):
        if assignment.get("type") == "current":
            return str(assignment.get("clubId"))
    return None


def get_market_value(player: dict[str, Any]) -> int:
    current = player.get("marketValueDetails", {}).get("current", {})
    return current.get("value") or 0


def get_value_at(history: dict[str, Any], cutoff: str) -> dict[str, Any] | None:
    rows = history.get("history", [])
    rows = [row for row in rows if row["marketValue"]["determined"] <= cutoff]
    return max(
        rows,
        key=lambda row: row["marketValue"]["determined"],
        default=None,
    )


def get_player_snapshot(
    profile: dict[str, Any],
    history: dict[str, Any],
    cutoff: str,
) -> dict[str, Any]:
    row = get_value_at(history, cutoff)
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
        "marketValueDetails": {
            "current": {
                "value": value,
                "determined": determined,
            },
        },
        "clubAssignments": (
            [] if not club_id else [{"type": "current", "clubId": club_id}]
        ),
        "missingMarketValueAtCutoff": row is None,
    }


# %% World Cup roster value helpers
def get_club_flags(
    current_id: str | None,
    country_id: int | None,
    current_clubs: dict[str, dict[str, Any]],
    country_confed: dict[int, int],
) -> tuple[bool, bool, bool]:
    if current_id is None:
        return False, False, False
    club = current_clubs.get(current_id, {})
    club_country = club.get("baseDetails", {}).get("countryId")
    is_domestic = club_country == country_id
    is_uefa = country_confed.get(club_country) == 6
    return True, is_domestic, is_uefa


def get_value_features(
    values: Sequence[int],
    ages: Sequence[int | float],
    age_value: int | float,
    position_values: dict[str, int],
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
    defense = position_values.get("DEFENDER", 0)
    defense += position_values.get("GOALKEEPER", 0) + 0.5 * mid
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


# %% World Cup captain helpers
# %% World Cup feature extraction
def get_roster_features(
    roster: Sequence[dict[str, Any]],
    squad: dict[str, Any],
    country_id: int | None,
    current_clubs: dict[str, dict[str, Any]],
    country_confed: dict[int, int],
) -> dict[str, Any]:
    values, ages, age_value = [], [], 0
    position_values: dict[str, int] = {}
    current_ids = set()
    domestic_count = current_count = 0
    domestic_value = uefa_value = u23_value = u25_value = over30_value = 0
    zero_count = missing_count = 0

    for player in roster:
        value = get_market_value(player)
        age = player.get("lifeDates", {}).get("age") or 0
        group = player.get("attributes", {}).get("positionGroup") or "UNKNOWN"
        current_id = get_current_club_id(player)
        has_club, domestic, uefa = get_club_flags(
            current_id,
            country_id,
            current_clubs,
            country_confed,
        )
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
    roster_share_feats =  {
        "u23ValueShare": ratio(u23_value, total),
        "u25ValueShare": ratio(u25_value, total),
        "over30ValueShare": ratio(over30_value, total),
        "domesticClubShare": ratio(domestic_count, current_count),
        "domesticValueShare": ratio(domestic_value, total),
        "uefaClubValueShare": ratio(uefa_value, total),
    }

    captain_ids = {p["playerId"] for p in squad["squad"] if p.get("isCaptain")}
    id2player = {player["id"]: player for player in roster}
    captain = next((id2player.get(pid) for pid in captain_ids), {})
    captain_feats = {
        "captainMarketValueEur": captain.get("marketValueDetails", {}).get("current", {}).get("value") or None,
        "captainAge": captain.get("lifeDates", {}).get("age") or None,
        "captainPosition": captain.get("attributes", {}).get("positionGroup") or None,
    }

    return get_value_features(values, ages, age_value, position_values) | {
        "squadCount": len(roster),
        "currentClubDiversity": len(current_ids - {""}),
        "squadMissingValueCount": missing_count,
        "squadZeroValueCount": zero_count,
    } | roster_share_feats | captain_feats


# %% World Cup retrieval helpers
def get_batched_rows(
    ids: Sequence[str],
    get_page: Callable[[Sequence[str]], dict[str, Any]],
    batch_size: int = TMAPI_BATCH_SIZE,
) -> dict[str, Any]:
    rows = {}
    for start in range(0, len(ids), batch_size):
        page_ids = ids[start : start + batch_size]
        for row in get_page(page_ids)["data"]:
            rows[row["id"]] = row
    return rows


# %%
def fetch_player_histories(
    client: TmClient,
    player_ids: Sequence[str],
    cache_dir: Path,
) -> dict[str, Any]:
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


def fetch_fiwc_squads(
    client: TmClient,
    club_ids: Sequence[str],
    season: int,
    out: Path,
) -> dict[str, Any]:
    squads = {}
    (out / "squads").mkdir(parents=True, exist_ok=True)
    for club_id in club_ids:
        squad = client.fetch(f"club/{club_id}/squad?season={season}")["data"]
        squads[club_id] = squad
        (out / "squads" / f"{club_id}.json").write_text(json.dumps(squad, ensure_ascii=False, indent=2))
    (out / "squads.json").write_text(json.dumps(squads, ensure_ascii=False, indent=2))
    return squads


# %%
def fetch_fiwc_year(client: TmClient, year: int, root: Path = datadir / "transfermarkt" / "fiwc") -> dict[str, Any]:
    """fetch supposed to be inspired by [pele] feats."""
    season = year - 1
    cutoff = WORLD_CUP_START_DATES[year]
    out = root / str(year)
    out.mkdir(parents=True, exist_ok=True)
    table = client.fetch(f"competition/FIWC/table?season={season}")["data"]
    (out / "table.json").write_text(json.dumps(table, ensure_ascii=False, indent=2))
    rows = [club for group in table["tables"] for club in group["clubs"]]
    club_ids = [row["clubId"] for row in rows]
    club_rows = {row["clubId"]: row for row in rows}

    clubs = get_batched_rows(club_ids, client.get_clubs_info)
    (out / "clubs.json").write_text(json.dumps(clubs, ensure_ascii=False, indent=2))

    squads = fetch_fiwc_squads(client, club_ids, season, out)
    player_ids = sorted(
        {pid for squad in squads.values() for pid in squad["playerIds"]},
        key=int,
    )
    players = get_batched_rows(player_ids, client.get_players_info)
    player_ids = sorted(players, key=int)
    (out / "player_profiles_current.json").write_text(json.dumps(players, ensure_ascii=False, indent=2))

    histories = fetch_player_histories(
        client,
        player_ids,
        root / "_player_market_value_history",
    )
    snapshots =  {
        player_id: get_player_snapshot(
            players[player_id],
            histories[player_id],
            cutoff,
        )
        for player_id in player_ids
    }
    (out / "player_snapshots.json").write_text(json.dumps(snapshots, ensure_ascii=False, indent=2))
    club_ids = sorted({ club_id for player in snapshots.values() if (club_id := get_current_club_id(player)) }, key=int)
    current_clubs = get_batched_rows(club_ids, client.get_clubs_info)
    (out / "clubs_at_value.json").write_text(json.dumps(current_clubs, ensure_ascii=False, indent=2))
    attributes = client.attributes()["data"]
    country_confed = {
        country["id"]: country["confederationId"]
        for country in attributes["countries"]
    }

    teams = {}
    for club_id, squad in squads.items():
        roster = [snapshots[player_id] for player_id in squad["playerIds"]]
        country_id = clubs[club_id].get("baseDetails", {}).get("countryId")
        teams[club_id] = get_roster_features(
            roster,
            squad,
            country_id,
            current_clubs,
            country_confed,
        ) | {"name": clubs[club_id].get("name"), "table": club_rows[club_id]}
    (out / "team_features.json").write_text(json.dumps(teams, ensure_ascii=False, indent=2))

    summary = {
        "year": year,
        "season": season,
        "cutoff": cutoff,
        "groups": len(table.get("tables", [])),
        "teams": len(club_ids),
        "squads": len(squads),
        "players": len(players),
        "missingMarketValues": sum(
            player["missingMarketValueAtCutoff"]
            for player in snapshots.values()
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


# %% CLI
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("values", nargs="*")
    parser.add_argument("--season")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--ttl", type=float, default=DEFAULT_TTL)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)
    client = TmClient(timeout=args.timeout, ttl=args.ttl, use_cache=not args.no_cache)

    name = COMMANDS[args.command]
    if spec := ENDPOINTS.get(name):
        names = [*spec.path_args, *spec.required_query]
        if spec.ids_query: names.append("ids")
        params = dict(zip(names, args.values, strict=True))
        if spec.ids_query:
            params["ids"] = [part.strip() for part in args.values[-1].split(",")]
        if "season" in spec.optional_query:
            params["season"] = args.season
        result = client.call(name, **params)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        name(client, )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
