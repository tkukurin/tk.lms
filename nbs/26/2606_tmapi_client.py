"""Query Transfermarkt's JSON API from notebooks or uv.

Examples:
  uv run --no-sync python nbs/26/2606_tmapi_client.py \
    search-transfermarkt "world cup"
  uv run --no-sync python nbs/26/2606_tmapi_client.py get-competition-info FIWC
"""
# %% Imports
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import partialmethod
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

CacheItem = tuple[float, Any]

BASE_URL = "https://tmapi-alpha.transfermarkt.technology/"
DEFAULT_TIMEOUT = 5.0
DEFAULT_TTL = 300.0
DEFAULT_HEADERS = {
    "Accept": "application/json",
    "Accept-Language": "pt-BR",
    "User-Agent": "Mozilla/5.0",
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

# %%

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("values", nargs="*")
    parser.add_argument("--season")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--ttl", type=float, default=DEFAULT_TTL)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)
    name = COMMANDS[args.command]
    spec = ENDPOINTS[name]
    names = [*spec.path_args, *spec.required_query]
    if spec.ids_query:
        names.append("ids")

    params = dict(zip(names, args.values, strict=True))
    if spec.ids_query:
        params["ids"] = [part.strip() for part in args.values[-1].split(",")]
    if "season" in spec.optional_query:
        params["season"] = args.season
    client = TmClient(
        timeout=args.timeout,
        ttl=args.ttl,
        use_cache=not args.no_cache,
    )
    result = client.call(name, **params)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
