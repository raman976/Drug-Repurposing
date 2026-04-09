from __future__ import annotations

import time
from dataclasses import dataclass

import requests


@dataclass
class StringConfig:
    base_url: str
    caller_identity: str


class StringClient:
    """Thin client for STRING API with polite rate limiting."""

    def __init__(self, config: StringConfig):
        self.config = config
        self._last_call = 0.0

    def _wait_if_needed(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)

    def _post_tsv(self, endpoint: str, payload: dict[str, str | int]) -> list[dict[str, str]]:
        self._wait_if_needed()
        url = f"{self.config.base_url}/api/tsv/{endpoint}"
        payload = {**payload, "caller_identity": self.config.caller_identity}
        response = requests.post(url, data=payload, timeout=20)
        response.raise_for_status()
        self._last_call = time.time()

        lines = [line for line in response.text.splitlines() if line.strip()]
        if len(lines) <= 1:
            return []

        headers = lines[0].split("\t")
        parsed: list[dict[str, str]] = []
        for line in lines[1:]:
            values = line.split("\t")
            row = dict(zip(headers, values))
            parsed.append(row)
        return parsed

    def map_to_string_ids(self, proteins: list[str], species: int = 9606) -> list[str]:
        if not proteins:
            return []
        rows = self._post_tsv(
            "get_string_ids",
            {
                "identifiers": "\r".join(proteins),
                "species": species,
                "echo_query": 1,
            },
        )
        return [row.get("stringId", "") for row in rows if row.get("stringId")]

    def interaction_partners(
        self,
        string_ids: list[str],
        species: int = 9606,
        required_score: int = 700,
        limit: int = 30,
    ) -> list[str]:
        partners_with_scores = self.interaction_partners_with_scores(
            string_ids=string_ids,
            species=species,
            required_score=required_score,
            limit=limit,
        )
        return sorted(partners_with_scores.keys())

    def interaction_partners_with_scores(
        self,
        string_ids: list[str],
        species: int = 9606,
        required_score: int = 700,
        limit: int = 30,
    ) -> dict[str, float]:
        if not string_ids:
            return {}
        rows = self._post_tsv(
            "interaction_partners",
            {
                "identifiers": "\r".join(string_ids),
                "species": species,
                "required_score": required_score,
                "limit": limit,
            },
        )
        partners: dict[str, float] = {}
        for row in rows:
            preferred_name = row.get("preferredName_B")
            if preferred_name:
                raw_score = row.get("score") or row.get("combined_score") or "0"
                try:
                    parsed = float(raw_score)
                except ValueError:
                    parsed = 0.0
                # STRING may provide score either in [0,1] or [0,1000].
                normalized = parsed if parsed <= 1 else (parsed / 1000.0)
                partners[preferred_name] = max(partners.get(preferred_name, 0.0), normalized)
        return partners
