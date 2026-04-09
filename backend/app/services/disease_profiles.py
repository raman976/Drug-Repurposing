"""Disease profile registry.

Loads structured disease knowledge from ``data/disease_profiles.yaml`` and
exposes it through :class:`DiseaseProfileRegistry`.  The reasoner uses this
instead of the giant hardcoded dicts that were previously baked into __init__.

Adding a new disease requires only a new YAML entry — no Python changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass(frozen=True)
class LifecycleStage:
    name: str
    prior: float
    targets: frozenset[str]           # upper-case gene symbols
    drug_keywords: tuple[str, ...]    # lower-case substring keywords


@dataclass(frozen=True)
class DiseaseProfile:
    """All disease-specific knowledge for the scoring pipeline."""

    token: str                              # canonical lowercase key
    aliases: tuple[str, ...]               # other names that map here
    neuro_disease: bool

    # Proteins whose modulation directly addresses this disease.
    known_targets: frozenset[str]

    # Protein-level relevance weights (0–1).  Absent proteins fall back to
    # generic heuristics in the reasoner.
    target_relevance: dict[str, float]

    # Added signal on top of raw score for proteins with strong disease
    # driver evidence (e.g. APP for Alzheimer's).
    biological_boost: dict[str, float]

    # Substring keywords whose presence in a drug name boosts evidence_weight.
    drug_evidence_keywords: tuple[str, ...]

    # Prior probability over mechanism groups — shapes the disease profile
    # that drug mechanisms are aligned against.
    mechanism_priors: dict[str, float]

    # Ordered disease lifecycle stages with associated targets and drugs.
    lifecycle_stages: dict[str, LifecycleStage]

    # ------------------------------------------------------------------ views

    @property
    def stage_targets(self) -> dict[str, frozenset[str]]:
        return {s: ls.targets for s, ls in self.lifecycle_stages.items()}

    @property
    def lifecycle_priors(self) -> dict[str, float]:
        return {s: ls.prior for s, ls in self.lifecycle_stages.items()}

    @property
    def stage_drug_keywords(self) -> dict[str, tuple[str, ...]]:
        return {s: ls.drug_keywords for s, ls in self.lifecycle_stages.items()}


class DiseaseProfileRegistry:
    """Load and serve :class:`DiseaseProfile` objects from a YAML file.

    Matching is intentionally permissive: an exact token/alias hit wins first,
    then substring containment is tried so that queries like
    "Alzheimer's disease" correctly resolve to the ``alzheimer`` profile.
    """

    def __init__(self, yaml_path: Path) -> None:
        self._profiles: dict[str, DiseaseProfile] = {}
        # Maps every alias (and the token itself) to the canonical token.
        self._alias_index: dict[str, str] = {}
        if yaml_path.exists():
            self._load(yaml_path)

    # ------------------------------------------------------------------ I/O

    def _load(self, yaml_path: Path) -> None:
        with yaml_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        for token, data in (raw or {}).get("diseases", {}).items():
            stages: dict[str, LifecycleStage] = {}
            for stage_name, sd in (data.get("lifecycle_stages") or {}).items():
                stages[stage_name] = LifecycleStage(
                    name=stage_name,
                    prior=float(sd.get("prior", 0.1)),
                    targets=frozenset(t.upper() for t in (sd.get("targets") or [])),
                    drug_keywords=tuple(sd.get("drug_keywords") or []),
                )

            profile = DiseaseProfile(
                token=token,
                aliases=tuple(a.lower() for a in (data.get("aliases") or [])),
                neuro_disease=bool(data.get("neuro_disease", False)),
                known_targets=frozenset(
                    t.upper() for t in (data.get("known_targets") or [])
                ),
                target_relevance={
                    k.upper(): float(v)
                    for k, v in (data.get("target_relevance") or {}).items()
                },
                biological_boost={
                    k.upper(): float(v)
                    for k, v in (data.get("biological_boost") or {}).items()
                },
                drug_evidence_keywords=tuple(data.get("drug_evidence_keywords") or []),
                mechanism_priors=dict(data.get("mechanism_priors") or {}),
                lifecycle_stages=stages,
            )

            self._profiles[token] = profile
            self._alias_index[token] = token
            for alias in profile.aliases:
                self._alias_index[alias] = token

    # ----------------------------------------------------------------- query

    def match(self, disease_string: str) -> Optional[DiseaseProfile]:
        """Return the best :class:`DiseaseProfile` for *disease_string*, or ``None``.

        Resolution order:
        1. Exact token or alias match.
        2. Token is a substring of the query  (e.g. "hiv infection").
        3. Any alias is a substring of the query or vice-versa.
        """
        key = disease_string.strip().lower()

        # --- 1. exact ---
        if key in self._alias_index:
            return self._profiles[self._alias_index[key]]

        # --- 2 & 3. substring ---
        for token, profile in self._profiles.items():
            if token in key:
                return profile
            for alias in profile.aliases:
                if alias in key or key in alias:
                    return profile

        return None

    @property
    def all_profiles(self) -> dict[str, DiseaseProfile]:
        return dict(self._profiles)
