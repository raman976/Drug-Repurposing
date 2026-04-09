from typing import List

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    disease: str = Field(..., min_length=2, description="Disease name to query")
    species: int = Field(default=9606, description="NCBI species identifier")
    expand_with_string: bool = Field(
        default=False,
        description="Whether to include STRING interaction partners in reasoning",
    )
    include_explanation: bool = Field(
        default=True,
        description="Whether to generate an explanation of reasoning",
    )


class CandidateDrug(BaseModel):
    drug: str
    matched_proteins: List[str]
    source: str
    support_count: int = 0
    score: float = 0.0

    # ── ranking score (unchanged) ────────────────────────────────────────────
    # score = biologically-weighted network signal, used for ordering only.
    # It is NOT a probability or a clinical confidence measure.

    # ── calibrated confidence (separate from score) ──────────────────────────
    # confidence reflects the *biological strength of the evidence*, not just
    # rank position.  A drug can have high score but Low confidence (e.g. large
    # network connectivity without direct disease protein hits).
    confidence: str = "Low"              # High | Medium | Low (calibrated)

    # ── evidence tier ────────────────────────────────────────────────────────
    # Tier 1 — Strong:      known drug class, direct targets, high mechanism alignment
    # Tier 2 — Hypothesis:  mechanistic support but indirect or unvalidated
    # Tier 3 — Exploratory: network signal only, weak biological grounding
    evidence_level: str = "Exploratory"  # Strong | Hypothesis | Exploratory
    tier: int = 3                        # 1 | 2 | 3

    # ── interpretability ─────────────────────────────────────────────────────
    primary_target: str | None = None
    mechanism_group: str | None = None
    mechanism_alignment: float | None = None
    lifecycle_stage: str | None = None
    lifecycle_alignment: float | None = None
    explanation: str | None = None

    # ── caveats: why this might be a false positive ───────────────────────────
    # Populated by the pipeline when there are specific reasons to be cautious.
    # An empty list means no red flags were detected — NOT that the drug is correct.
    caveats: List[str] = Field(default_factory=list)


class HypothesisSummary(BaseModel):
    pathway: str
    supporting_proteins: List[str] = Field(default_factory=list)
    candidate_drugs: List[str] = Field(default_factory=list)
    top_score: float = 0.0
    confidence: str = "Low"
    status: str = "supported"
    summary: str = ""


class QueryResponse(BaseModel):
    disease: str
    normalized_disease: str
    direct_proteins: List[str]
    expanded_proteins: List[str]
    candidates: List[CandidateDrug]
    mechanism_groups: dict[str, int] = {}
    strategy: str | None = None
    explanation: str | None = None
    summary: str | None = None
    reasoning_trace: List[str] = Field(default_factory=list)
    hypotheses: List[HypothesisSummary] = Field(default_factory=list)
    # Tier breakdown: how many candidates fall into each evidence tier.
    # Useful for the frontend to show a quality summary at a glance.
    tier_summary: dict[str, int] = Field(default_factory=dict)


class DiseaseSuggestionsResponse(BaseModel):
    query: str
    suggestions: List[str]