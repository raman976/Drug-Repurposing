"""
LLM tool nodes for the drug repurposing agent — constrained + verified.

Design principles implemented here:
  1. LLM proposes → system verifies → system trusts
     Every LLM claim is checked against ground-truth data before use.
  2. Evidence-bound prompts
     The LLM never sees just "disease + protein list".  It always receives
     structured context: mechanism group, disease pathway, and why the protein
     appears in the dataset — reducing free-form hallucination.
  3. Numeric confidence thresholds
     Driver boosts are suppressed if LLM self-reports confidence < 0.65.
  4. Self-consistency check
     Driver assessment calls the LLM twice and accepts only proteins that
     appear in BOTH responses.  Single-run artefacts are dropped.
  5. Negative filter in self-check
     Self-check explicitly asks for reasons AGAINST each candidate, not just
     a binary plausible/implausible verdict.
  6. LLM as explainer, not authority
     LLM re-ranking is advisory: its proposed order is blended with the
     deterministic score rather than replacing it outright.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from groq import Groq

from app.models import CandidateDrug

_GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

# Minimum numeric confidence the LLM must self-report before we trust a driver.
_DRIVER_CONFIDENCE_THRESHOLD = 0.65


# ─── shared helpers ──────────────────────────────────────────────────────────

def _groq_client() -> Optional[Groq]:
    key = os.getenv("OPENAI_API_KEY")
    return Groq(api_key=key) if key else None


def _call_llm(prompt: str, max_tokens: int = 700) -> Optional[str]:
    """Try each model in order, return the first non-empty JSON response."""
    client = _groq_client()
    if not client:
        return None
    for model in _GROQ_MODELS:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
                stream=False,
            )
            content = resp.choices[0].message.content if resp.choices else None
            if content and content.strip():
                return content.strip()
        except Exception:
            continue
    return None


def _build_protein_context_block(
    proteins: list[str],
    mechanism_groups: dict[str, str],
    disease_proteins: set[str],
    protein_disease_degree: dict[str, int],
) -> str:
    """Build a structured context block for each protein.

    Each row gives the LLM:
      - gene symbol
      - mechanism group (from our classifier, verified)
      - specificity tier (how disease-specific this protein is)
      - whether it is directly in the disease's protein set

    This replaces the bare "protein: pathway" format with richer, factual
    context that grounds the LLM's reasoning in verified data.
    """
    lines = []
    for p in proteins:
        mechanism = mechanism_groups.get(p, "unclassified")
        degree = protein_disease_degree.get(p, 1)
        if degree <= 3:
            tier = "high specificity (appears in ≤3 diseases)"
        elif degree <= 15:
            tier = "moderate specificity"
        else:
            tier = "low specificity (hub protein, appears in many diseases)"
        direct = "DIRECT disease association" if p in disease_proteins else "network-expanded partner"
        lines.append(f"  {p:12} | pathway: {mechanism} | {tier} | {direct}")
    return "\n".join(lines)


# ─── Tool 1: Driver assessment with self-consistency ─────────────────────────

@dataclass
class DriverAssessmentResult:
    driver_proteins: list[str] = field(default_factory=list)
    passenger_proteins: list[str] = field(default_factory=list)
    rationale: dict[str, str] = field(default_factory=dict)
    confidence: str = "none"
    confidence_score: float = 0.0      # numeric 0–1; boost suppressed if < threshold
    trace_line: str = ""


def assess_driver_proteins(
    disease: str,
    proteins: list[str],
    mechanism_groups: dict[str, str],
    disease_proteins: set[str] | None = None,
    protein_disease_degree: dict[str, int] | None = None,
) -> DriverAssessmentResult:
    """Identify causal driver proteins using evidence-bound prompts + self-consistency.

    What is different from a naive LLM call:

    Evidence-bound prompts:
        Each protein is presented with its mechanism group, disease-specificity
        tier, and direct/expanded status — factual data from our pipeline.
        The LLM is explicitly told not to reason beyond what is provided.

    Self-consistency (ask-twice):
        The prompt is called twice.  A protein is accepted as a driver only if
        it appears in BOTH responses.  Single-run artefacts (LLM guessing
        randomly) are dropped.

    Confidence threshold:
        The LLM self-reports a numeric confidence 0–1.  If the average of the
        two runs is below _DRIVER_CONFIDENCE_THRESHOLD (0.65), the driver boost
        is suppressed — the LLM is uncertain and the deterministic score should
        dominate.
    """
    if not proteins:
        return DriverAssessmentResult(
            trace_line="No proteins — driver assessment skipped."
        )

    disease_proteins_set = disease_proteins or set()
    deg_map = protein_disease_degree or {}
    sample = proteins[:20]     # cap tokens — 20 is enough for structured context

    context_block = _build_protein_context_block(
        sample, mechanism_groups, disease_proteins_set, deg_map
    )

    prompt = f"""\
You are a biomedical expert classifying proteins for a computational drug repurposing study.

Disease: {disease}

Proteins (with verified pathway and specificity data from our pipeline):
{context_block}

Task: Classify each protein as DRIVER or PASSENGER.
  DRIVER   = protein whose DIRECT dysfunction, mutation, or overactivation
             causally propagates this specific disease.
             Examples: oncogene for cancer, viral entry receptor for infection,
             aggregating protein for neurodegeneration.
  PASSENGER = biomarker, downstream consequence, or coincidental association.

IMPORTANT CONSTRAINTS:
1. Only classify proteins from the list above — do not mention any others.
2. Be conservative: classify as PASSENGER unless you are confident it is a driver.
3. High-specificity proteins are more likely to be genuine drivers.
4. Hub proteins (low specificity) are almost always passengers.
5. Report your overall confidence as a number between 0.0 and 1.0.

Return ONLY valid JSON:
{{
  "drivers": ["PROTEIN_A"],
  "passengers": ["PROTEIN_B", "PROTEIN_C"],
  "rationale": {{
    "PROTEIN_A": "one sentence grounded in disease mechanism"
  }},
  "confidence_score": 0.85
}}"""

    # ── self-consistency: call twice, intersect drivers ───────────────────────
    results = []
    for _ in range(2):
        raw = _call_llm(prompt, max_tokens=600)
        if not raw:
            break
        try:
            results.append(json.loads(raw))
        except json.JSONDecodeError:
            break

    if not results:
        return DriverAssessmentResult(
            trace_line="LLM unavailable — driver assessment skipped, all proteins weighted equally."
        )

    allowed = {p.upper() for p in proteins}

    if len(results) == 2:
        # Accept only drivers that appear in both responses (self-consistency).
        drivers_run1 = {p.upper() for p in results[0].get("drivers", []) if p.upper() in allowed}
        drivers_run2 = {p.upper() for p in results[1].get("drivers", []) if p.upper() in allowed}
        consistent_drivers = list(drivers_run1 & drivers_run2)
        dropped = (drivers_run1 | drivers_run2) - (drivers_run1 & drivers_run2)

        # Average numeric confidence across runs.
        conf_scores = [
            float(r.get("confidence_score", 0.5))
            for r in results
            if isinstance(r.get("confidence_score"), (int, float))
        ]
        avg_confidence = sum(conf_scores) / max(len(conf_scores), 1)

        # Pick rationale from whichever run returned more.
        best = results[0] if len(results[0].get("rationale", {})) >= len(results[1].get("rationale", {})) else results[1]
    else:
        # Only one response — use it, but halve confidence (no consistency check).
        drivers_run1 = {p.upper() for p in results[0].get("drivers", []) if p.upper() in allowed}
        consistent_drivers = list(drivers_run1)
        dropped = set()
        avg_confidence = float(results[0].get("confidence_score", 0.5)) * 0.5
        best = results[0]

    rationale = {
        k.upper(): v
        for k, v in (best.get("rationale") or {}).items()
        if k.upper() in allowed
    }
    passengers = [p.upper() for p in best.get("passengers", []) if p.upper() in allowed]

    # Suppress driver boost if confidence is below threshold.
    if avg_confidence < _DRIVER_CONFIDENCE_THRESHOLD:
        suppressed_drivers = consistent_drivers
        consistent_drivers = []
        trace = (
            f"LLM driver assessment: {len(suppressed_drivers)} candidates identified but "
            f"SUPPRESSED (avg confidence {avg_confidence:.2f} < threshold {_DRIVER_CONFIDENCE_THRESHOLD}). "
            "Deterministic scoring used instead."
        )
    elif consistent_drivers:
        conf_label = "high" if avg_confidence >= 0.8 else "medium"
        consistency_note = (
            f" ({len(dropped)} inconsistent candidates dropped)" if dropped else ""
        )
        trace = (
            f"LLM driver assessment ({conf_label}, score={avg_confidence:.2f}): "
            f"{len(consistent_drivers)} drivers confirmed by self-consistency "
            f"[{', '.join(consistent_drivers[:5])}]{consistency_note}. "
            "Driver-protein scoring boost applied."
        )
    else:
        trace = (
            f"LLM driver assessment (confidence={avg_confidence:.2f}): "
            "no proteins passed self-consistency check — all weighted by specificity."
        )

    conf_label = "high" if avg_confidence >= 0.8 else ("medium" if avg_confidence >= 0.65 else "low")
    return DriverAssessmentResult(
        driver_proteins=consistent_drivers,
        passenger_proteins=passengers,
        rationale=rationale,
        confidence=conf_label,
        confidence_score=avg_confidence,
        trace_line=trace,
    )


# ─── Tool 2: Biological self-check with negative filter ──────────────────────

@dataclass
class SelfCheckResult:
    passed: bool = True
    issues: list[str] = field(default_factory=list)
    drugs_to_remove: list[str] = field(default_factory=list)
    biological_verdict: str = ""
    trace_line: str = ""


def biological_self_check(
    disease: str,
    candidates: list[CandidateDrug],
    driver_proteins: list[str],
    protein_to_drugs: dict[str, set[str]] | None = None,
) -> SelfCheckResult:
    """Check candidate biological plausibility using a negative filter.

    Two-stage approach:
    1. Dataset pre-filter: candidates whose primary target has NO verified
       link to the drug in the ground-truth database are flagged immediately,
       before any LLM call.
    2. LLM negative filter: for surviving candidates, the LLM is asked to
       articulate reasons AGAINST each drug — not just whether it's plausible.
       This asymmetric framing reduces the LLM's tendency to approve everything.

    The LLM never removes a drug on its own — it raises flags.  Removal
    requires BOTH an LLM flag AND low computational score (< 0.5).
    """
    if not candidates:
        return SelfCheckResult(passed=True, trace_line="No candidates to check.")

    top = candidates[:8]
    drug_to_drugs_map = protein_to_drugs or {}

    # ── Stage 1: dataset pre-filter ──────────────────────────────────────────
    dataset_flagged: list[str] = []
    for c in top:
        if not c.primary_target:
            continue
        verified_drugs = {d.lower() for d in drug_to_drugs_map.get(c.primary_target, set())}
        if c.drug.lower() not in verified_drugs and not verified_drugs:
            # Primary target has zero verified drugs in dataset — suspicious.
            dataset_flagged.append(c.drug)

    # ── Stage 2: LLM negative filter ─────────────────────────────────────────
    driver_note = (
        f"Confirmed causal drivers for this disease: {', '.join(driver_proteins[:6])}"
        if driver_proteins else ""
    )
    candidate_block = "\n".join(
        f"  {i+1}. {c.drug:30} | mechanism: {c.mechanism_group or 'unknown':40} "
        f"| primary target: {c.primary_target or 'N/A':10} | score: {c.score:.3f}"
        for i, c in enumerate(top)
    )

    prompt = f"""\
You are a critical biomedical reviewer for a drug repurposing pipeline.

Disease: {disease}
{driver_note}

Candidate drugs ranked by a computational scoring system:
{candidate_block}

Task: For each candidate, identify specific reasons it should NOT be used to treat {disease}.
Consider:
1. Mechanism mismatch — does the drug's mechanism of action have no established link to this disease?
2. Wrong disease class — e.g. a chemotherapy agent for an infectious disease, or vice versa.
3. Tissue inaccessibility — e.g. a CNS-impermeable drug for a brain disease.
4. Off-target toxicity — known severe side effects that would be unacceptable for this patient population.
5. Pathway irrelevance — drug targets a downstream consequence, not a disease driver.

Rules:
1. Only reference drugs from the list above.
2. Only flag a drug if you have a SPECIFIC, ARTICULABLE reason against it — not vague concerns.
3. If no clear reason exists, leave "concerns" empty for that drug.
4. Do NOT flag a drug just because it is unfamiliar to you.

Return ONLY valid JSON:
{{
  "drug_concerns": {{
    "DRUG_NAME": ["specific reason 1", "specific reason 2"],
    "DRUG_NAME_2": []
  }},
  "verdict": "one-sentence overall quality summary"
}}"""

    raw = _call_llm(prompt, max_tokens=600)
    if not raw:
        if dataset_flagged:
            return SelfCheckResult(
                passed=False,
                issues=[f"{d}: primary target has no verified drug links in dataset" for d in dataset_flagged],
                drugs_to_remove=[],   # no LLM confirmation — don't remove, just warn
                trace_line=(
                    f"LLM unavailable. Dataset pre-filter flagged {len(dataset_flagged)} candidates "
                    f"with unverified primary targets: {', '.join(dataset_flagged)}. Retained but scored lower."
                ),
            )
        return SelfCheckResult(
            passed=True,
            trace_line="LLM unavailable — self-check skipped, all candidates retained.",
        )

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return SelfCheckResult(
            passed=True,
            trace_line="LLM returned malformed JSON for self-check — retaining all candidates.",
        )

    candidate_map = {c.drug.upper(): c for c in top}
    drug_concerns: dict[str, list[str]] = {
        k.upper(): v
        for k, v in (data.get("drug_concerns") or {}).items()
        if k.upper() in candidate_map
    }

    # A drug is removed only if the LLM raised specific concerns AND its
    # computational score is below 0.5 (low confidence from the pipeline too).
    # This prevents the LLM from unilaterally removing high-scoring candidates.
    to_remove: list[str] = []
    issues: list[str] = []
    for drug_upper, concerns in drug_concerns.items():
        if not concerns:
            continue
        candidate = candidate_map[drug_upper]
        issues.append(f"{candidate.drug}: {'; '.join(concerns[:2])}")
        if candidate.score < 0.5:
            to_remove.append(candidate.drug)

    passed = len(to_remove) == 0
    verdict = data.get("verdict", "")

    if to_remove:
        trace = (
            f"Self-check negative filter: removed {len(to_remove)} candidates with both LLM concerns "
            f"AND low pipeline score (< 0.5): {', '.join(to_remove)}."
        )
    elif issues:
        trace = (
            f"Self-check flagged concerns for {len(issues)} candidate(s) but all retained "
            f"(high pipeline scores override LLM caution): {'; '.join(issues[:2])}."
        )
    else:
        trace = f"Self-check passed — {verdict}"

    if dataset_flagged and not to_remove:
        trace += f" Dataset pre-filter noted {len(dataset_flagged)} candidates with sparse target verification."

    return SelfCheckResult(
        passed=passed,
        issues=issues,
        drugs_to_remove=to_remove,
        biological_verdict=verdict,
        trace_line=trace,
    )


# ─── Tool 3: Advisory re-ranking (blended, not authoritative) ────────────────

@dataclass
class RerankResult:
    ordered_drugs: list[str] = field(default_factory=list)
    rationale: dict[str, str] = field(default_factory=dict)
    trace_line: str = ""


def llm_rerank_candidates(
    disease: str,
    candidates: list[CandidateDrug],
    driver_proteins: list[str],
    self_check_issues: list[str],
) -> RerankResult:
    """Advisory re-ranking: LLM rank is blended with deterministic score.

    Unlike a pure LLM re-ranker (which can override correct deterministic
    results with hallucinated reasoning), this implementation:

    1. Asks the LLM for a ranked order + biological rationale per drug.
    2. Converts the LLM rank to a positional score (1.0 for top, decaying).
    3. Blends LLM positional score (30%) with deterministic score (70%).
    4. Final order is by blended score — LLM nudges, does not override.

    If LLM is unavailable, original deterministic order is kept unchanged.
    """
    if len(candidates) <= 1:
        return RerankResult(
            ordered_drugs=[c.drug for c in candidates],
            trace_line="Too few candidates for re-ranking — original order retained.",
        )

    driver_note = f"Confirmed causal drivers: {', '.join(driver_proteins[:6])}" if driver_proteins else ""
    issue_note = f"Issues from self-check: {'; '.join(self_check_issues[:3])}" if self_check_issues else ""

    candidate_block = "\n".join(
        f"  {c.drug:30} | mechanism: {c.mechanism_group or 'unknown':40} "
        f"| primary target: {c.primary_target or 'N/A':10} | pipeline score: {c.score:.3f}"
        for c in candidates[:10]
    )

    prompt = f"""\
You are a biomedical expert providing advisory re-ranking for a drug repurposing pipeline.

Disease: {disease}
{driver_note}
{issue_note}

Current candidates (ordered by computational pipeline score):
{candidate_block}

Task: Re-order these candidates by biological plausibility for treating {disease}.
Prioritise:
1. Drugs whose mechanism directly targets a causal disease pathway.
2. Drugs with published evidence in this disease area or closely related conditions.
3. Drugs with an acceptable safety profile for this patient population.

Rules:
1. Only use drug names from the list above — do not invent or add any.
2. Include ALL drugs — this is a re-ordering, not a filter.
3. Provide a concise one-sentence biological rationale for the top 5 only.
4. Your ranking is advisory and will be blended with the computational score.

Return ONLY valid JSON:
{{
  "ordered_drugs": ["DRUG_1", "DRUG_2", ...],
  "rationale": {{
    "DRUG_1": "one sentence",
    "DRUG_2": "one sentence"
  }}
}}"""

    raw = _call_llm(prompt, max_tokens=700)
    if not raw:
        return RerankResult(
            ordered_drugs=[c.drug for c in candidates],
            trace_line="LLM unavailable — original deterministic ranking retained.",
        )

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return RerankResult(
            ordered_drugs=[c.drug for c in candidates],
            trace_line="LLM returned malformed JSON — original deterministic ranking retained.",
        )

    name_map = {c.drug.upper(): c for c in candidates}
    llm_order_raw = data.get("ordered_drugs") or []
    llm_order = [d for d in llm_order_raw if d.upper() in name_map]

    # Append any drugs the LLM missed (preserve completeness).
    seen = {d.upper() for d in llm_order}
    for c in candidates:
        if c.drug.upper() not in seen:
            llm_order.append(c.drug)

    n = len(llm_order)

    # ── Blend: 70% deterministic + 30% LLM positional ────────────────────────
    # LLM positional score: rank 1 → 1.0, rank n → 1/n (linear decay).
    llm_pos_score: dict[str, float] = {
        drug.upper(): (n - i) / n
        for i, drug in enumerate(llm_order)
    }
    # Deterministic scores are already in [0, 1].
    blended: list[tuple[float, CandidateDrug]] = []
    for c in candidates:
        det_score = c.score
        llm_score = llm_pos_score.get(c.drug.upper(), 0.5)
        combined = 0.70 * det_score + 0.30 * llm_score
        blended.append((combined, c))

    blended.sort(key=lambda x: -x[0])
    final_order = [c.drug for _, c in blended]

    rationale = {
        k: v
        for k, v in (data.get("rationale") or {}).items()
        if k.upper() in name_map
    }

    top3 = ", ".join(final_order[:3])
    trace = (
        f"LLM advisory re-ranking blended (70% deterministic + 30% LLM positional). "
        f"New order led by: {top3}. Rationale recorded for {len(rationale)} candidates."
    )

    return RerankResult(ordered_drugs=final_order, rationale=rationale, trace_line=trace)
