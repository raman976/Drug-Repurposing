"""
Drug repurposing agent — LangGraph-orchestrated, tool-calling architecture.

The agent is NOT a pipeline.  At each node the system makes a decision:

  retrieve      → normalise disease, load direct proteins
  assess_drivers→ LLM TOOL: which proteins are causal drivers? (boosts ranking)
  expand        → STRING PPI network expansion (conditional)
  rank          → deterministic scoring with driver-protein priority boost
  self_check    → LLM TOOL: are these candidates biologically coherent?
  rerank        → LLM TOOL: re-order by biological plausibility (conditional)
  synthesize    → LLM generates final natural-language explanation

The LLM tools are in agent_tools.py.  They call Groq and degrade gracefully
(return empty/passthrough) when OPENAI_API_KEY is not set.

Every decision, critique, and LLM rationale is appended to `reasoning_trace`
so the frontend can show exactly why each candidate was surfaced or removed.
"""

from __future__ import annotations

import os
from typing import Callable, TypedDict

from groq import Groq
from langgraph.graph import END, START, StateGraph

from app.models import CandidateDrug
from app.services.agent_tools import (
    DriverAssessmentResult,
    RerankResult,
    SelfCheckResult,
    assess_driver_proteins,
    biological_self_check,
    llm_rerank_candidates,
)
from app.services.reasoner import DrugRepurposingReasoner, ReasoningResult


# ─── Agent state ─────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    # ── inputs ──────────────────────────────────────────────────────────────
    disease: str
    species: int
    expand_with_string: bool

    # ── working memory (built incrementally by nodes) ────────────────────────
    normalized_disease: str
    direct_proteins: list[str]
    expanded_proteins: list[str]
    interaction_scores: dict[str, float]
    reasoning_trace: list[str]

    # ── LLM tool outputs ─────────────────────────────────────────────────────
    driver_proteins: list[str]          # from assess_drivers node
    driver_rationale: dict[str, str]    # protein → reasoning sentence
    driver_confidence: str

    self_check_passed: bool
    self_check_issues: list[str]
    drugs_to_remove: list[str]          # from self_check node

    rerank_applied: bool
    llm_rationale: dict[str, str]       # drug → rerank reasoning sentence

    # ── ranking results ───────────────────────────────────────────────────────
    ranked_candidates: list[CandidateDrug]

    # ── final outputs ─────────────────────────────────────────────────────────
    result: ReasoningResult | None
    explanation: str | None
    summary: str | None
    on_progress: Callable[[str], None]


# ─── Agent ────────────────────────────────────────────────────────────────────

class DrugRepurposingAgent:
    """Six-node LangGraph agent for drug repurposing.

    Nodes
    -----
    retrieve       deterministic — data store lookup + disease normalisation
    assess_drivers LLM tool     — causal driver identification
    expand         deterministic — STRING PPI expansion (conditional)
    rank           deterministic — mechanistic scoring w/ driver boost
    self_check     LLM tool     — biological plausibility filter
    rerank         LLM tool     — biological re-ordering (conditional)
    synthesize     LLM          — natural language explanation
    """

    def __init__(self, reasoner: DrugRepurposingReasoner) -> None:
        self.reasoner = reasoner
        self._graph = self._build_graph()

    @staticmethod
    def _emit_new_trace(state: AgentState, trace: list[str], start_index: int) -> None:
        callback = state.get("on_progress")
        if callback:
            for step in trace[start_index:]:
                callback(step)

    # ─── graph construction ──────────────────────────────────────────────────

    def _build_graph(self):
        g = StateGraph(AgentState)

        g.add_node("retrieve",       self._retrieve_node)
        g.add_node("assess_drivers", self._assess_drivers_node)
        g.add_node("expand",         self._expand_node)
        g.add_node("rank",           self._rank_node)
        g.add_node("self_check",     self._self_check_node)
        g.add_node("rerank",         self._rerank_node)
        g.add_node("synthesize",     self._synthesize_node)

        g.add_edge(START, "retrieve")

        # After retrieve: no proteins → skip to synthesize, else assess drivers
        g.add_conditional_edges(
            "retrieve",
            lambda s: "synthesize" if not s.get("direct_proteins") else "assess_drivers",
        )

        # After driver assessment: expand or go straight to ranking
        g.add_conditional_edges(
            "assess_drivers",
            lambda s: "expand" if s.get("expand_with_string") else "rank",
        )

        g.add_edge("expand", "rank")
        g.add_edge("rank",   "self_check")

        # After self-check: if it failed and we have issues → rerank, else synthesize
        g.add_conditional_edges(
            "self_check",
            lambda s: "rerank" if not s.get("self_check_passed", True) else "synthesize",
        )

        g.add_edge("rerank",    "synthesize")
        g.add_edge("synthesize", END)

        return g.compile()

    # ─── node 1: retrieve ────────────────────────────────────────────────────

    def _retrieve_node(self, state: AgentState) -> AgentState:
        disease = state["disease"]
        trace: list[str] = []
        start_index = len(trace)

        normalized = self.reasoner.normalize_disease(disease)
        trace.append(f"Normalised query '{disease}' → '{normalized}'.")

        disease_key = self.reasoner._clean_text(normalized)
        dataset_proteins = set(self.reasoner.store.disease_to_proteins.get(disease_key, set()))

        direct_proteins = sorted(dataset_proteins)

        if direct_proteins:
            clusters = self.reasoner.cluster_proteins(direct_proteins)
            pathway_summary = ", ".join(
                f"{pw} ({len(ps)})" for pw, ps in list(clusters.items())[:5]
            )
            trace.append(
                f"Retrieved {len(direct_proteins)} direct proteins for '{normalized}' "
                f"from dataset. Pathway distribution: {pathway_summary}."
            )
        else:
            trace.append(
                f"No direct proteins found for '{normalized}'. "
                "Agent will attempt to return an informative empty result."
            )

        self._emit_new_trace(state, trace, start_index)
        return {
            **state,
            "normalized_disease": normalized,
            "direct_proteins": direct_proteins,
            "reasoning_trace": trace,
            "driver_proteins": [],
            "driver_rationale": {},
            "driver_confidence": "none",
            "expanded_proteins": [],
            "interaction_scores": {},
            "self_check_passed": True,
            "self_check_issues": [],
            "drugs_to_remove": [],
            "rerank_applied": False,
            "llm_rationale": {},
            "ranked_candidates": [],
        }

    # ─── node 2: assess drivers (LLM tool) ───────────────────────────────────

    def _assess_drivers_node(self, state: AgentState) -> AgentState:
        trace = list(state.get("reasoning_trace", []))
        start_index = len(trace)
        direct_proteins = state.get("direct_proteins", [])
        normalized = state.get("normalized_disease", state["disease"])

        mechanism_groups = {
            p: self.reasoner.mechanism_group_from_protein(p)
            for p in direct_proteins
        }

        trace.append(
            f"Invoking LLM driver assessment tool for {len(direct_proteins)} proteins..."
        )
        disease_key = self.reasoner._clean_text(normalized)
        disease_protein_set = self.reasoner.store.disease_to_proteins.get(disease_key, set())
        result: DriverAssessmentResult = assess_driver_proteins(
            disease=normalized,
            proteins=direct_proteins,
            mechanism_groups=mechanism_groups,
            disease_proteins=disease_protein_set,
            protein_disease_degree=dict(self.reasoner._protein_disease_degree),
        )
        trace.append(result.trace_line)

        if result.rationale:
            for protein, reason in list(result.rationale.items())[:5]:
                trace.append(f"  Driver [{protein}]: {reason}")

        self._emit_new_trace(state, trace, start_index)
        return {
            **state,
            "driver_proteins": result.driver_proteins,
            "driver_rationale": result.rationale,
            "driver_confidence": result.confidence,
            "reasoning_trace": trace,
        }

    # ─── node 3: expand (STRING PPI) ─────────────────────────────────────────

    def _expand_node(self, state: AgentState) -> AgentState:
        trace = list(state.get("reasoning_trace", []))
        start_index = len(trace)
        direct_proteins = state.get("direct_proteins", [])
        species = state.get("species", 9606)

        direct_count = len(direct_proteins)
        # Adaptive strategy: fewer direct proteins → deeper expansion
        if direct_count < 5:
            required_score, limit = 600, 120
            trace.append(
                f"Sparse direct signal ({direct_count} proteins) — selecting deep STRING expansion "
                f"(required_score={required_score}, limit={limit})."
            )
        else:
            required_score, limit = 800, 50
            trace.append(
                f"Strong direct signal ({direct_count} proteins) — selecting focused STRING expansion "
                f"(required_score={required_score}, limit={limit})."
            )

        expanded, scores = self.reasoner.expand_proteins(
            direct_proteins,
            species=species,
            required_score=required_score,
            limit=limit,
            min_interaction_score=max(0.7, required_score / 1000.0),
        )

        # Remove nonspecific hub proteins from expansion
        nonspecific = [p for p in expanded if self.reasoner._is_nonspecific_protein(p)]
        if nonspecific:
            expanded = [p for p in expanded if not self.reasoner._is_nonspecific_protein(p)]
            scores = {p: s for p, s in scores.items() if p in expanded}
            trace.append(
                f"Removed {len(nonspecific)} non-specific hub proteins from STRING expansion "
                f"({', '.join(nonspecific[:5])}{'...' if len(nonspecific) > 5 else ''})."
            )

        interaction_scores: dict[str, float] = {p: 1.0 for p in direct_proteins}
        interaction_scores.update(scores)

        trace.append(
            f"STRING expansion complete: {len(expanded)} interaction partners added "
            f"(total protein pool: {len(direct_proteins) + len(expanded)})."
        )

        self._emit_new_trace(state, trace, start_index)
        return {
            **state,
            "expanded_proteins": expanded,
            "interaction_scores": interaction_scores,
            "reasoning_trace": trace,
        }

    # ─── node 4: rank ─────────────────────────────────────────────────────────

    def _rank_node(self, state: AgentState) -> AgentState:
        trace = list(state.get("reasoning_trace", []))
        start_index = len(trace)
        direct_proteins = state.get("direct_proteins", [])
        expanded_proteins = state.get("expanded_proteins", [])
        normalized = state.get("normalized_disease", state["disease"])
        driver_proteins = frozenset(state.get("driver_proteins", []))
        interaction_scores = state.get("interaction_scores", {p: 1.0 for p in direct_proteins})

        proteins_for_drugs = sorted(set(direct_proteins) | set(expanded_proteins))
        clusters = self.reasoner.cluster_proteins(proteins_for_drugs)

        if clusters:
            trace.append(
                "Pathway clusters identified: "
                + ", ".join(f"{pw} ({len(ps)})" for pw, ps in list(clusters.items())[:6])
            )

        candidate_map = self.reasoner.get_candidate_drugs(proteins_for_drugs)
        trace.append(f"Candidate pool before ranking: {len(candidate_map)} drugs.")

        ranked = self.reasoner.rank_drugs(
            drugs=candidate_map,
            proteins=proteins_for_drugs,
            direct_proteins=direct_proteins,
            disease=normalized,
            interaction_scores=interaction_scores,
            pathway_sizes={pw: len(ps) for pw, ps in clusters.items()},
            driver_proteins=driver_proteins,
        )

        if driver_proteins and ranked:
            trace.append(
                f"Driver-protein priority boost applied to {len(driver_proteins)} causal proteins — "
                f"candidates targeting drivers receive a 1.3× advisory boost (data-driven driver_ratio also active)."
            )

        shortlisted = self.reasoner.diversify_candidates(ranked, top_k=self.reasoner.top_k)
        shortlisted = self.reasoner._attach_explanations(
            disease=normalized,
            direct_proteins=set(direct_proteins),
            candidates=shortlisted,
        )

        if shortlisted:
            top = shortlisted[0]
            trace.append(
                f"Ranked {len(ranked)} candidates → shortlisted {len(shortlisted)} after diversity filtering. "
                f"Top candidate: {top.drug} (score={top.score:.3f}, "
                f"mechanism={top.mechanism_group or 'N/A'}, stage={top.lifecycle_stage or 'N/A'})."
            )
        else:
            trace.append("No candidates survived ranking thresholds.")

        self._emit_new_trace(state, trace, start_index)
        return {
            **state,
            "ranked_candidates": shortlisted,
            "reasoning_trace": trace,
        }

    # ─── node 5: self-check (LLM tool) ────────────────────────────────────────

    def _self_check_node(self, state: AgentState) -> AgentState:
        trace = list(state.get("reasoning_trace", []))
        start_index = len(trace)
        candidates = state.get("ranked_candidates", [])
        normalized = state.get("normalized_disease", state["disease"])
        drivers = state.get("driver_proteins", [])

        if not candidates:
            trace.append("Self-check skipped — no candidates to evaluate.")
            self._emit_new_trace(state, trace, start_index)
            return {**state, "self_check_passed": True, "reasoning_trace": trace}

        trace.append(
            f"Invoking LLM biological self-check on top {min(len(candidates), 8)} candidates..."
        )
        result: SelfCheckResult = biological_self_check(
            disease=normalized,
            candidates=candidates,
            driver_proteins=drivers,
            protein_to_drugs=self.reasoner.store.protein_to_drugs,
        )
        trace.append(result.trace_line)

        if result.biological_verdict:
            trace.append(f"Biological verdict: {result.biological_verdict}")

        self._emit_new_trace(state, trace, start_index)
        return {
            **state,
            "self_check_passed": result.passed,
            "self_check_issues": result.issues,
            "drugs_to_remove": result.drugs_to_remove,
            "reasoning_trace": trace,
        }

    # ─── node 6: rerank (LLM tool, conditional) ───────────────────────────────

    def _rerank_node(self, state: AgentState) -> AgentState:
        trace = list(state.get("reasoning_trace", []))
        start_index = len(trace)
        candidates = state.get("ranked_candidates", [])
        normalized = state.get("normalized_disease", state["disease"])
        drivers = state.get("driver_proteins", [])
        issues = state.get("self_check_issues", [])
        to_remove = set(d.upper() for d in state.get("drugs_to_remove", []))

        # Apply self-check filter first
        filtered = [c for c in candidates if c.drug.upper() not in to_remove]
        if len(filtered) < len(candidates):
            removed_names = [c.drug for c in candidates if c.drug.upper() in to_remove]
            trace.append(
                f"Applied self-check filter: removed {len(removed_names)} candidates "
                f"({', '.join(removed_names)})."
            )

        trace.append(
            f"Invoking LLM re-ranking on {len(filtered)} remaining candidates..."
        )
        result: RerankResult = llm_rerank_candidates(
            disease=normalized,
            candidates=filtered,
            driver_proteins=drivers,
            self_check_issues=issues,
        )
        trace.append(result.trace_line)

        # Apply the new order while keeping all fields from original CandidateDrug objects
        name_to_candidate = {c.drug: c for c in filtered}
        reordered = [
            name_to_candidate[drug]
            for drug in result.ordered_drugs
            if drug in name_to_candidate
        ]

        # Attach LLM rationale to explanation field
        for candidate in reordered:
            llm_reason = result.rationale.get(candidate.drug) or result.rationale.get(candidate.drug.upper())
            if llm_reason:
                candidate.explanation = (
                    f"{candidate.explanation or ''} [LLM: {llm_reason}]".strip()
                )

        self._emit_new_trace(state, trace, start_index)
        return {
            **state,
            "ranked_candidates": reordered,
            "rerank_applied": True,
            "llm_rationale": result.rationale,
            "reasoning_trace": trace,
        }

    # ─── node 7: synthesize ───────────────────────────────────────────────────

    def _synthesize_node(self, state: AgentState) -> AgentState:
        trace = list(state.get("reasoning_trace", []))
        start_index = len(trace)
        normalized = state.get("normalized_disease", state["disease"])
        direct_proteins = state.get("direct_proteins", [])
        expanded_proteins = state.get("expanded_proteins", [])
        candidates = state.get("ranked_candidates", [])
        driver_proteins = state.get("driver_proteins", [])
        to_remove = set(d.upper() for d in state.get("drugs_to_remove", []))
        rerank_applied = state.get("rerank_applied", False)

        # If we arrived here directly from self_check (passed) we may still
        # need to apply the drugs_to_remove filter.
        if to_remove and not rerank_applied:
            before = len(candidates)
            candidates = [c for c in candidates if c.drug.upper() not in to_remove]
            if len(candidates) < before:
                trace.append(
                    f"Applied self-check filter in synthesis: "
                    f"removed {before - len(candidates)} implausible candidates."
                )

        # Build hypotheses from current (possibly reranked) candidates
        proteins_for_drugs = sorted(set(direct_proteins) | set(expanded_proteins))
        clusters = self.reasoner.cluster_proteins(proteins_for_drugs) if proteins_for_drugs else {}
        hypotheses = self.reasoner.generate_hypotheses(
            disease=normalized,
            clusters=clusters,
            candidates=candidates,
        )

        strategy_parts = [
            f"expand={'on' if expanded_proteins else 'off'}",
            f"driver_boost={'on' if driver_proteins else 'off'}",
            f"self_check={'passed' if state.get('self_check_passed', True) else 'fixed'}",
            f"rerank={'on' if rerank_applied else 'off'}",
        ]
        strategy = "agentic: " + ", ".join(strategy_parts)

        summary = self.reasoner.build_summary(
            disease=state["disease"],
            normalized_disease=normalized,
            reasoning_trace=trace,
            hypotheses=hypotheses,
            candidates=candidates,
        )

        result = ReasoningResult(
            disease=state["disease"],
            normalized_disease=normalized,
            direct_proteins=direct_proteins,
            expanded_proteins=expanded_proteins,
            candidates=candidates,
            strategy=strategy,
            reasoning_trace=trace,
            hypotheses=hypotheses,
            summary=summary,
        )

        # LLM final explanation
        explanation = self._llm_explain(result, driver_proteins, state.get("llm_rationale", {}))
        trace.append("Agent completed reasoning and generated final explanation.")

        self._emit_new_trace(state, trace, start_index)
        return {
            **state,
            "ranked_candidates": candidates,
            "result": result,
            "explanation": explanation,
            "summary": summary,
            "reasoning_trace": trace,
        }

    # ─── LLM explanation helper ──────────────────────────────────────────────

    def _llm_explain(
        self,
        result: ReasoningResult,
        driver_proteins: list[str],
        llm_rationale: dict[str, str],
    ) -> str:
        if not result.candidates:
            return (
                f"No repurposing candidates were found for {result.normalized_disease} "
                "in the current dataset under the applied thresholds."
            )

        top = result.candidates[0]
        top_hypothesis = next(
            (h for h in result.hypotheses if h.status == "supported"), None
        )

        # Build a rich fallback explanation that always works without LLM
        driver_note = (
            f" LLM identified {len(driver_proteins)} causal drivers "
            f"({', '.join(driver_proteins[:4])}) which received scoring priority."
            if driver_proteins else ""
        )
        rationale_note = ""
        if llm_rationale and top.drug in llm_rationale:
            rationale_note = f" Biological rationale: {llm_rationale[top.drug]}"
        elif llm_rationale and top.drug.upper() in {k.upper() for k in llm_rationale}:
            key = next(k for k in llm_rationale if k.upper() == top.drug.upper())
            rationale_note = f" Biological rationale: {llm_rationale[key]}"

        fallback = (
            f"For {result.normalized_disease}, the agent explored "
            f"{len(result.reasoning_trace)} reasoning steps across "
            f"{len(result.direct_proteins)} direct proteins"
            f"{f' and {len(result.expanded_proteins)} STRING-expanded interaction partners' if result.expanded_proteins else ''}."
            f"{driver_note}"
            f" The top candidate is {top.drug} (score={top.score:.3f}), "
            f"targeting {', '.join(top.matched_proteins[:4])} "
            f"via {top.mechanism_group or 'a protein-target interaction'}."
            f"{rationale_note}"
            f" This is hypothesis-generating computational evidence, not a clinical recommendation."
        )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return fallback

        hypothesis_text = (
            f"Top hypothesis: {top_hypothesis.pathway} "
            f"({len(top_hypothesis.supporting_proteins)} supporting proteins, "
            f"{top_hypothesis.confidence} confidence)."
            if top_hypothesis else ""
        )
        driver_context = (
            f"Causal drivers identified: {', '.join(driver_proteins[:5])}."
            if driver_proteins else ""
        )
        rerank_context = (
            f"LLM biological rationale for {top.drug}: {llm_rationale.get(top.drug, '')}"
            if llm_rationale.get(top.drug) else ""
        )

        prompt = (
            "You are a biomedical reasoning agent reporting a drug repurposing hypothesis. "
            "Write 3 concise, cautious scientific sentences. "
            "Do not claim clinical efficacy. State this is computational hypothesis-generating evidence. "
            f"Disease: {result.normalized_disease}. "
            f"Top candidate: {top.drug} (score={top.score:.3f}). "
            f"Mechanism: {top.mechanism_group}. Stage: {top.lifecycle_stage or 'N/A'}. "
            f"Supporting proteins: {', '.join(top.matched_proteins[:6])}. "
            f"{hypothesis_text} {driver_context} {rerank_context}"
        )

        try:
            client = Groq(api_key=api_key)
            for model in ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_completion_tokens=300,
                    stream=False,
                )
                content = completion.choices[0].message.content if completion.choices else None
                if isinstance(content, str) and content.strip():
                    return content.strip()
        except Exception:
            pass

        return fallback

    # ─── public API ──────────────────────────────────────────────────────────

    def run(
        self,
        disease: str,
        species: int = 9606,
        expand_with_string: bool = True,
        on_progress: Callable[[str], None] | None = None,
    ) -> AgentState:
        return self._graph.invoke(
            {
                "disease": disease,
                "species": species,
                "expand_with_string": expand_with_string,
                "on_progress": on_progress,
            }
        )
