from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import difflib
import math
import os
from pathlib import Path
import re
from typing import Callable, Iterable, Optional

from groq import Groq

from app.models import CandidateDrug
from app.services.data_store import BiomedicalStore
from app.services.disease_profiles import DiseaseProfile, DiseaseProfileRegistry
from app.services.string_client import StringClient

_PROFILES_YAML = Path(__file__).resolve().parents[2] / "data" / "disease_profiles.yaml"


@dataclass
class HypothesisSummary:
    pathway: str
    supporting_proteins: list[str] = field(default_factory=list)
    candidate_drugs: list[str] = field(default_factory=list)
    top_score: float = 0.0
    confidence: str = "Low"
    status: str = "supported"
    summary: str = ""


@dataclass
class ReasoningResult:
    disease: str
    normalized_disease: str
    direct_proteins: list[str]
    expanded_proteins: list[str]
    candidates: list[CandidateDrug]
    strategy: str
    reasoning_trace: list[str] = field(default_factory=list)
    hypotheses: list[HypothesisSummary] = field(default_factory=list)
    summary: str = ""


class _ProgressTrace(list[str]):
    """Trace list that can mirror new steps to a streaming callback."""

    def __init__(self, on_progress: Optional[Callable[[str], None]] = None):
        super().__init__()
        self._on_progress = on_progress

    def append(self, item: str) -> None:  # type: ignore[override]
        super().append(item)
        if self._on_progress:
            self._on_progress(item)


class DrugRepurposingReasoner:
    def __init__(self, store: BiomedicalStore, string_client: StringClient):
        self.store = store
        self.string_client = string_client
        self.top_k = 10
        self._normalization_cache: dict[str, str] = {}
        self._protein_disease_degree: dict[str, int] = defaultdict(int)
        for proteins in self.store.disease_to_proteins.values():
            for protein in proteins:
                self._protein_disease_degree[protein] += 1
        self._nonspecific_prefixes: tuple[str, ...] = (
            "UGT",
            "CYP",
            "SLC",
            "ABCB",
            "ABCC",
            "ABCG",
        )
        self._nonspecific_keywords: tuple[str, ...] = (
            "metabolism",
            "detox",
            "xenobiotic",
        )
        # Disease-specific knowledge is now loaded from disease_profiles.yaml.
        # This removes ~220 lines of hardcoded dicts and makes the pipeline
        # work for every disease in the dataset, not only HIV/COVID/Alzheimer.
        self._registry = DiseaseProfileRegistry(_PROFILES_YAML)
        self._non_cns_drug_keywords: tuple[str, ...] = (
            "sunitinib",
            "imatinib",
            "dasatinib",
            "erlotinib",
            "sorafenib",
            "nilotinib",
            "lapatinib",
            "ponatinib",
            "gefitinib",
            "regorafenib",
        )
        self._kinase_markers: tuple[str, ...] = (
            "CDK",
            "MAPK",
            "JAK",
            "TYK",
            "AURK",
            "PIK3",
            "EGFR",
            "FLT",
            "AKT",
            "MTOR",
            "SRC",
            "ABL",
            "RAF",
            "MEK",
            "ERK",
            "IKB",
            "TBK",
            "PLK",
            "EPHA",
            "EPHB",
            "KIT",
            "RET",
            "PDGFR",
            "FGFR",
        )
        # Lifecycle priors, stage targets, and drug keywords are all in the
        # YAML-backed registry above.  Nothing to initialise here.

    @property
    def all_diseases(self) -> list[str]:
        return sorted(self.store.disease_display.values())

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join(text.strip().lower().replace("_", " ").split())

    def _best_dataset_match(self, text: str) -> tuple[str, float]:
        cleaned = self._clean_text(text)
        if not cleaned:
            return "", 0.0

        query_tokens = set(cleaned.split())
        best_key = ""
        best_score = 0.0
        for disease_key in self.store.disease_to_proteins:
            disease_tokens = set(disease_key.split())
            overlap = len(query_tokens & disease_tokens)
            token_score = overlap / max(len(query_tokens), len(disease_tokens), 1)
            ratio = difflib.SequenceMatcher(None, cleaned, disease_key).ratio()
            substring_bonus = 0.25 if cleaned in disease_key or disease_key in cleaned else 0.0
            score = (token_score * 0.6) + (ratio * 0.3) + substring_bonus
            if score > best_score:
                best_score = score
                best_key = disease_key
        return best_key, best_score

    def _llm_generate_synonyms(self, query: str) -> list[str]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return []

        prompt = (
            "Given a user disease query, return up to 8 normalized medical synonym candidates. "
            "Return ONLY a plain newline-separated list, no numbering, no extra text. "
            "Include canonical disease names when possible. "
            f"Query: {query}"
        )
        try:
            client = Groq(api_key=api_key)
            for model_name in ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_completion_tokens=128,
                    top_p=1,
                    stream=False,
                )
                content = completion.choices[0].message.content if completion.choices else ""
                if not content:
                    continue
                candidates = []
                for line in content.splitlines():
                    cleaned = self._clean_text(re.sub(r"^[0-9\-\.\)\s]+", "", line))
                    if cleaned:
                        candidates.append(cleaned)
                if candidates:
                    return candidates[:8]
            return []
        except Exception:
            return []

    def suggest_diseases(self, query: str, limit: int = 8) -> list[str]:
        cleaned = self._clean_text(query)
        if not cleaned:
            return self.all_diseases[:limit]

        scored: list[tuple[float, str]] = []
        llm_candidates = self._llm_generate_synonyms(query)

        llm_keys: set[str] = set()
        for candidate in llm_candidates:
            key, score = self._best_dataset_match(candidate)
            if key and score >= 0.35:
                llm_keys.add(key)

        for disease_key, display in self.store.disease_display.items():
            ratio = difflib.SequenceMatcher(None, cleaned, disease_key).ratio()
            token_overlap = len(set(cleaned.split()) & set(disease_key.split()))
            token_score = token_overlap / max(len(cleaned.split()), len(disease_key.split()), 1)
            starts_bonus = 0.2 if disease_key.startswith(cleaned) else 0.0
            contains_bonus = 0.15 if cleaned in disease_key else 0.0
            llm_bonus = 0.35 if disease_key in llm_keys else 0.0
            score = (token_score * 0.5) + (ratio * 0.35) + starts_bonus + contains_bonus + llm_bonus
            if score > 0.12:
                scored.append((score, display))

        scored.sort(key=lambda item: (-item[0], item[1]))
        deduped: list[str] = []
        seen: set[str] = set()
        for _, disease in scored:
            if disease in seen:
                continue
            seen.add(disease)
            deduped.append(disease)
            if len(deduped) >= limit:
                break
        return deduped

    def normalize_disease(self, query: str) -> str:
        """Map user query to the closest disease represented in the dataset."""
        cleaned = self._clean_text(query)
        if not cleaned:
            return ""

        cached = self._normalization_cache.get(cleaned)
        if cached:
            return cached

        if cleaned in self.store.disease_to_proteins:
            normalized = self.store.disease_display.get(cleaned, cleaned)
            self._normalization_cache[cleaned] = normalized
            return normalized

        best_key, best_score = self._best_dataset_match(cleaned)
        llm_candidates = self._llm_generate_synonyms(query)
        llm_best_key = ""
        llm_best_score = 0.0
        for candidate in llm_candidates:
            key, score = self._best_dataset_match(candidate)
            if score > llm_best_score:
                llm_best_key = key
                llm_best_score = score

        # Prefer LLM-guided mapping when lexical matching is only moderately confident.
        if llm_best_key and llm_best_score >= 0.35 and (
            llm_best_score >= (best_score + 0.05) or best_score < 0.65
        ):
            normalized = self.store.disease_display.get(llm_best_key, llm_best_key)
            self._normalization_cache[cleaned] = normalized
            return normalized

        if best_key and best_score >= 0.5:
            normalized = self.store.disease_display.get(best_key, best_key)
            self._normalization_cache[cleaned] = normalized
            return normalized

        # Final fallback keeps behavior deterministic even without LLM.
        if best_key:
            normalized = self.store.disease_display.get(best_key, best_key)
            self._normalization_cache[cleaned] = normalized
            return normalized

        self._normalization_cache[cleaned] = cleaned
        return cleaned

    @staticmethod
    def confidence_label(score: float) -> str:
        if score >= 0.75:
            return "High"
        if score >= 0.5:
            return "Medium"
        return "Low"

    @staticmethod
    def mechanism_group_from_protein(protein: str) -> str:  # noqa: PLR0911
        """Map a gene symbol to its primary signalling mechanism group.

        Coverage is deliberately broad so that diseases outside the three
        curated profiles (HIV/COVID/AD) still receive meaningful pathway
        assignments.  Add new entries here before declaring a protein
        "General".
        """
        key = protein.upper()

        # --- viral entry (HIV + respiratory) --------------------------------
        if key in {"CCR5", "CXCR4", "CD4", "CXCL12", "CXCR6", "CCR2", "CCR7",
                   "ACE2", "TMPRSS2", "ADAM17", "DPP4"}:
            return "Viral entry / host co-receptor axis"

        # --- neurodegeneration ----------------------------------------------
        if key in {"APP", "BACE1", "MAPT", "PSEN1", "PSEN2", "APOE", "TREM2",
                   "ACHE", "CHRNA7", "GRIN2B", "SNCA", "LRRK2", "PINK1",
                   "PARK7", "GBA", "TARDBP", "FUS", "SOD1", "HTT"}:
            return "Neurodegeneration / amyloid-tau axis"

        # --- JAK–STAT / cytokine --------------------------------------------
        if key.startswith(("JAK", "STAT", "IFN", "IL", "CXCL", "CSF",
                            "TGFB", "IFNG", "IFNA", "IFNB")):
            return "JAK-STAT / cytokine signaling"

        # --- NF-kB / inflammation -------------------------------------------
        if key.startswith(("NFKB", "CHUK", "IKBK", "IKBA", "RELA", "RELB",
                            "REL", "TNF", "TNFR", "BIRC", "RIPK")):
            return "NF-kB / inflammatory signaling"

        # --- MAPK / stress --------------------------------------------------
        if key.startswith(("MAPK", "MAP2K", "MAP3K", "ERK", "RAF", "BRAF",
                            "MEK", "P38", "JNK", "DUSP")):
            return "MAPK stress-response signaling"

        # --- innate immune sensing ------------------------------------------
        if key.startswith(("TLR", "TBK", "IRF", "STING", "CGAS", "DDX",
                            "IFIH", "MDA5", "RIG")):
            return "Innate immune sensing"

        # --- PI3K / AKT / mTOR ----------------------------------------------
        if key.startswith(("PIK3", "AKT", "MTOR", "PTEN", "TSC", "RPTOR",
                            "RPS6K", "EIF4E", "PDPK")):
            return "PI3K/AKT/mTOR signaling"

        # --- apoptosis / cell survival --------------------------------------
        if key in {"BCL2", "BCL2L1", "BCL2L11", "BAX", "BAK1", "BID",
                   "PUMA", "NOXA", "MCL1", "BCL2A1", "XIAP"} or key.startswith(
                   ("CASP", "BIRC", "DIABLO", "CYTO")):
            return "Apoptosis / cell survival"

        # --- cell cycle & DNA-damage response -------------------------------
        if key in {"TP53", "MDM2", "MDM4", "RB1", "CDKN1A", "CDKN1B",
                   "CDKN2A", "CDKN2B", "BRCA1", "BRCA2", "ATM", "ATR",
                   "CHEK1", "CHEK2", "RAD51", "PARP1", "WEE1",
                   "AURKA", "AURKB", "PLK1",
                   # DNA replication machinery
                   "POLA1", "POLA2", "POLD1", "POLD2", "POLE", "POLE2",
                   "PCNA", "RFC1", "FEN1", "LIG1", "RRM1", "RRM2",
                   "TYMS", "DHFR"} or key.startswith(
                   ("CDK", "CCND", "CCNE", "CCNB", "E2F")):
            return "Cell cycle / DNA damage response"

        # --- RTK / RAS-RAF oncogenic ----------------------------------------
        if key in {"KRAS", "NRAS", "HRAS", "NF1", "SOS1", "GRB2",
                   "EGFR", "ERBB2", "ERBB3", "ERBB4", "MET", "ALK",
                   "ROS1", "FGFR1", "FGFR2", "FGFR3", "FGFR4",
                   "KIT", "FLT3", "PDGFRA", "PDGFRB", "RET",
                   "ABL1", "BCR", "SRC", "YES1",
                   # TEC-family non-receptor tyrosine kinases
                   "BTK", "ITK", "TEC", "BMX", "TXK"} or key.startswith(
                   ("FGFR", "VEGFR", "PDGFR", "IGFR", "EPHA", "EPHB")):
            return "RTK / RAS-RAF oncogenic signaling"

        # --- nuclear receptor / transcriptional regulation ------------------
        if key in {"AR", "ESR1", "ESR2", "PGR", "PPARA", "PPARG", "PPARD",
                   "RARA", "RARB", "RARG", "RXRA", "VDR", "THRB", "NR3C1",
                   "NR3C2", "HNF4A", "RORC"} or key.startswith(("NR0", "NR1",
                   "NR2", "NR3", "NR4", "NR5", "NR6")):
            return "Nuclear receptor / transcriptional regulation"

        # --- Wnt / β-catenin ------------------------------------------------
        if key in {"CTNNB1", "APC", "AXIN1", "AXIN2", "GSK3B", "GSK3A",
                   "DVL1", "DVL2", "DVL3", "TCF7L2", "LRP5", "LRP6",
                   "SFRP1", "DKK1"} or key.startswith("WNT"):
            return "Wnt / β-catenin signaling"

        # --- Notch ----------------------------------------------------------
        if key.startswith(("NOTCH", "JAG", "DLL", "HES", "HEY")) or key in {
                "RBPJ", "MAML1", "MAML2", "ADAM10"}:
            return "Notch signaling"

        # --- epigenetic regulation ------------------------------------------
        if key.startswith(("EZH", "KDM", "HDAC", "DNMT", "BRD", "KAT",
                            "SETD", "DOT1", "PRMT")) or key in {
                "EED", "SUZ12", "RING1", "BMI1", "ASXL1", "IDH1", "IDH2",
                "TET1", "TET2", "ARID1A"}:
            return "Epigenetic regulation"

        # --- metabolic signalling -------------------------------------------
        if key in {"HMGCR", "FASN", "ACACA", "ACLY", "LDHA", "PKM",
                   "HK1", "HK2", "G6PD", "IDH1", "IDH2", "PHGDH",
                   "NAMPT", "PDHK1", "CPT1A", "ACAD"} or key.startswith(
                   ("AMPK", "PRKAA", "PRKAB", "SIRT", "PPARGC")):
            return "Metabolic signaling"

        # --- ubiquitin / protein degradation --------------------------------
        if key in {"MDM2", "FBXW7", "SKP2", "SPOP", "KEAP1", "CUL3",
                   "VHL", "NEDD4", "SMURF1", "SMURF2", "WWP2",
                   "PSMD", "PSMC"} or key.startswith(
                   ("UBE", "USP", "TRIM", "RNF", "MARCH")):
            return "Ubiquitin / protein degradation"

        # --- chromatin remodelling / SWI-SNF --------------------------------
        if key in {"SMARCA4", "SMARCB1", "ARID1A", "ARID1B", "ARID2",
                   "PBRM1", "SMARCC1", "SMARCC2"}:
            return "Chromatin remodelling"

        return "General protein-target interaction"

    def cluster_proteins(self, proteins: list[str]) -> dict[str, list[str]]:
        clusters: dict[str, list[str]] = defaultdict(list)
        for protein in proteins:
            clusters[self.mechanism_group_from_protein(protein)].append(protein)
        for pathway in clusters:
            clusters[pathway] = sorted(dict.fromkeys(clusters[pathway]))
        return dict(sorted(clusters.items(), key=lambda item: (-len(item[1]), item[0])))

    @staticmethod
    def _normalize_profile(profile: dict[str, float]) -> dict[str, float]:
        total = sum(profile.values())
        if total <= 0:
            return {}
        return {key: value / total for key, value in profile.items()}

    def _get_profile(self, disease: str) -> Optional[DiseaseProfile]:
        """Return the :class:`DiseaseProfile` for *disease*, or ``None``."""
        return self._registry.match(disease)

    def _dynamic_target_relevance_map(self, disease_key: str, proteins: list[str]) -> dict[str, float]:
        """Multi-signal relevance for diseases without a curated profile.

        Three independent signals are combined:
        - *Specificity*: proteins appearing in fewer diseases are more specific
          to this disease context.  Score = 1/log(degree + 1.5).
        - *Druggability*: proteins with more known drugs are better validated
          drug targets.  Score = log(drug_count + 1) / log(MAX_DRUGS).
        - *Direct association*: protein is in the disease's direct protein set
          (bonus +0.20).

        All three are blended and normalised to [0, 1].
        """
        scores: dict[str, float] = {}
        disease_proteins = self.store.disease_to_proteins.get(disease_key, set())
        # Pre-compute max drug count for normalisation (lazy, once per instance
        # would be cleaner but this is called rarely enough to be fine).
        max_drugs = max(
            (len(v) for v in self.store.protein_to_drugs.values()), default=1
        )
        log_max_drugs = math.log(max_drugs + 1)

        for protein in proteins:
            degree = max(1, self._protein_disease_degree.get(protein, 1))
            specificity = 1.0 / math.log(degree + 1.5)

            drug_count = len(self.store.protein_to_drugs.get(protein, set()))
            druggability = math.log(drug_count + 1) / log_max_drugs

            direct_bonus = 0.20 if protein in disease_proteins else 0.0

            scores[protein.upper()] = (
                0.50 * specificity
                + 0.30 * druggability
                + direct_bonus
            )

        if scores:
            max_val = max(scores.values())
            if max_val > 0:
                scores = {k: min(1.0, v / max_val) for k, v in scores.items()}
        return scores

    def _compute_dynamic_disease_priors(self, proteins: list[str]) -> dict[str, float]:
        """Derive mechanism priors from a disease's own protein set.

        Each protein contributes to its mechanism group proportionally to its
        disease-specificity score — specific proteins carry more weight than
        ubiquitous hubs.  This ensures that a leukemia with many BCL2/CDK/RTK
        proteins actually gets apoptosis/cell-cycle/RTK priors, not silence.
        """
        counts: dict[str, float] = defaultdict(float)
        for protein in proteins:
            mechanism = self.mechanism_group_from_protein(protein)
            degree = max(1, self._protein_disease_degree.get(protein, 1))
            specificity_weight = 1.0 / math.log(degree + 1.5)
            counts[mechanism] += specificity_weight
        # Down-weight the catch-all bucket so genuinely classified proteins
        # drive the profile, not an accumulation of unclassified ones.
        if "General protein-target interaction" in counts:
            counts["General protein-target interaction"] *= 0.4
        return self._normalize_profile(dict(counts))

    def _disease_mechanism_profile(self, disease: str, direct_proteins: list[str]) -> dict[str, float]:
        counts: dict[str, float] = defaultdict(float)
        known_targets = self._known_targets_for_disease(disease)
        for protein in direct_proteins:
            mechanism = self.mechanism_group_from_protein(protein)
            weight = 1.5 if protein.upper() in known_targets else 1.0
            counts[mechanism] += weight

        profile = self._get_profile(disease)
        if profile:
            # Curated expert priors get strong injection weight.
            priors = profile.mechanism_priors
            prior_strength = 3.0
        else:
            # For unknown diseases, derive priors from the protein set itself
            # so mechanism alignment scoring is still meaningful.
            priors = self._compute_dynamic_disease_priors(direct_proteins)
            prior_strength = 2.0  # slightly lower than curated

        for mechanism, prior_weight in priors.items():
            counts[mechanism] += prior_weight * prior_strength

        return self._normalize_profile(dict(counts))

    def _drug_mechanism_profile(self, matched_proteins: list[str], primary_target: str) -> dict[str, float]:
        counts: dict[str, float] = defaultdict(float)
        for protein in matched_proteins:
            mechanism = self.mechanism_group_from_protein(protein)
            counts[mechanism] += 1.0
        if primary_target:
            counts[self.mechanism_group_from_protein(primary_target)] += 1.5
        return self._normalize_profile(dict(counts))

    def _infer_lifecycle_stage(self, drug: str, matched_proteins: list[str], disease: str) -> tuple[str, float]:
        profile = self._get_profile(disease)
        if not profile or not profile.lifecycle_stages:
            return "General disease stage", 0.5

        stage_scores: dict[str, float] = defaultdict(float)
        for protein in matched_proteins:
            upper = protein.upper().replace("-", "_")
            for stage, targets in profile.stage_targets.items():
                if upper in targets:
                    stage_scores[stage] += 1.0

        drug_lower = drug.lower()
        for stage, keywords in profile.stage_drug_keywords.items():
            if any(keyword in drug_lower for keyword in keywords):
                stage_scores[stage] += 2.0

        if not stage_scores:
            return "General disease stage", 0.35

        ordered = sorted(stage_scores.items(), key=lambda item: (-item[1], item[0]))
        stage_name, top_score = ordered[0]
        total = sum(stage_scores.values())
        confidence = max(0.3, min(1.0, top_score / max(total, 1e-9)))
        return stage_name, confidence

    def _lifecycle_alignment(self, stage: str, disease: str, stage_confidence: float) -> float:
        profile = self._get_profile(disease)
        priors = profile.lifecycle_priors if profile else {}
        if not priors:
            return 0.5
        prior = priors.get(stage, 0.1)
        score = (0.55 * stage_confidence) + (0.45 * min(1.0, prior * 2.5))
        return max(0.2, min(1.0, score))

    @staticmethod
    def _mechanism_alignment(drug_profile: dict[str, float], disease_profile: dict[str, float]) -> float:
        if not drug_profile or not disease_profile:
            return 0.35
        overlap = 0.0
        for mechanism, weight in drug_profile.items():
            overlap += min(weight, disease_profile.get(mechanism, 0.0))
        # Keep a non-zero floor so candidates with partial evidence remain rankable.
        return max(0.2, min(1.0, overlap))

    def _is_nonspecific_protein(self, protein: str) -> bool:
        key = protein.upper()
        if key.startswith(self._nonspecific_prefixes):
            return True
        return any(token in key for token in self._nonspecific_keywords)

    def _known_targets_for_disease(self, disease: str) -> frozenset[str]:
        profile = self._get_profile(disease)
        if profile:
            return profile.known_targets
        # Dynamic fallback: treat every direct disease protein as a known target.
        disease_key = self._clean_text(disease)
        return frozenset(self.store.disease_to_proteins.get(disease_key, set()))

    def _target_relevance(
        self,
        proteins: list[str],
        disease: str,
        direct_pool: set[str],
        known_targets: frozenset[str],
    ) -> float:
        profile = self._get_profile(disease)
        if profile:
            relevance_map = profile.target_relevance
        else:
            # Data-driven: specificity from how many diseases each protein spans.
            relevance_map = self._dynamic_target_relevance_map(
                self._clean_text(disease), proteins
            )

        scores: list[float] = []
        for protein in proteins:
            upper = protein.upper()
            if upper in relevance_map:
                scores.append(relevance_map[upper])
            elif upper in known_targets:
                scores.append(0.82)
            elif protein in direct_pool:
                scores.append(0.70)
            elif self._is_nonspecific_protein(protein):
                scores.append(0.22)
            else:
                scores.append(0.38)
        return max(0.12, min(1.0, sum(scores) / max(len(scores), 1)))

    def _biological_boost(self, proteins: list[str], disease: str) -> float:
        profile = self._get_profile(disease)
        if profile:
            # Use curated boost values for well-characterised diseases.
            return max(0.0, sum(profile.biological_boost.get(p.upper(), 0.0) for p in proteins))

        # Dynamic fallback: reward proteins that are both specific to few
        # diseases AND directly associated with this disease.  The scale is
        # intentionally smaller than curated boosts (max ~1.5 per protein vs
        # 2–3 for curated) so it nudges rather than dominates.
        disease_key = self._clean_text(disease)
        disease_proteins = self.store.disease_to_proteins.get(disease_key, set())
        total = 0.0
        for protein in proteins:
            if protein not in disease_proteins:
                continue
            degree = max(1, self._protein_disease_degree.get(protein, 1))
            specificity = 1.0 / math.log(degree + 1.5)
            # Scale to ~[0, 1.5] — specific direct proteins get the boost
            total += specificity * 1.5
        return max(0.0, total)

    def _evidence_weight(self, drug: str, disease: str, matched_proteins: list[str], known_targets: frozenset[str]) -> float:
        profile = self._get_profile(disease)
        drug_lower = drug.lower()
        if profile and any(kw in drug_lower for kw in profile.drug_evidence_keywords):
            return 1.5
        if any(p.upper() in known_targets for p in matched_proteins):
            return 1.2
        return 1.0

    @staticmethod
    def _distance_decay(matched_proteins: list[str], direct_pool: set[str]) -> float:
        """Exponential hop penalty: direct=1.0, 1-hop=0.80, 2-hop=0.64.

        Using 0.8^hop instead of 1/avg_len gives steeper, principled decay —
        each additional network hop halves the confidence by ~20%, preventing
        STRING-expanded distant neighbours from swamping direct disease links.
        """
        if not matched_proteins:
            return 0.1
        # Each protein contributes 0.8^hop; average over all matched proteins.
        hop_scores = [1.0 if p in direct_pool else 0.8 for p in matched_proteins]
        return max(0.25, min(1.0, sum(hop_scores) / len(hop_scores)))

    def _kinase_penalty(self, matched_proteins: list[str], drug: str) -> float:
        kinase_like = 0
        for protein in matched_proteins:
            upper = protein.upper()
            if any(marker in upper for marker in self._kinase_markers):
                kinase_like += 1
        drug_lower = drug.lower()
        if kinase_like >= 6 or drug_lower.endswith("nib"):
            return 0.5
        if kinase_like >= 3:
            return 0.75
        return 1.0

    def _cns_factor(self, drug: str, disease: str, mechanism: str, primary_target: str) -> float:
        profile = self._get_profile(disease)
        if profile:
            neuro_disease = profile.neuro_disease
        else:
            disease_key = self._clean_text(disease)
            neuro_disease = any(
                t in disease_key
                for t in ("alzheimer", "dementia", "parkinson", "huntington", "als", "neurodegenerat")
            )
        if not neuro_disease:
            return 1.0

        drug_lower = drug.lower()
        if any(keyword in drug_lower for keyword in self._non_cns_drug_keywords):
            return 0.2
        if primary_target.upper() in {"APP", "BACE1", "MAPT", "ACHE", "PSEN1", "PSEN2"}:
            return 1.0
        if mechanism == "General protein-target interaction":
            return 0.7
        return 0.9

    @staticmethod
    @staticmethod
    def _confidence_from_score(score: float) -> str:
        # Legacy helper — used only inside generate_hypotheses.
        # rank_drugs now calls _calibrated_confidence instead.
        if score >= 0.78:
            return "High"
        if score >= 0.55:
            return "Medium"
        return "Low"

    def _calibrated_confidence(
        self,
        score: float,
        direct_hits: int,
        has_known_target: bool,
        mechanism_alignment: float,
        trust_factor: float,
        source: str,
    ) -> str:
        """Confidence calibrated to biological evidence strength, not just rank.

        Four independent signals are combined:
          direct_hits       — protein is directly in disease set (strongest signal)
          has_known_target  — matches a curated known target for this disease
          mechanism_alignment — drug's mechanism aligns with disease biology
          trust_factor      — fraction of drug-target links verified in dataset

        A drug can have a high ranking score but Low confidence if its score
        was driven by network connectivity rather than direct biological evidence.
        """
        signals: list[float] = []

        # Signal 1: direct protein hit (0 or 1 direct hits → 0.0, any direct → 0.8+)
        if direct_hits >= 2:
            signals.append(1.0)
        elif direct_hits == 1:
            signals.append(0.75)
        else:
            signals.append(0.2)   # expanded-only — weakest

        # Signal 2: known target for disease (curated or dynamic)
        signals.append(0.9 if has_known_target else 0.4)

        # Signal 3: mechanism alignment (already [0, 1])
        signals.append(mechanism_alignment)

        # Signal 4: dataset link verification
        signals.append(trust_factor)

        # Weighted average: direct hits and known targets matter most
        weights = [0.35, 0.30, 0.20, 0.15]
        calibrated = sum(s * w for s, w in zip(signals, weights))

        # No hard caps — let the calibrated signal decide.
        # Signal 2 already penalises missing known targets (0.9 vs 0.4),
        # so the weighted average naturally suppresses confidence when a drug
        # lacks known-target support — without blocking genuine novel discoveries.
        if calibrated >= 0.70:
            return "High"
        if calibrated >= 0.48:
            return "Medium"
        return "Low"

    def _compute_evidence_level(
        self,
        direct_hits: int,
        has_known_target: bool,
        mechanism_alignment: float,
        trust_factor: float,
        source: str,
        known_drug: bool,
    ) -> tuple[str, int]:
        """Return (evidence_level, tier) describing hypothesis quality.

        Tier 1 — Strong:      Direct target + known target + strong mechanism
        Tier 2 — Hypothesis:  Some direct signal or good mechanism, but not both
        Tier 3 — Exploratory: Network-only or low verification
        """
        # Strong: needs direct hit AND known target AND decent mechanism alignment
        if direct_hits >= 1 and has_known_target and mechanism_alignment >= 0.55 and trust_factor >= 0.5:
            return "Strong", 1

        # Hypothesis: at least one strong signal
        if (direct_hits >= 1 and mechanism_alignment >= 0.45) or \
           (has_known_target and trust_factor >= 0.5) or \
           (mechanism_alignment >= 0.65 and trust_factor >= 0.5):
            return "Hypothesis", 2

        # Exploratory: everything else
        return "Exploratory", 3

    def _compute_caveats(
        self,
        drug: str,
        matched_proteins: list[str],
        direct_pool: set[str],
        known_targets: frozenset[str],
        mechanism: str,
        trust_factor: float,
        kinase_penalty: float,
        cns_factor: float,
        direct_hits: int,
    ) -> list[str]:
        """Generate specific, articulable reasons why this candidate may be a false positive.

        Each caveat is a short, factual sentence a researcher would write in a
        methods section — not a vague disclaimer.  Returns an empty list when
        no specific red flags are detected.
        """
        caveats: list[str] = []

        if direct_hits == 0:
            caveats.append(
                "No direct disease-protein association — signal is derived entirely "
                "from STRING network expansion (indirect evidence only)."
            )

        if trust_factor < 0.5:
            unverified = sum(
                1 for p in matched_proteins
                if drug not in self.store.protein_to_drugs.get(p, set())
            )
            caveats.append(
                f"{unverified} of {len(matched_proteins)} drug-target links are not present "
                "in the ground-truth protein-drug dataset — possible network artefact."
            )

        if mechanism == "General protein-target interaction":
            caveats.append(
                "Primary target could not be assigned to a specific disease pathway — "
                "mechanistic relevance is unresolved."
            )

        nonspecific = [p for p in matched_proteins if self._is_nonspecific_protein(p)]
        if nonspecific and len(nonspecific) >= len(matched_proteins) // 2:
            caveats.append(
                f"Majority of matched proteins ({', '.join(nonspecific[:3])}) are metabolic "
                "or transport enzymes with broad disease associations, not disease-specific drivers."
            )

        if kinase_penalty < 0.75:
            caveats.append(
                "Drug is a broad-spectrum kinase inhibitor — high off-target activity may "
                "confound disease-specific interpretation."
            )

        if cns_factor < 0.5:
            caveats.append(
                "Drug is associated with poor CNS penetration — may be unsuitable for "
                "neurological disease contexts."
            )

        known_hit = any(p.upper() in known_targets for p in matched_proteins)
        if not known_hit and direct_hits > 0:
            caveats.append(
                "Matched proteins are in the disease protein set but not in the curated "
                "known-target list — association may reflect comorbidity rather than causality."
            )

        return caveats

    def generate_hypotheses(
        self,
        disease: str,
        clusters: dict[str, list[str]],
        candidates: list[CandidateDrug],
    ) -> list[HypothesisSummary]:
        grouped_candidates: dict[str, list[CandidateDrug]] = defaultdict(list)
        for candidate in candidates:
            grouped_candidates[candidate.mechanism_group or "General protein-target interaction"].append(candidate)

        hypotheses: list[HypothesisSummary] = []
        for pathway, proteins in clusters.items():
            pathway_candidates = grouped_candidates.get(pathway, [])
            top_drugs = [item.drug for item in pathway_candidates[:3]]
            top_score = pathway_candidates[0].score if pathway_candidates else 0.0
            confidence = self._confidence_from_score(top_score)
            if pathway_candidates:
                summary = (
                    f"Targeting {pathway} may help modulate {disease} through {len(proteins)} supporting proteins. "
                    f"Top drugs: {', '.join(top_drugs) if top_drugs else 'none'}."
                )
                status = "supported"
            else:
                summary = (
                    f"{pathway} shows biological relevance for {disease}, but no high-confidence drug candidates "
                    "passed the current ranking thresholds."
                )
                status = "discarded"

            hypotheses.append(
                HypothesisSummary(
                    pathway=pathway,
                    supporting_proteins=proteins,
                    candidate_drugs=top_drugs,
                    top_score=top_score,
                    confidence=confidence,
                    status=status,
                    summary=summary,
                )
            )

        hypotheses.sort(
            key=lambda item: (
                item.status != "supported",
                item.pathway == "General protein-target interaction",
                -item.top_score,
                -len(item.supporting_proteins),
                item.pathway,
            )
        )
        return hypotheses

    @staticmethod
    def build_summary(
        disease: str,
        normalized_disease: str,
        reasoning_trace: list[str],
        hypotheses: list[HypothesisSummary],
        candidates: list[CandidateDrug],
    ) -> str:
        if not candidates:
            return (
                f"I explored the disease query {disease} as {normalized_disease}, but the current thresholds did not "
                "yield high-confidence repurposing candidates."
            )

        top_candidate = candidates[0]
        active_hypotheses = [item for item in hypotheses if item.status == "supported"]
        active_hypotheses.sort(
            key=lambda item: (
                item.pathway == "General protein-target interaction",
                -item.top_score,
                -len(item.supporting_proteins),
                item.pathway,
            )
        )
        explored = len(reasoning_trace)
        if active_hypotheses:
            top_hypothesis = active_hypotheses[0]
            uncertainty_note = ""
            if top_hypothesis.pathway == "General protein-target interaction":
                uncertainty_note = (
                    " The agent notes that while this broad interaction hypothesis is statistically strong, "
                    "it is less pathway-specific and may include off-target effects."
                )
            return (
                f"I explored {len(active_hypotheses)} biological strategies for {normalized_disease}. "
                f"After iterative critique and refinement, I retained {len(candidates)} high-confidence candidates. "
                f"The strongest hypothesis centers on {top_hypothesis.pathway} with "
                f"{len(top_hypothesis.supporting_proteins)} supporting proteins. The top candidate is "
                f"{top_candidate.drug}, supported by {top_candidate.support_count} proteins and scored "
                f"{top_candidate.score:.3f}. Reasoning steps recorded: {explored}.{uncertainty_note}"
            )
        return (
            f"I explored {normalized_disease} through the current ranking loop and kept {len(candidates)} candidates. "
            f"The top candidate is {top_candidate.drug} with score {top_candidate.score:.3f}."
        )

    def expand_proteins(
        self,
        proteins: list[str],
        species: int = 9606,
        required_score: int = 700,
        limit: int = 60,
        min_interaction_score: float = 0.7,
    ) -> tuple[list[str], dict[str, float]]:
        """Expand proteins through STRING interaction partners."""
        if not proteins:
            return [], {}

        string_ids = self.string_client.map_to_string_ids(proteins, species=species)
        if not string_ids:
            return [], {}

        adaptive_limit = min(limit, max(20, len(proteins) * 20))
        partner_scores = self.string_client.interaction_partners_with_scores(
            string_ids,
            species=species,
            required_score=required_score,
            limit=adaptive_limit,
        )
        filtered_scores: dict[str, float] = {
            protein: score
            for protein, score in partner_scores.items()
            if score >= min_interaction_score
        }
        return sorted(filtered_scores.keys()), filtered_scores

    def get_candidate_drugs(self, proteins: list[str]) -> dict[str, set[str]]:
        """Find drugs that target any of the supplied proteins."""
        drug_to_proteins: dict[str, set[str]] = defaultdict(set)
        for protein in proteins:
            for drug in self.store.protein_to_drugs.get(protein, set()):
                drug_to_proteins[drug].add(protein)
        return drug_to_proteins

    def rank_drugs(
        self,
        drugs: dict[str, set[str]],
        proteins: list[str],
        direct_proteins: list[str],
        disease: str,
        interaction_scores: dict[str, float] | None = None,
        pathway_sizes: dict[str, int] | None = None,
        driver_proteins: frozenset[str] | None = None,
    ) -> list[CandidateDrug]:
        """Rank drugs using a biologically-informed multiplicative score.

        Args:
            driver_proteins: LLM-assessed causal driver proteins.  Drugs whose
                matched proteins include a driver receive a 1.25× score boost,
                promoting disease-causal mechanism over mere association.
        """
        if not drugs:
            return []

        interaction_scores = interaction_scores or {}
        pathway_sizes = pathway_sizes or {}
        driver_pool: frozenset[str] = driver_proteins or frozenset()
        direct_pool = set(direct_proteins)
        known_targets = self._known_targets_for_disease(disease)
        disease_mechanism_profile = self._disease_mechanism_profile(disease, direct_proteins)

        ranked: list[CandidateDrug] = []
        for drug, matched in drugs.items():
            matched_sorted = sorted(matched)
            direct_hits = len(matched & direct_pool)
            # Controlled graph signal: support is normalized by local connectivity.
            degree_values = [
                len(self.store.protein_to_drugs.get(protein, set())) + self._protein_disease_degree.get(protein, 0)
                for protein in matched_sorted
            ]
            avg_degree = sum(degree_values) / max(len(degree_values), 1)
            connectivity_raw = len(matched_sorted) / max(math.log(avg_degree + 1.0), 1.0)
            connectivity_score = max(0.05, min(1.0, connectivity_raw / 2.8))

            primary_target = next((protein for protein in matched_sorted if protein in direct_pool), matched_sorted[0])
            mechanism = self.mechanism_group_from_protein(primary_target)
            pathway_size = max(pathway_sizes.get(mechanism, 0), 1)
            pathway_specificity = 1.0 / max(math.log(pathway_size + 1.0), 1.0)
            drug_mechanism_profile = self._drug_mechanism_profile(matched_sorted, primary_target)
            mechanism_alignment = self._mechanism_alignment(drug_mechanism_profile, disease_mechanism_profile)
            primary_mechanism_weight = disease_mechanism_profile.get(mechanism, 0.0)
            mechanism_alignment = max(mechanism_alignment, 0.2 + (0.8 * primary_mechanism_weight))
            if mechanism == "General protein-target interaction":
                strongest_specific = max(
                    (
                        value
                        for key, value in disease_mechanism_profile.items()
                        if key != "General protein-target interaction"
                    ),
                    default=0.0,
                )
                if strongest_specific >= 0.3:
                    mechanism_alignment *= 0.75
            mechanism_alignment = max(0.15, min(1.0, mechanism_alignment))
            lifecycle_stage, stage_confidence = self._infer_lifecycle_stage(drug, matched_sorted, disease)
            lifecycle_alignment = self._lifecycle_alignment(lifecycle_stage, disease, stage_confidence)

            nonspecific_hits = sum(1 for protein in matched_sorted if self._is_nonspecific_protein(protein))
            nonspecific_fraction = nonspecific_hits / max(len(matched_sorted), 1)
            primary_nonspecific = self._is_nonspecific_protein(primary_target)
            hub_penalty = min(0.65, nonspecific_fraction * 0.65)

            target_relevance = self._target_relevance(matched_sorted, disease, direct_pool, known_targets)
            distance_decay = self._distance_decay(matched_sorted, direct_pool)
            kinase_penalty = self._kinase_penalty(matched_sorted, drug)
            cns_factor = self._cns_factor(drug, disease, mechanism, primary_target)
            biological_boost = self._biological_boost(matched_sorted, disease)
            evidence_weight = self._evidence_weight(drug, disease, matched_sorted, known_targets)

            # ── Dataset trust filter ─────────────────────────────────────────
            # Verify each matched protein → drug link against the ground-truth
            # protein_to_drugs map.  Proteins whose link to this drug is NOT in
            # the dataset are STRING-expansion artefacts or LLM hallucinations;
            # they receive no credit.  If NONE of the links are verified, apply
            # a heavy unverified-link penalty so the drug still appears (for
            # transparency) but ranks well below dataset-verified candidates.
            verified = sum(
                1 for p in matched_sorted
                if drug in self.store.protein_to_drugs.get(p, set())
                or drug.upper() in {d.upper() for d in self.store.protein_to_drugs.get(p, set())}
            )
            total_matched = max(len(matched_sorted), 1)
            verification_ratio = verified / total_matched
            # verified_fraction penalty: 1.0 if all links verified, 0.2 if none.
            trust_factor = max(0.2, verification_ratio)

            # Remove strongly nonspecific hits unless they are directly disease-linked or hit known drivers.
            has_known_target = any(protein.upper() in known_targets for protein in matched_sorted)
            if primary_nonspecific and not has_known_target and direct_hits == 0:
                continue

            # ── General pathway soft penalty ─────────────────────────────────
            # A drug whose primary target falls into "General protein-target
            # interaction" has no resolved disease pathway.  Rather than a flat
            # penalty (which over-punishes specific proteins that happen to be
            # unclassified), we scale the penalty by the *inverse* of the target's
            # disease-specificity: a rare, disease-specific protein that lacks a
            # pathway label gets a small penalty; a ubiquitous hub gets a larger one.
            #   specificity = 1/log(degree + 1.5):  range ≈ (0.2, 1.1)
            #   multiplier  = 0.7 + 0.3 * min(1, specificity): range [0.70, 1.0]
            #   penalty     = 1 − multiplier:                   range [0.00, 0.30]
            broad_pathway_penalty = 0.0
            if mechanism == "General protein-target interaction":
                degree_pt = max(1, self._protein_disease_degree.get(primary_target, 1))
                pt_specificity = min(1.0, 1.0 / math.log(degree_pt + 1.5))
                multiplier = 0.7 + 0.3 * pt_specificity
                broad_pathway_penalty = 1.0 - multiplier   # [0.00, 0.30]

            # ── Driver-ratio boost ───────────────────────────────────────────
            # Count how many matched proteins are known targets (curated or
            # dynamic) — i.e. disease-relevant proteins.  A drug with more
            # disease-specific target hits gets a controlled boost.
            # No penalty for zero driver hits — data may simply not have curated
            # known targets; other signals (direct_hits, trust) handle that.
            #   driver_ratio=0 → multiply by 1.0 (neutral)
            #   driver_ratio=1 → multiply by 1.5 (moderate boost)
            driver_hit_count = sum(1 for p in matched_sorted if p.upper() in known_targets)
            driver_ratio = driver_hit_count / total_matched
            driver_ratio_factor = 1.0 + 0.5 * driver_ratio   # range [1.0, 1.5]

            base_score = (
                connectivity_score
                * pathway_specificity
                * distance_decay
                * target_relevance
                * evidence_weight
                * cns_factor
                * kinase_penalty
                * mechanism_alignment
                * lifecycle_alignment
                * trust_factor           # dataset-backed link verification
                * driver_ratio_factor    # causality signal: driver proteins dominate
            )

            if has_known_target:
                base_score *= 1.15
            if primary_target.upper() in known_targets:
                base_score *= 1.08

            # LLM driver boost: controlled 1.3× advisory signal.
            # The driver_ratio_factor already rewards driver coverage from data.
            # This adds a small LLM-advisory nudge for confirmed causal proteins,
            # but cannot override the data-driven score on its own.
            if driver_pool and any(p.upper() in driver_pool for p in matched_sorted):
                base_score *= 1.3

            base_score *= (1.0 - hub_penalty)
            base_score *= (1.0 - broad_pathway_penalty)

            # Amplify true disease signal while keeping calibration bounded.
            raw_score = (base_score + biological_boost) / (1.0 + biological_boost)

            # Smooth bounded projection to [0, 1] for stable ranking/confidence labels.
            score = 1.0 - math.exp(-4.0 * max(raw_score, 0.0))
            score = max(0.0, min(1.0, score))
            source = "direct+string-expanded" if any(p not in direct_pool for p in matched) else "direct"

            # ── calibrated confidence + evidence tier + caveats ──────────────
            calibrated_conf = self._calibrated_confidence(
                score=score,
                direct_hits=direct_hits,
                has_known_target=has_known_target,
                mechanism_alignment=mechanism_alignment,
                trust_factor=trust_factor,
                source=source,
            )
            evidence_level, tier = self._compute_evidence_level(
                direct_hits=direct_hits,
                has_known_target=has_known_target,
                mechanism_alignment=mechanism_alignment,
                trust_factor=trust_factor,
                source=source,
                known_drug=any(
                    kw in drug.lower()
                    for profile in self._registry.all_profiles.values()
                    for kw in profile.drug_evidence_keywords
                ),
            )
            caveats = self._compute_caveats(
                drug=drug,
                matched_proteins=matched_sorted,
                direct_pool=direct_pool,
                known_targets=known_targets,
                mechanism=mechanism,
                trust_factor=trust_factor,
                kinase_penalty=kinase_penalty,
                cns_factor=cns_factor,
                direct_hits=direct_hits,
            )

            ranked.append(
                CandidateDrug(
                    drug=drug,
                    matched_proteins=matched_sorted,
                    source=source,
                    support_count=len(matched_sorted),
                    score=round(score, 4),
                    confidence=calibrated_conf,
                    evidence_level=evidence_level,
                    tier=tier,
                    primary_target=primary_target,
                    mechanism_group=mechanism,
                    mechanism_alignment=round(mechanism_alignment, 4),
                    lifecycle_stage=lifecycle_stage,
                    lifecycle_alignment=round(lifecycle_alignment, 4),
                    explanation=None,
                    caveats=caveats,
                )
            )

        ranked.sort(key=lambda item: (-item.score, -item.support_count, item.drug))
        return ranked

    def diversify_candidates(self, ranked: list[CandidateDrug], top_k: int | None = None) -> list[CandidateDrug]:
        if not ranked:
            return []
        limit = top_k if top_k is not None else self.top_k

        diversified: list[CandidateDrug] = []
        seen_targets: set[str] = set()
        for candidate in ranked:
            target = (candidate.primary_target or "").upper()
            if target and target in seen_targets:
                continue
            diversified.append(candidate)
            if target:
                seen_targets.add(target)
            if len(diversified) >= limit:
                break

        if len(diversified) < limit:
            chosen = {entry.drug for entry in diversified}
            for candidate in ranked:
                if candidate.drug in chosen:
                    continue
                diversified.append(candidate)
                if len(diversified) >= limit:
                    break
        return diversified

    def explain(
        self,
        drug: str,
        disease: str,
        path: list[str],
        mechanism_group: str | None = None,
        mechanism_alignment: float | None = None,
        lifecycle_stage: str | None = None,
        lifecycle_alignment: float | None = None,
    ) -> str:
        """Create an interpretable path-based explanation for a ranked candidate."""
        pathway = mechanism_group or (self.mechanism_group_from_protein(path[0]) if path else "General protein-target interaction")
        alignment_note = ""
        if mechanism_alignment is not None:
            alignment_note = f" Mechanism alignment score: {mechanism_alignment:.2f}."
        stage_note = ""
        if lifecycle_stage:
            stage_note = f" Lifecycle stage: {lifecycle_stage}."
            if lifecycle_alignment is not None:
                stage_note += f" Stage alignment score: {lifecycle_alignment:.2f}."

        if not path:
            return (
                f"{drug} is linked to {disease} through inferred protein evidence in {pathway}."
                f"{alignment_note}{stage_note}"
            )

        first = path[0]
        if len(path) == 1:
            return (
                f"{drug} targets {first}, a disease-linked protein for {disease}. "
                f"This supports a direct repurposing hypothesis via {pathway}.{alignment_note}{stage_note}"
            )
        second = path[1]
        return (
            f"{drug} targets {first}, which interacts with {second}; this multi-hop chain links the drug "
            f"to {disease} through {pathway}.{alignment_note}{stage_note}"
        )

    def _attach_explanations(
        self,
        disease: str,
        direct_proteins: set[str],
        candidates: list[CandidateDrug],
    ) -> list[CandidateDrug]:
        for candidate in candidates:
            path = self._build_path(candidate.matched_proteins, direct_proteins)
            candidate.explanation = self.explain(
                candidate.drug,
                disease,
                path,
                mechanism_group=candidate.mechanism_group,
                mechanism_alignment=candidate.mechanism_alignment,
                lifecycle_stage=candidate.lifecycle_stage,
                lifecycle_alignment=candidate.lifecycle_alignment,
            )
        return candidates

    @staticmethod
    def _build_path(matched_proteins: Iterable[str], direct_proteins: set[str]) -> list[str]:
        matched = list(matched_proteins)
        if not matched:
            return []
        direct = [p for p in matched if p in direct_proteins]
        expanded = [p for p in matched if p not in direct_proteins]
        if direct and expanded:
            return [direct[0], expanded[0]]
        return [matched[0]]

    def _select_strategy(self, direct_count: int, expand_with_string: bool) -> tuple[str, int, int, float, int]:
        if not expand_with_string:
            return ("direct_only", 1000, 0, 0.50, 2)
        if direct_count < 5:
            return ("deep_exploration", 600, 120, 0.35, 4)
        return ("focused_search", 800, 50, 0.50, 3)

    @staticmethod
    def _needs_stronger_diversity(candidates: list[CandidateDrug]) -> bool:
        if len(candidates) < 4:
            return False
        mechanism_counts: dict[str, int] = defaultdict(int)
        for item in candidates:
            mechanism_counts[item.mechanism_group or "General protein-target interaction"] += 1
        dominant = max(mechanism_counts.values()) if mechanism_counts else 0
        return dominant >= max(4, int(len(candidates) * 0.7))

    def _apply_stronger_diversity(self, ranked: list[CandidateDrug], limit: int) -> list[CandidateDrug]:
        diversified: list[CandidateDrug] = []
        seen_mechanisms: set[str] = set()
        seen_targets: set[str] = set()

        for candidate in ranked:
            mechanism = (candidate.mechanism_group or "General protein-target interaction").upper()
            target = (candidate.primary_target or "").upper()
            if mechanism in seen_mechanisms:
                continue
            diversified.append(candidate)
            seen_mechanisms.add(mechanism)
            if target:
                seen_targets.add(target)
            if len(diversified) >= limit:
                return diversified

        for candidate in ranked:
            mechanism = (candidate.mechanism_group or "General protein-target interaction").upper()
            target = (candidate.primary_target or "").upper()
            if target and target in seen_targets:
                continue
            if mechanism in seen_mechanisms and len(diversified) >= max(3, limit // 2):
                continue
            diversified.append(candidate)
            if target:
                seen_targets.add(target)
            if len(diversified) >= limit:
                break
        return diversified

    def _critique_attempt(
        self,
        ranked: list[CandidateDrug],
        filtered: list[CandidateDrug],
        shortlisted: list[CandidateDrug],
        expanded_count: int,
        threshold: float,
    ) -> list[str]:
        critiques: list[str] = []
        if not ranked:
            critiques.append("no_candidates")
            return critiques

        top_score = ranked[0].score
        if not filtered or top_score < (threshold + 0.05):
            critiques.append("low_confidence")
        if expanded_count == 0:
            critiques.append("weak_network_signal")
        if self._needs_stronger_diversity(shortlisted):
            critiques.append("low_diversity")
        return critiques

    def refine_search(
        self,
        disease: str,
        threshold: float = 0.7,
        species: int = 9606,
        expand_with_string: bool = True,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> ReasoningResult:
        """Agent behavior: relax thresholds and broaden expansion when results are weak."""
        reasoning_trace: list[str] = _ProgressTrace(on_progress)
        normalized = self.normalize_disease(disease)
        reasoning_trace.append(f"Normalized disease query to {normalized or 'an empty value'}.")
        disease_key = self._clean_text(normalized)
        direct_proteins = sorted(self.store.disease_to_proteins.get(disease_key, set()))
        if not direct_proteins:
            reasoning_trace.append("No direct disease proteins were found in the dataset.")
            return ReasoningResult(
                disease=disease,
                normalized_disease=normalized,
                direct_proteins=[],
                expanded_proteins=[],
                candidates=[],
                strategy="No disease match in dataset",
                reasoning_trace=reasoning_trace,
                hypotheses=[],
                summary=f"No direct dataset match was found for {normalized or disease}.",
            )

        reasoning_trace.append(f"Retrieved {len(direct_proteins)} direct disease proteins.")

        strategy_name, required_score, limit, current_threshold, max_attempts = self._select_strategy(
            direct_count=len(direct_proteins),
            expand_with_string=expand_with_string,
        )
        current_threshold = min(current_threshold, max(0.2, threshold - 0.15))
        if strategy_name == "focused_search":
            reasoning_trace.append(
                f"The agent observed a strong direct protein signal ({len(direct_proteins)} proteins), so it "
                "avoided deep exploration and chose a focused search strategy to reduce noise and prioritize precision. "
                f"Initial controls: required_score={required_score}, limit={limit}, threshold={current_threshold:.2f}."
            )
        elif strategy_name == "deep_exploration":
            reasoning_trace.append(
                f"The agent observed limited direct evidence ({len(direct_proteins)} proteins), so it selected "
                "a deep exploration strategy to expand multi-hop hypotheses before ranking. "
                f"Initial controls: required_score={required_score}, limit={limit}, threshold={current_threshold:.2f}."
            )
        else:
            reasoning_trace.append(
                "STRING expansion is disabled, so the agent selected a direct-only strategy focused on "
                "high-confidence direct links. "
                f"Initial controls: required_score={required_score}, limit={limit}, threshold={current_threshold:.2f}."
            )

        best_non_filtered: ReasoningResult | None = None
        best_non_filtered_score = -1.0

        for attempt_index in range(1, max_attempts + 1):
            reasoning_trace.append(
                f"Attempt {attempt_index}: required_score={required_score}, limit={limit}, threshold={current_threshold:.2f}."
            )
            expanded_proteins: list[str] = []
            interaction_scores: dict[str, float] = {protein: 1.0 for protein in direct_proteins}
            if limit > 0:
                expanded_proteins, expanded_scores = self.expand_proteins(
                    direct_proteins,
                    species=species,
                    required_score=required_score,
                    limit=limit,
                    min_interaction_score=max(0.7, required_score / 1000.0),
                )
                removed_nonspecific = [protein for protein in expanded_proteins if self._is_nonspecific_protein(protein)]
                if removed_nonspecific:
                    expanded_proteins = [protein for protein in expanded_proteins if not self._is_nonspecific_protein(protein)]
                    expanded_scores = {
                        protein: score for protein, score in expanded_scores.items() if protein in expanded_proteins
                    }
                    reasoning_trace.append(
                        f"Filtered {len(removed_nonspecific)} likely non-specific hub proteins from STRING expansion."
                    )
                interaction_scores.update(expanded_scores)
                reasoning_trace.append(
                    f"Expanded to {len(expanded_proteins)} STRING proteins with {len(expanded_scores)} scored interactions."
                )
            else:
                reasoning_trace.append("Skipped STRING expansion for the fallback direct-only pass.")

            proteins_for_drugs = sorted(set(direct_proteins) | set(expanded_proteins))
            reasoning_trace.append(
                f"Constructed a protein pool of {len(proteins_for_drugs)} nodes for candidate retrieval."
            )
            clusters = self.cluster_proteins(proteins_for_drugs)
            if clusters:
                reasoning_trace.append(
                    "Identified pathway clusters: "
                    + ", ".join(f"{pathway} ({len(items)})" for pathway, items in clusters.items())
                )
            candidate_map = self.get_candidate_drugs(proteins_for_drugs)
            reasoning_trace.append(f"Generated {len(candidate_map)} candidate drugs before ranking."
            )
            ranked = self.rank_drugs(
                drugs=candidate_map,
                proteins=proteins_for_drugs,
                direct_proteins=direct_proteins,
                disease=normalized,
                interaction_scores=interaction_scores,
                pathway_sizes={pathway: len(nodes) for pathway, nodes in clusters.items()},
            )
            reasoning_trace.append(
                f"Ranked {len(ranked)} candidates with mechanism-aware scoring (network evidence + biological alignment)."
            )

            if ranked:
                top_score = ranked[0].score
                if len(ranked) > self.top_k:
                    reasoning_trace.append(
                        f"Candidate pool was large, so diversity filtering was applied to reduce redundancy to {self.top_k}."
                    )
                should_replace = top_score > best_non_filtered_score
                if (
                    not should_replace
                    and best_non_filtered is not None
                    and expand_with_string
                    and expanded_proteins
                    and not best_non_filtered.expanded_proteins
                ):
                    should_replace = True
                if should_replace:
                    best_non_filtered_score = top_score
                    shortlisted = self.diversify_candidates(ranked, top_k=self.top_k)
                    hypotheses = self.generate_hypotheses(
                        disease=normalized,
                        clusters=clusters,
                        candidates=shortlisted,
                    )
                    best_non_filtered = ReasoningResult(
                        disease=disease,
                        normalized_disease=normalized,
                        direct_proteins=direct_proteins,
                        expanded_proteins=expanded_proteins,
                        candidates=self._attach_explanations(
                            disease=normalized,
                            direct_proteins=set(direct_proteins),
                            candidates=shortlisted,
                        ),
                        strategy=(
                            f"fallback_top_score={top_score:.3f}, required_score={required_score}, "
                            f"expansion={'on' if limit > 0 else 'off'}"
                        ),
                        reasoning_trace=reasoning_trace[:],
                        hypotheses=hypotheses,
                        summary=self.build_summary(
                            disease=disease,
                            normalized_disease=normalized,
                            reasoning_trace=reasoning_trace,
                            hypotheses=hypotheses,
                            candidates=shortlisted,
                        ),
                    )

            filtered = [item for item in ranked if item.score >= current_threshold]
            shortlisted = self.diversify_candidates(filtered if filtered else ranked, top_k=self.top_k)

            critiques = self._critique_attempt(
                ranked=ranked,
                filtered=filtered,
                shortlisted=shortlisted,
                expanded_count=len(expanded_proteins),
                threshold=current_threshold,
            )

            if "low_diversity" in critiques:
                shortlisted = self._apply_stronger_diversity(filtered if filtered else ranked, self.top_k)
                reasoning_trace.append(
                    "Self-critique detected high mechanism redundancy, so stronger diversity constraints were applied."
                )

            must_refine_broad_signal = (
                attempt_index == 1
                and expand_with_string
                and len(candidate_map) > 1200
                and len(shortlisted) >= self.top_k
            )
            if must_refine_broad_signal:
                critiques.append("broad_candidate_space")
                reasoning_trace.append(
                    "Attempt 1 produced a very broad candidate space, so the agent scheduled a refinement pass "
                    "to increase specificity before finalizing recommendations."
                )

            hypotheses = self.generate_hypotheses(
                disease=normalized,
                clusters=clusters,
                candidates=shortlisted,
            )
            if hypotheses:
                pathway_previews: list[str] = []
                for item in hypotheses[:4]:
                    pathway_previews.append(
                        f"{item.pathway}: {item.confidence.lower()} support ({len(item.supporting_proteins)} proteins)"
                    )
                reasoning_trace.append(
                    "The agent evaluated competing biological hypotheses and prioritized by evidence density: "
                    + "; ".join(pathway_previews)
                    + "."
                )
            explained = self._attach_explanations(
                disease=normalized,
                direct_proteins=set(direct_proteins),
                candidates=shortlisted,
            )

            reasoning_trace.append(
                f"Threshold {current_threshold:.2f} kept {len(filtered)} candidates; shortlist size is {len(shortlisted)} after diversity controls."
            )

            if not critiques or (
                "low_diversity" in critiques and len(critiques) == 1 and filtered
            ):
                if attempt_index == 1 and max_attempts > 1 and expand_with_string and len(candidate_map) > 500:
                    reasoning_trace.append(
                        "A quick validation pass is triggered to confirm stability of top mechanisms under tighter controls."
                    )
                    current_threshold = min(0.9, current_threshold + 0.03)
                    required_score = min(950, required_score + 30)
                    limit = max(35, limit)
                    continue
                strategy = (
                    f"strategy={strategy_name}, threshold={current_threshold:.2f}, required_score={required_score}, "
                    f"expansion={'on' if limit > 0 else 'off'}, top_k={self.top_k}, diversity=strong"
                )
                return ReasoningResult(
                    disease=disease,
                    normalized_disease=normalized,
                    direct_proteins=direct_proteins,
                    expanded_proteins=expanded_proteins,
                    candidates=explained,
                    strategy=strategy,
                    reasoning_trace=reasoning_trace,
                    hypotheses=hypotheses,
                    summary=self.build_summary(
                        disease=disease,
                        normalized_disease=normalized,
                        reasoning_trace=reasoning_trace,
                        hypotheses=hypotheses,
                        candidates=explained,
                    ),
                )

            reasoning_trace.append(
                "Self-critique findings: " + ", ".join(critiques) + "."
            )
            if "low_confidence" in critiques:
                current_threshold = max(0.2, current_threshold - 0.08)
                required_score = max(450, required_score - 100)
                limit = max(limit, 80 if expand_with_string else 0)
                reasoning_trace.append(
                    "Adapted strategy for low confidence: lowered threshold, relaxed STRING score, and widened partner search."
                )
            if "weak_network_signal" in critiques and expand_with_string:
                limit = max(limit, 140)
                required_score = max(450, required_score - 50)
                reasoning_trace.append(
                    "Adapted strategy for weak network signal: increased expansion depth and accepted broader interaction confidence."
                )
            if "broad_candidate_space" in critiques:
                current_threshold = min(0.9, current_threshold + 0.05)
                required_score = min(980, required_score + 80)
                limit = max(30, int(limit * 0.8)) if limit > 0 else 0
                reasoning_trace.append(
                    "Adapted strategy for overly broad evidence: tightened score thresholds, strengthened STRING confidence, "
                    "and narrowed partner breadth for a higher-specificity second pass."
                )

            best_non_filtered = ReasoningResult(
                disease=disease,
                normalized_disease=normalized,
                direct_proteins=direct_proteins,
                expanded_proteins=expanded_proteins,
                candidates=explained,
                strategy=(
                    f"fallback_strategy={strategy_name}, threshold={current_threshold:.2f}, required_score={required_score}, "
                    f"expansion={'on' if limit > 0 else 'off'}"
                ),
                reasoning_trace=reasoning_trace[:],
                hypotheses=hypotheses,
                summary=self.build_summary(
                    disease=disease,
                    normalized_disease=normalized,
                    reasoning_trace=reasoning_trace,
                    hypotheses=hypotheses,
                    candidates=explained,
                ),
            )

        if best_non_filtered is not None:
            return best_non_filtered

        return ReasoningResult(
            disease=disease,
            normalized_disease=normalized,
            direct_proteins=direct_proteins,
            expanded_proteins=[],
            candidates=[],
            strategy=f"strategy={strategy_name}, outcome=no candidates passed iterative thresholds",
            reasoning_trace=reasoning_trace + ["No candidates passed iterative thresholds."],
            hypotheses=self.generate_hypotheses(normalized, self.cluster_proteins(direct_proteins), []),
            summary=f"I explored {normalized} but no candidates passed the iterative thresholds.",
        )

    def agent(
        self,
        query: str,
        species: int = 9606,
        expand_with_string: bool = True,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> ReasoningResult:
        """Main orchestrator: understand, expand, reason, rank, explain, return."""
        return self.refine_search(
            disease=query,
            threshold=0.7,
            species=species,
            expand_with_string=expand_with_string,
            on_progress=on_progress,
        )

    def run(
        self,
        disease: str,
        species: int = 9606,
        expand_with_string: bool = False,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> ReasoningResult:
        return self.agent(
            query=disease,
            species=species,
            expand_with_string=expand_with_string,
            on_progress=on_progress,
        )
