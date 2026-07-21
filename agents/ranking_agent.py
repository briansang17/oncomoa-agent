"""
OncoMOA Agent — Biomarker Ranking Agent
Implements the deterministic evidence scoring engine.
Aggregates all normalized evidence into per-biomarker CandidateBiomarker objects,
computes composite scores, and returns a ranked list ready for LLM synthesis.

Scoring weights (from config.py):
  CIViC A=5, B=4, C=3, D=2, E=1
  Phase III=3, II=2, I=1
  PubMed hit = 0.1
  Direct target = 3.0
  Pathway-connected = 2.0

Example:
    agent = RankingAgent()
    ranked = agent.run(all_evidence, target_genes, pathway_genes, trials, kg_candidates)
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from config import (
    CIVIC_EVIDENCE_WEIGHTS,
    TRIAL_PHASE_WEIGHTS,
    PUBMED_HIT_WEIGHT,
    DIRECT_TARGET_WEIGHT,
    PATHWAY_CONNECTED_WEIGHT,
    PREDICTIVE_THRESHOLD,
    PROGNOSTIC_THRESHOLD,
    DEFAULT_TOP_N,
    MIN_DB_SOURCES,
    MIN_PUB_SOURCES,
)
from models.schemas import (
    NormalizedEvidence,
    CandidateBiomarker,
    BiomarkerCategory,
    BiomarkerType,
    ResponseDirection,
    EvidenceType,
    BiomarkerHypothesis,
    RankingRationale,
    SupportingEvidence,
)

logger = logging.getLogger(__name__)

# Heuristics for biomarker category classification
MUTATION_VARIANTS = re.compile(
    r"\b([A-Z]\d+[A-Z]|del|ins|fs|splice|stop|trunc|amp|G12C|V600E|exon\s*\d+)\b",
    re.IGNORECASE,
)
EXPRESSION_KEYWORDS = ["expression", "overexpression", "high", "low", "RNA", "mRNA", "IHC"]
FUSION_KEYWORDS = ["fusion", "rearrangement", "translocation", "EML4-ALK"]
SIGNATURE_KEYWORDS = ["TMB", "MSI", "HRD", "score", "signature", "burden", "instability"]
IMMUNE_KEYWORDS = ["PD-L1", "CD8", "TIL", "immune", "checkpoint", "CPS", "TPS", "foxp3"]
CNV_KEYWORDS = ["amplification", "deletion", "copy number", "gain", "loss", "CNV", "CNA"]


def _classify_biomarker_category(gene: str, variant: str | None) -> BiomarkerCategory:
    """Infer biomarker category from gene name and variant description."""
    label = f"{gene} {variant or ''}".upper()

    if any(kw.upper() in label for kw in IMMUNE_KEYWORDS + ["TMB", "MSI"]):
        return BiomarkerCategory.IMMUNE_SIGNATURE
    if any(kw.upper() in label for kw in SIGNATURE_KEYWORDS):
        return BiomarkerCategory.PATHWAY_SIGNATURE
    if any(kw.upper() in label for kw in FUSION_KEYWORDS):
        return BiomarkerCategory.FUSION
    if any(kw.upper() in label for kw in CNV_KEYWORDS):
        return BiomarkerCategory.COPY_NUMBER
    if any(kw.upper() in label for kw in EXPRESSION_KEYWORDS):
        return BiomarkerCategory.EXPRESSION
    if variant and MUTATION_VARIANTS.search(variant):
        return BiomarkerCategory.MUTATION
    if variant:
        return BiomarkerCategory.MUTATION
    return BiomarkerCategory.OTHER


def _classify_response_direction(evidence_items: list[NormalizedEvidence]) -> ResponseDirection:
    """Infer response direction from evidence text."""
    support_count = 0
    resist_count = 0
    for ev in evidence_items:
        claim_lower = ev.claim.lower()
        if any(w in claim_lower for w in ["resistance", "resistant", "lack of response", "does not"]):
            resist_count += 1
        elif any(w in claim_lower for w in ["response", "sensitive", "benefit", "supports"]):
            support_count += 1

    if resist_count > support_count:
        return ResponseDirection.RESISTANCE
    if support_count > 0:
        return ResponseDirection.POSITIVE
    return ResponseDirection.UNKNOWN


def _get_civic_level_order(level: str) -> int:
    """Return sort order for CIViC levels (A=0 is best)."""
    return {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}.get(level, 5)


class RankingAgent:
    """
    Deterministic evidence scoring and biomarker ranking engine.
    Operates entirely on normalized evidence — no LLM required.
    """

    def run(
        self,
        all_evidence: list[NormalizedEvidence],
        target_genes: list[str],
        pathway_genes: list[str],
        kg_candidates: list[str],
        top_n: int = DEFAULT_TOP_N,
    ) -> list[CandidateBiomarker]:
        """
        Score and rank candidate biomarkers from all evidence.

        Args:
            all_evidence: Normalized evidence from all agents.
            target_genes: Direct drug target genes.
            pathway_genes: Pathway-connected candidate genes.
            kg_candidates: Genes from knowledge graph expansion.
            top_n: Number of top biomarkers to return.

        Returns:
            Ranked list of CandidateBiomarker objects (normalized scores 0-100).
        """
        logger.info(
            "[RankingAgent] Scoring %d evidence items across %d target + %d pathway + %d KG genes",
            len(all_evidence),
            len(target_genes),
            len(pathway_genes),
            len(kg_candidates),
        )

        # Map gene → CandidateBiomarker accumulators
        candidates: dict[str, CandidateBiomarker] = {}
        target_set = {g.upper() for g in target_genes}
        pathway_set = {g.upper() for g in pathway_genes}

        def get_or_create(gene: str, variant: str | None = None) -> CandidateBiomarker:
            key = f"{gene}:{variant}" if variant else gene
            if key not in candidates:
                label = f"{gene} {variant}".strip() if variant else gene
                candidates[key] = CandidateBiomarker(
                    gene=gene,
                    variant=variant,
                    biomarker_label=label,
                    category=_classify_biomarker_category(gene, variant),
                    is_direct_target=gene.upper() in target_set,
                    is_pathway_connected=gene.upper() in pathway_set,
                )
                if gene.upper() in target_set:
                    candidates[key].target_score += DIRECT_TARGET_WEIGHT
                if gene.upper() in pathway_set:
                    candidates[key].pathway_score += PATHWAY_CONNECTED_WEIGHT
            return candidates[key]

        # Score each evidence item
        for ev in all_evidence:
            gene = ev.gene
            if not gene:
                continue

            cand = get_or_create(gene, ev.variant if ev.variant else None)
            cand.evidence_items.append(ev)

            if ev.source == "CIViC":
                level = ev.evidence_level or "E"
                score = CIVIC_EVIDENCE_WEIGHTS.get(level, 0.5)
                cand.civic_score += score
                cand.civic_ids.append(ev.source_id)

                if ev.evidence_type == EvidenceType.PREDICTIVE:
                    cand.predictive_raw += score
                elif ev.evidence_type == EvidenceType.PROGNOSTIC:
                    cand.prognostic_raw += score

                # Track best CIViC level
                if cand.best_civic_level is None or (
                    _get_civic_level_order(level) < _get_civic_level_order(cand.best_civic_level)
                ):
                    cand.best_civic_level = level

            elif ev.source == "ClinicalTrials":
                phase_score = 0.0
                claim_upper = ev.claim.upper()
                for phase_key, ps in TRIAL_PHASE_WEIGHTS.items():
                    if phase_key.upper() in claim_upper:
                        phase_score = max(phase_score, ps)
                        break
                else:
                    phase_score = 1.0
                cand.trial_score += phase_score
                cand.trial_ids.append(ev.source_id)
                cand.predictive_raw += phase_score * 0.5

            elif ev.source == "PubMed":
                cand.pubmed_score += PUBMED_HIT_WEIGHT
                pmid = ev.source_id.replace("PMID:", "")
                if pmid not in cand.pubmed_ids:
                    cand.pubmed_ids.append(pmid)
                if ev.evidence_type == EvidenceType.PREDICTIVE:
                    cand.predictive_raw += PUBMED_HIT_WEIGHT
                elif ev.evidence_type == EvidenceType.PROGNOSTIC:
                    cand.prognostic_raw += PUBMED_HIT_WEIGHT
                else:
                    cand.predictive_raw += PUBMED_HIT_WEIGHT * 0.5
                    cand.prognostic_raw += PUBMED_HIT_WEIGHT * 0.5

            else:
                # Other sources: small additive weight
                cand.pubmed_score += ev.strength * 0.1

        # Also register KG-expanded candidates that may not have evidence yet
        for gene in kg_candidates:
            get_or_create(gene)

        # Normalize scores to 0-100
        all_candidates = list(candidates.values())
        if not all_candidates:
            return []

        max_raw = max(c.total_raw_score for c in all_candidates) or 1.0
        max_pred = max(c.predictive_raw for c in all_candidates) or 1.0
        max_prog = max(c.prognostic_raw for c in all_candidates) or 1.0

        for cand in all_candidates:
            cand.civic_score = min(100.0, (cand.total_raw_score / max_raw) * 100.0)
            cand.predictive_raw = min(100.0, (cand.predictive_raw / max_pred) * 100.0)
            cand.prognostic_raw = min(100.0, (cand.prognostic_raw / max_prog) * 100.0)

        # Sort by total raw score descending
        all_candidates.sort(key=lambda c: c.total_raw_score, reverse=True)

        # Grounding gate: candidates need independent structured and publication
        # evidence before they can become biomarker hypotheses.
        valid = [
            c for c in all_candidates
            if c.has_minimum_evidence
        ]

        logger.info(
            "[RankingAgent] %d valid candidates after minimum evidence gate "
            "(%d DB source(s), %d publication(s); top gene: %s, score=%.1f)",
            len(valid),
            MIN_DB_SOURCES,
            MIN_PUB_SOURCES,
            valid[0].biomarker_label if valid else "N/A",
            valid[0].total_raw_score if valid else 0,
        )

        return valid[:top_n]

    def to_hypotheses(
        self,
        ranked_candidates: list[CandidateBiomarker],
    ) -> list[BiomarkerHypothesis]:
        """
        Convert ranked CandidateBiomarker objects to BiomarkerHypothesis schema.
        Assigns biomarker_type based on predictive/prognostic score thresholds.
        """
        hypotheses: list[BiomarkerHypothesis] = []

        for rank, cand in enumerate(ranked_candidates, start=1):
            pred_score = cand.predictive_raw
            prog_score = cand.prognostic_raw

            if pred_score >= PREDICTIVE_THRESHOLD and prog_score >= PROGNOSTIC_THRESHOLD:
                bm_type = BiomarkerType.BOTH
            elif pred_score >= PREDICTIVE_THRESHOLD:
                bm_type = BiomarkerType.PREDICTIVE
            elif prog_score >= PROGNOSTIC_THRESHOLD:
                bm_type = BiomarkerType.PROGNOSTIC
            else:
                bm_type = BiomarkerType.UNKNOWN

            direction = _classify_response_direction(cand.evidence_items)

            # Collect supporting sources
            sources: list[str] = []
            for ev in cand.evidence_items[:5]:
                sources.append(ev.source_id)

            supporting_evidence = [
                SupportingEvidence(
                    source=ev.source,
                    id=ev.source_id,
                    claim=ev.claim[:200],
                )
                for ev in cand.evidence_items[:5]
            ]

            # Drug relevance string
            if cand.is_direct_target:
                relevance = "Direct molecular target"
            elif cand.is_pathway_connected:
                relevance = "Pathway-connected biomarker"
            else:
                relevance = "Evidence-supported candidate"

            rationale = RankingRationale(
                direct_target=cand.is_direct_target,
                civic_level=cand.best_civic_level,
                civic_evidence_count=len(cand.civic_ids),
                pubmed_hits=len(cand.pubmed_ids),
                clinical_trials=len(cand.trial_ids),
                pathway_support=cand.is_pathway_connected,
                raw_score=cand.total_raw_score,
            )

            hypotheses.append(
                BiomarkerHypothesis(
                    rank=rank,
                    biomarker=cand.biomarker_label,
                    biomarker_category=cand.category,
                    biomarker_type=bm_type,
                    direction=direction,
                    confidence_score=cand.civic_score,
                    predictive_score=pred_score,
                    prognostic_score=prog_score,
                    evidence_level=cand.best_civic_level,
                    drug_relevance=relevance,
                    supporting_sources=sources,
                    supporting_evidence=supporting_evidence,
                    ranking_rationale=rationale,
                    hypothesis="",  # To be filled by LLM synthesis
                )
            )

        return hypotheses
