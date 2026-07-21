"""
OncoMOA — Unit Tests: Evidence Scoring & Ranking Engine

Tests:
  - CIViC weight computation
  - Trial phase scoring
  - Predictive vs prognostic classification
  - Score normalization to 0-100
  - Biomarker category classification

Run:
    pytest tests/test_scoring.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models.schemas import (
    NormalizedEvidence,
    EvidenceType,
    EvidenceDirection,
    BiomarkerType,
    BiomarkerCategory,
)
from agents.ranking_agent import RankingAgent, _classify_biomarker_category, _classify_response_direction
from config import CIVIC_EVIDENCE_WEIGHTS, TRIAL_PHASE_WEIGHTS, PREDICTIVE_THRESHOLD


def make_civic_evidence(gene: str, level: str, ev_type: EvidenceType = EvidenceType.PREDICTIVE) -> NormalizedEvidence:
    """Helper: create a CIViC evidence item."""
    return NormalizedEvidence(
        source="CIViC",
        source_id=f"CIViC_{gene}_{level}",
        gene=gene,
        evidence_type=ev_type,
        evidence_direction=EvidenceDirection.SUPPORTS,
        evidence_level=level,
        claim=f"{gene} predicts response (Level {level})",
        strength=CIVIC_EVIDENCE_WEIGHTS.get(level, 1.0),
    )


def make_trial_evidence(gene: str, phase: str) -> NormalizedEvidence:
    """Helper: create a trial evidence item."""
    return NormalizedEvidence(
        source="ClinicalTrials",
        source_id=f"NCT{gene}",
        gene=gene,
        evidence_type=EvidenceType.PREDICTIVE,
        evidence_direction=EvidenceDirection.SUPPORTS,
        claim=f"Phase {phase} trial for drug in {gene} cancer (Phase {phase})",
        strength=TRIAL_PHASE_WEIGHTS.get(f"Phase {phase}", 1.0),
    )


def make_pubmed_evidence(gene: str, pmid: str) -> NormalizedEvidence:
    """Helper: create a PubMed evidence item."""
    return NormalizedEvidence(
        source="PubMed",
        source_id=f"PMID:{pmid}",
        gene=gene,
        evidence_type=EvidenceType.PREDICTIVE,
        evidence_direction=EvidenceDirection.SUPPORTS,
        claim=f"{gene} biomarker study",
        strength=0.1,
    )


# ─── Scoring Weight Tests ─────────────────────────────────────────────────────

class TestCivicWeights:
    """CIViC evidence level weights applied correctly."""

    def test_level_a_highest_weight(self):
        """Level A should score 5.0."""
        assert CIVIC_EVIDENCE_WEIGHTS["A"] == 5.0

    def test_level_e_lowest_weight(self):
        """Level E should score 1.0."""
        assert CIVIC_EVIDENCE_WEIGHTS["E"] == 1.0

    def test_weights_descending(self):
        """Weights should be strictly descending A > B > C > D > E."""
        levels = ["A", "B", "C", "D", "E"]
        weights = [CIVIC_EVIDENCE_WEIGHTS[l] for l in levels]
        assert weights == sorted(weights, reverse=True)


class TestTrialWeights:
    """Trial phase weights applied correctly."""

    def test_phase3_highest(self):
        """Phase 3 should score 3.0."""
        assert TRIAL_PHASE_WEIGHTS["Phase 3"] == 3.0

    def test_phase1_lowest(self):
        """Phase 1 should score 1.0."""
        assert TRIAL_PHASE_WEIGHTS["Phase 1"] == 1.0

    def test_phase_aliases(self):
        """Phase III alias should equal Phase 3."""
        assert TRIAL_PHASE_WEIGHTS["Phase III"] == TRIAL_PHASE_WEIGHTS["Phase 3"]


# ─── Biomarker Category Classification ───────────────────────────────────────

class TestBiomarkerClassification:
    """Biomarker category inference from gene + variant."""

    def test_kras_g12c_is_mutation(self):
        result = _classify_biomarker_category("KRAS", "G12C")
        assert result == BiomarkerCategory.MUTATION

    def test_tmb_is_immune_signature(self):
        result = _classify_biomarker_category("TMB", None)
        assert result == BiomarkerCategory.IMMUNE_SIGNATURE

    def test_pdl1_expression_is_immune(self):
        result = _classify_biomarker_category("PD-L1", "expression")
        assert result in (BiomarkerCategory.IMMUNE_SIGNATURE, BiomarkerCategory.EXPRESSION)

    def test_eml4_alk_is_fusion(self):
        result = _classify_biomarker_category("ALK", "EML4-ALK fusion")
        assert result == BiomarkerCategory.FUSION

    def test_unknown_gene_is_other(self):
        result = _classify_biomarker_category("XYZ123", None)
        assert result == BiomarkerCategory.OTHER


# ─── Response Direction Classification ────────────────────────────────────────

class TestResponseDirection:
    """Response direction inferred from evidence claims."""

    def test_positive_direction(self):
        evidence = [
            NormalizedEvidence(
                source="CIViC", source_id="x", gene="KRAS",
                evidence_type=EvidenceType.PREDICTIVE,
                evidence_direction=EvidenceDirection.SUPPORTS,
                claim="KRAS G12C predicts response to sotorasib",
                strength=5.0,
            )
        ]
        from models.schemas import ResponseDirection
        result = _classify_response_direction(evidence)
        assert result == ResponseDirection.POSITIVE

    def test_resistance_direction(self):
        evidence = [
            NormalizedEvidence(
                source="CIViC", source_id="y", gene="KRAS",
                evidence_type=EvidenceType.PREDICTIVE,
                evidence_direction=EvidenceDirection.DOES_NOT_SUPPORT,
                claim="TP53 mutation predicts resistance to treatment",
                strength=3.0,
            )
        ]
        from models.schemas import ResponseDirection
        result = _classify_response_direction(evidence)
        assert result == ResponseDirection.RESISTANCE


# ─── Ranking Engine Integration ───────────────────────────────────────────────

class TestRankingEngine:
    """Full scoring pipeline tests."""

    def test_direct_target_gets_bonus(self):
        """Direct target gene should score higher than pathway gene."""
        agent = RankingAgent()
        evidence = [
            make_civic_evidence("KRAS", "A"),
            make_pubmed_evidence("KRAS", "111111"),
            make_civic_evidence("STK11", "B"),
            make_pubmed_evidence("STK11", "222222"),
        ]
        results = agent.run(
            all_evidence=evidence,
            target_genes=["KRAS"],
            pathway_genes=["STK11"],
            kg_candidates=[],
            top_n=5,
        )
        gene_names = [r.gene for r in results]
        assert "KRAS" in gene_names
        kras_idx = gene_names.index("KRAS")
        # KRAS should be ranked first (direct target bonus)
        assert kras_idx == 0

    def test_score_normalized_to_100(self):
        """All confidence scores should be in [0, 100]."""
        agent = RankingAgent()
        evidence = [
            make_civic_evidence("BRCA1", "A"),
            make_trial_evidence("BRCA1", "3"),
            make_pubmed_evidence("BRCA1", "33333"),
            make_civic_evidence("PALB2", "B"),
        ]
        results = agent.run(
            all_evidence=evidence,
            target_genes=["BRCA1"],
            pathway_genes=["PALB2"],
            kg_candidates=[],
            top_n=5,
        )
        hypotheses = agent.to_hypotheses(results)
        for h in hypotheses:
            assert 0.0 <= h.confidence_score <= 100.0
            assert 0.0 <= h.predictive_score <= 100.0
            assert 0.0 <= h.prognostic_score <= 100.0

    def test_empty_evidence_returns_empty(self):
        """Empty evidence list should return empty results."""
        agent = RankingAgent()
        results = agent.run(
            all_evidence=[],
            target_genes=["KRAS"],
            pathway_genes=[],
            kg_candidates=[],
            top_n=5,
        )
        assert results == []

    def test_top_n_respected(self):
        """Should return at most top_n results."""
        agent = RankingAgent()
        evidence = [make_civic_evidence(f"GENE{i}", "C") for i in range(20)]
        results = agent.run(
            all_evidence=evidence,
            target_genes=["GENE0"],
            pathway_genes=[f"GENE{i}" for i in range(1, 20)],
            kg_candidates=[],
            top_n=5,
        )
        assert len(results) <= 5

    def test_predictive_prognostic_separation(self):
        """Predictive-only evidence should yield predictive type."""
        agent = RankingAgent()
        evidence = [
            make_civic_evidence("KRAS", "A", EvidenceType.PREDICTIVE),
            make_civic_evidence("KRAS", "B", EvidenceType.PREDICTIVE),
            make_pubmed_evidence("KRAS", "444444"),
        ]
        results = agent.run(
            all_evidence=evidence,
            target_genes=["KRAS"],
            pathway_genes=[],
            kg_candidates=[],
            top_n=3,
        )
        hypotheses = agent.to_hypotheses(results)
        assert any(h.biomarker_type == BiomarkerType.PREDICTIVE for h in hypotheses)
