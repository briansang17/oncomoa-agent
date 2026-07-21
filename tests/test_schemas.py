"""
OncoMOA — Unit Tests: Pydantic Schema Validation

Tests:
  - Score clamping to [0, 100]
  - Required fields enforcement
  - Enum validation
  - AgentOutput structure
  - CandidateBiomarker evidence gate

Run:
    pytest tests/test_schemas.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pydantic import ValidationError

from models.schemas import (
    BiomarkerHypothesis,
    BiomarkerCategory,
    BiomarkerType,
    ResponseDirection,
    NormalizedEvidence,
    EvidenceType,
    EvidenceDirection,
    AgentOutput,
    KnowledgeGraphSummary,
    CandidateBiomarker,
    RankingRationale,
    SupportingEvidence,
)


class TestBiomarkerHypothesisSchema:
    """BiomarkerHypothesis Pydantic model validation."""

    def test_valid_hypothesis(self):
        """Valid hypothesis should parse without errors."""
        h = BiomarkerHypothesis(
            rank=1,
            biomarker="KRAS G12C",
            biomarker_category=BiomarkerCategory.MUTATION,
            biomarker_type=BiomarkerType.PREDICTIVE,
            direction=ResponseDirection.POSITIVE,
            confidence_score=95.0,
            predictive_score=91.0,
            prognostic_score=12.0,
            evidence_level="A",
            drug_relevance="Direct molecular target",
            supporting_sources=["CIViC_12", "PMID:33836569"],
            supporting_evidence=[
                SupportingEvidence(source="CIViC", id="12", claim="KRAS G12C predicts response")
            ],
            ranking_rationale=RankingRationale(direct_target=True, civic_level="A"),
            hypothesis="KRAS G12C is the direct molecular target of sotorasib.",
        )
        assert h.rank == 1
        assert h.confidence_score == 95.0

    def test_score_clamping_over_100(self):
        """Scores over 100 should be clamped to 100."""
        h = BiomarkerHypothesis(
            rank=1,
            biomarker="KRAS",
            confidence_score=150.0,
            predictive_score=200.0,
            prognostic_score=-10.0,
        )
        assert h.confidence_score == 100.0
        assert h.predictive_score == 100.0
        assert h.prognostic_score == 0.0

    def test_default_values(self):
        """Default values should be correctly set."""
        h = BiomarkerHypothesis(rank=1, biomarker="BRCA1", confidence_score=50.0,
                                predictive_score=50.0, prognostic_score=20.0)
        assert h.biomarker_category == BiomarkerCategory.OTHER
        assert h.biomarker_type == BiomarkerType.UNKNOWN
        assert h.direction == ResponseDirection.UNKNOWN
        assert h.supporting_sources == []

    def test_enum_validation(self):
        """Invalid enum values should raise ValidationError."""
        with pytest.raises(ValidationError):
            BiomarkerHypothesis(
                rank=1,
                biomarker="KRAS",
                confidence_score=50.0,
                predictive_score=50.0,
                prognostic_score=20.0,
                biomarker_category="invalid_category",
            )


class TestNormalizedEvidenceSchema:
    """NormalizedEvidence schema validation."""

    def test_valid_evidence(self):
        """Valid evidence should parse correctly."""
        ev = NormalizedEvidence(
            source="CIViC",
            source_id="CIViC_12",
            gene="KRAS",
            variant="G12C",
            evidence_type=EvidenceType.PREDICTIVE,
            evidence_direction=EvidenceDirection.SUPPORTS,
            evidence_level="A",
            claim="KRAS G12C is predictive of sotorasib response",
            strength=5.0,
        )
        assert ev.source == "CIViC"
        assert ev.strength == 5.0

    def test_default_evidence_type(self):
        """Default evidence type should be UNKNOWN."""
        ev = NormalizedEvidence(source="Test", source_id="T1", claim="test", strength=1.0)
        assert ev.evidence_type == EvidenceType.UNKNOWN

    def test_raw_data_defaults_empty(self):
        """raw_data should default to empty dict."""
        ev = NormalizedEvidence(source="X", source_id="Y", claim="Z", strength=0.1)
        assert ev.raw_data == {}


class TestAgentOutputSchema:
    """AgentOutput top-level schema validation."""

    def test_valid_output(self):
        """Valid AgentOutput should parse correctly."""
        output = AgentOutput(
            drug_name="sotorasib",
            moa_description="KRAS G12C inhibitor",
            target_genes=["KRAS"],
            llm_backend_used="Ollama/meditron",
            knowledge_graph_summary=KnowledgeGraphSummary(node_count=50, edge_count=80),
            hypotheses=[],
            total_evidence_items=150,
        )
        assert output.drug_name == "sotorasib"
        assert output.total_evidence_items == 150

    def test_empty_hypotheses_allowed(self):
        """Empty hypothesis list should be valid."""
        output = AgentOutput(
            drug_name="test_drug",
            moa_description="test moa",
        )
        assert output.hypotheses == []

    def test_failed_sources_tracking(self):
        """Failed sources list should be tracked."""
        output = AgentOutput(
            drug_name="test",
            moa_description="test",
            failed_sources=["CIViC", "PubMed"],
        )
        assert "CIViC" in output.failed_sources


class TestCandidateBiomarkerGate:
    """Evidence grounding gate validation."""

    def test_candidate_with_db_and_pub_passes_gate(self):
        """Candidate with both DB source and publication should pass gate."""
        cand = CandidateBiomarker(gene="KRAS")
        cand.evidence_items = [
            NormalizedEvidence(
                source="CIViC", source_id="CIViC_1", gene="KRAS",
                claim="CIViC evidence", strength=5.0
            )
        ]
        cand.pubmed_ids = ["33836569"]
        assert cand.has_minimum_evidence is True

    def test_candidate_with_no_pub_fails_gate(self):
        """Candidate without any publication should fail gate."""
        cand = CandidateBiomarker(gene="KRAS")
        cand.evidence_items = [
            NormalizedEvidence(
                source="CIViC", source_id="CIViC_1", gene="KRAS",
                claim="CIViC evidence", strength=5.0
            )
        ]
        cand.pubmed_ids = []
        assert cand.has_minimum_evidence is False

    def test_candidate_with_no_db_fails_gate(self):
        """Candidate without any DB source should fail gate."""
        cand = CandidateBiomarker(gene="KRAS")
        cand.evidence_items = [
            NormalizedEvidence(
                source="PubMed", source_id="PMID:111", gene="KRAS",
                claim="PubMed only", strength=0.1
            )
        ]
        cand.pubmed_ids = ["111"]
        assert cand.has_minimum_evidence is False
