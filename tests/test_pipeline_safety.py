"""Regression tests for fail-closed evidence and LLM-synthesis behavior."""

import asyncio
import json
import os
import sys
from contextlib import ExitStack
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.biomarker_agent import BiomarkerSynthesisAgent
from agents.ranking_agent import RankingAgent
from llm.backend import get_backend
from models.schemas import (
    BiomarkerCategory,
    BiomarkerHypothesis,
    BiomarkerType,
    DrugInfo,
    EvidenceDirection,
    EvidenceType,
    KnowledgeGraphSummary,
    NormalizedEvidence,
    RankingRationale,
    ResponseDirection,
    SupportingEvidence,
)
from tools.cache import _is_cacheable_api_result


def make_hypothesis() -> BiomarkerHypothesis:
    """Build one deterministic, evidence-backed hypothesis for validation tests."""
    return BiomarkerHypothesis(
        rank=1,
        biomarker="KRAS G12C",
        biomarker_category=BiomarkerCategory.MUTATION,
        biomarker_type=BiomarkerType.PREDICTIVE,
        direction=ResponseDirection.POSITIVE,
        confidence_score=80,
        predictive_score=80,
        prognostic_score=0,
        supporting_sources=["CIViC:1", "PMID:1"],
        supporting_evidence=[
            SupportingEvidence(source="CIViC", id="CIViC:1", claim="Supported"),
            SupportingEvidence(source="PubMed", id="PMID:1", claim="Published"),
        ],
        ranking_rationale=RankingRationale(raw_score=5),
    )


def test_none_backend_is_not_auto_routed():
    """Ensure the disabled backend choice cannot silently select an LLM."""
    try:
        get_backend(override="none")
    except ValueError as exc:
        assert "disabled" in str(exc)
    else:
        raise AssertionError("The disabled backend must raise ValueError.")


def test_ranking_rejects_db_only_candidates():
    """Require both structured and publication evidence before ranking a candidate."""
    evidence = [
        NormalizedEvidence(
            source="CIViC",
            source_id="CIViC:1",
            gene="KRAS",
            evidence_type=EvidenceType.PREDICTIVE,
            evidence_direction=EvidenceDirection.SUPPORTS,
            claim="KRAS predicts response",
            strength=5,
        )
    ]

    ranked = RankingAgent().run(
        all_evidence=evidence,
        target_genes=["KRAS"],
        pathway_genes=[],
        kg_candidates=[],
    )

    assert ranked == []


def test_synthesis_rejects_llm_only_biomarkers():
    """Reject a valid-looking model response for a non-deterministic candidate."""
    agent = BiomarkerSynthesisAgent(object())
    fabricated = make_hypothesis().model_dump()
    fabricated["biomarker"] = "VHL"
    fabricated["supporting_evidence"] = [
        {"source": "PubMed", "id": "PMID:invented", "claim": "Invented"}
    ]

    assert agent._parse_and_validate(json.dumps([fabricated]), [make_hypothesis()]) == []


def test_synthesis_skips_empty_deterministic_ranking():
    """Avoid contacting a backend when no grounded hypotheses are available."""
    result = asyncio.run(
        BiomarkerSynthesisAgent(object()).run(
            drug_name="belzutifan",
            moa_description="HIF-2alpha inhibitor",
            pre_ranked_hypotheses=[],
            all_evidence=[],
            target_genes=[],
        )
    )

    assert result == []


def test_empty_api_results_are_not_cacheable():
    """Prevent ambiguous transport-error empties from poisoning the cache."""
    assert _is_cacheable_api_result([]) is False
    assert _is_cacheable_api_result({}) is False
    assert _is_cacheable_api_result([{"id": "record"}]) is True


def test_literature_preserves_pubmed_query_gene_without_pubtator_annotations():
    """Retain PubMed query attribution when PubTator cannot annotate a record."""
    from agents.literature_agent import LiteratureAgent
    from models.schemas import PubMedArticle

    evidence = LiteratureAgent()._articles_to_evidence(
        articles=[PubMedArticle(pmid="123", title="Pembrolizumab biomarker study")],
        article_query_genes={"123": ["PDCD1"]},
        drug_name="pembrolizumab",
    )

    assert len(evidence) == 1
    assert evidence[0].gene == "PDCD1"
    assert evidence[0].source_id == "PMID:123"


def test_orchestrator_fails_closed_and_skips_llm():
    """Return no hypotheses and clear metadata when every retrieval stage is empty."""
    import agents.orchestrator as module

    class EmptyDrugAgent:
        """Return no resolved targets or source records."""

        async def run(self, drug_name, moa_description):
            return DrugInfo(drug_name=drug_name)

    class EmptyTargetBiologyAgent:
        """Return no biological enrichment."""

        async def run(self, target_genes):
            return [], []

    class EmptyPathwayAgent:
        """Return no pathway expansion."""

        async def run(self, target_genes, gene_infos):
            return [], []

    class EmptyClinicalEvidenceAgent:
        """Return no structured clinical evidence."""

        async def run(self, candidate_genes, drug_name):
            return [], {"sources": []}

    class EmptyLiteratureAgent:
        """Return no literature evidence."""

        async def run(self, candidate_genes, drug_name):
            return [], []

    class EmptyTrialAgent:
        """Return no trial evidence."""

        async def run(self, drug_name, candidate_genes):
            return [], []

    class EmptyGraphAgent:
        """Return an empty graph summary without writing output files."""

        async def run(self, drug_info, pathways, evidence, trials, articles):
            return None, KnowledgeGraphSummary()

    def backend_must_not_be_called(*args, **kwargs):
        """Fail the test if an LLM backend is initialized."""
        raise AssertionError("LLM backend should not be initialized")

    with ExitStack() as patches:
        patches.enter_context(patch.object(module, "DrugAgent", EmptyDrugAgent))
        patches.enter_context(patch.object(module, "TargetBiologyAgent", EmptyTargetBiologyAgent))
        patches.enter_context(patch.object(module, "PathwayAgent", EmptyPathwayAgent))
        patches.enter_context(patch.object(module, "ClinicalEvidenceAgent", EmptyClinicalEvidenceAgent))
        patches.enter_context(patch.object(module, "LiteratureAgent", EmptyLiteratureAgent))
        patches.enter_context(patch.object(module, "TrialAgent", EmptyTrialAgent))
        patches.enter_context(patch.object(module, "GraphAgent", EmptyGraphAgent))
        patches.enter_context(patch.object(module, "get_backend", backend_must_not_be_called))
        output = asyncio.run(
            module.OncologyOrchestrator().run(
                drug_name="belzutifan",
                moa_description="HIF-2alpha inhibitor",
            )
        )

    assert output.hypotheses == []
    assert output.llm_backend_used == "none"
    assert output.run_metadata["insufficient_evidence"] is True
    assert output.run_metadata["llm_synthesis_skipped_reason"] == "no_grounded_candidates"
    assert "DrugTargetResolution" in output.failed_sources
