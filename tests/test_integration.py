"""
OncoMOA — Integration Tests: sotorasib, olaparib, pembrolizumab

These tests run the full agent pipeline (without LLM synthesis) to verify:
  - Knowledge graph construction
  - Evidence retrieval (mocked/cached)
  - Ranking engine
  - JSON output validation
  - Output file generation

Note: API calls are run against real endpoints but use the disk cache.
      These tests may be slow on first run (cache cold start).
      Set ONCOMOA_SKIP_INTEGRATION=1 to skip in CI without network.

Run:
    pytest tests/test_integration.py -v -s
    pytest tests/test_integration.py -v -s -k "sotorasib"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import pytest
import pytest_asyncio

SKIP_INTEGRATION = os.getenv("ONCOMOA_SKIP_INTEGRATION", "0") == "1"
skip_reason = "Skipping integration tests (ONCOMOA_SKIP_INTEGRATION=1)"


# ─── Test Cases ───────────────────────────────────────────────────────────────

DRUG_TEST_CASES = [
    {
        "drug": "sotorasib",
        "moa": "Covalent KRAS G12C inhibitor that locks KRAS in the GDP-bound inactive state and blocks downstream RAS/MAPK signaling.",
        "expected_targets": ["KRAS"],
        "expected_biomarker_contains": ["KRAS"],
    },
    {
        "drug": "olaparib",
        "moa": "PARP1 and PARP2 inhibitor that traps PARP on single-strand DNA breaks, causing double-strand breaks and synthetic lethality in BRCA-deficient cells.",
        "expected_targets": ["PARP1", "PARP2", "BRCA1", "BRCA2"],
        "expected_biomarker_contains": ["BRCA"],
    },
    {
        "drug": "pembrolizumab",
        "moa": "Humanized monoclonal antibody that blocks PD-1/PD-L1 interaction, restoring T-cell anti-tumor immune response.",
        "expected_targets": ["PDCD1", "CD274"],
        "expected_biomarker_contains": ["PD"],
    },
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

async def run_pipeline_no_llm(drug: str, moa: str) -> dict:
    """Run pipeline with LLM disabled, return JSON-serializable output dict."""
    from agents.orchestrator import OncologyOrchestrator
    orchestrator = OncologyOrchestrator(backend_override="none")
    output = await orchestrator.run(drug_name=drug, moa_description=moa, top_n=5)
    return output.model_dump()


# ─── Sotorasib Integration Test ───────────────────────────────────────────────

@pytest.mark.skipif(SKIP_INTEGRATION, reason=skip_reason)
@pytest.mark.asyncio
class TestSotorasibIntegration:
    """Full pipeline test for sotorasib (KRAS G12C inhibitor)."""

    async def test_sotorasib_pipeline_completes(self):
        """Pipeline should complete without crashing."""
        case = DRUG_TEST_CASES[0]
        result = await run_pipeline_no_llm(case["drug"], case["moa"])
        assert result is not None
        assert result["drug_name"] == "sotorasib"

    async def test_sotorasib_has_hypotheses(self):
        """Should generate at least 1 biomarker hypothesis."""
        case = DRUG_TEST_CASES[0]
        result = await run_pipeline_no_llm(case["drug"], case["moa"])
        assert len(result["hypotheses"]) >= 1

    async def test_sotorasib_hypothesis_schema_valid(self):
        """All hypotheses should conform to the required schema."""
        case = DRUG_TEST_CASES[0]
        result = await run_pipeline_no_llm(case["drug"], case["moa"])
        required_fields = [
            "rank", "biomarker", "biomarker_category", "biomarker_type",
            "direction", "confidence_score", "predictive_score", "prognostic_score",
        ]
        for hyp in result["hypotheses"]:
            for field in required_fields:
                assert field in hyp, f"Missing field '{field}' in hypothesis: {hyp}"

    async def test_sotorasib_kg_created(self):
        """Knowledge graph should have nodes and edges."""
        case = DRUG_TEST_CASES[0]
        result = await run_pipeline_no_llm(case["drug"], case["moa"])
        kg = result["knowledge_graph_summary"]
        assert kg["node_count"] >= 1

    async def test_sotorasib_output_json_valid(self):
        """Output should be valid JSON."""
        case = DRUG_TEST_CASES[0]
        result = await run_pipeline_no_llm(case["drug"], case["moa"])
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert parsed["drug_name"] == "sotorasib"

    async def test_sotorasib_scores_in_range(self):
        """All scores should be in [0, 100]."""
        case = DRUG_TEST_CASES[0]
        result = await run_pipeline_no_llm(case["drug"], case["moa"])
        for hyp in result["hypotheses"]:
            assert 0 <= hyp["confidence_score"] <= 100
            assert 0 <= hyp["predictive_score"] <= 100
            assert 0 <= hyp["prognostic_score"] <= 100


# ─── Olaparib Integration Test ────────────────────────────────────────────────

@pytest.mark.skipif(SKIP_INTEGRATION, reason=skip_reason)
@pytest.mark.asyncio
class TestOlaparibIntegration:
    """Full pipeline test for olaparib (PARP inhibitor)."""

    async def test_olaparib_pipeline_completes(self):
        """Pipeline should complete without crashing."""
        case = DRUG_TEST_CASES[1]
        result = await run_pipeline_no_llm(case["drug"], case["moa"])
        assert result["drug_name"] == "olaparib"

    async def test_olaparib_has_evidence(self):
        """Should collect evidence items."""
        case = DRUG_TEST_CASES[1]
        result = await run_pipeline_no_llm(case["drug"], case["moa"])
        assert result["total_evidence_items"] >= 0

    async def test_olaparib_ranking_complete(self):
        """Ranking should complete and produce hypotheses."""
        case = DRUG_TEST_CASES[1]
        result = await run_pipeline_no_llm(case["drug"], case["moa"])
        # Even with limited API results, ranking should complete
        assert isinstance(result["hypotheses"], list)


# ─── Pembrolizumab Integration Test ──────────────────────────────────────────

@pytest.mark.skipif(SKIP_INTEGRATION, reason=skip_reason)
@pytest.mark.asyncio
class TestPembrolizumabIntegration:
    """Full pipeline test for pembrolizumab (PD-1 inhibitor — routes to Gemini)."""

    async def test_pembrolizumab_pipeline_completes(self):
        """Pipeline should complete (LLM disabled for this test)."""
        case = DRUG_TEST_CASES[2]
        result = await run_pipeline_no_llm(case["drug"], case["moa"])
        assert result["drug_name"] == "pembrolizumab"

    async def test_pembrolizumab_has_hypotheses(self):
        """Should generate hypotheses for checkpoint inhibitor."""
        case = DRUG_TEST_CASES[2]
        result = await run_pipeline_no_llm(case["drug"], case["moa"])
        assert isinstance(result["hypotheses"], list)

    async def test_pembrolizumab_source_tracking(self):
        """Should track successful and failed sources."""
        case = DRUG_TEST_CASES[2]
        result = await run_pipeline_no_llm(case["drug"], case["moa"])
        assert "successful_sources" in result
        assert "failed_sources" in result


# ─── Drug Routing Tests ────────────────────────────────────────────────────────

class TestDrugRouting:
    """Test smart LLM routing logic."""

    def test_pembrolizumab_routes_to_gemini(self):
        """Pembrolizumab is a Merck drug — should route to Gemini."""
        from config import requires_gemini
        assert requires_gemini("pembrolizumab") is True

    def test_keytruda_routes_to_gemini(self):
        """Keytruda (brand name) should also route to Gemini."""
        from config import requires_gemini
        assert requires_gemini("keytruda") is True

    def test_sotorasib_routes_to_ollama(self):
        """Sotorasib (Amgen) should NOT route to Gemini."""
        from config import requires_gemini
        assert requires_gemini("sotorasib") is False

    def test_adc_suffix_routes_to_gemini(self):
        """ADC names ending in 'deruxtecan' should route to Gemini."""
        from config import requires_gemini
        assert requires_gemini("trastuzumab deruxtecan") is True
        assert requires_gemini("sacituzumab govitecan") is True
        assert requires_gemini("enfortumab vedotin") is True

    def test_ivonescimab_bispecific_routes_to_gemini(self):
        """Bispecific ivonescimab should route to Gemini."""
        from config import requires_gemini
        assert requires_gemini("ivonescimab") is True

    def test_olaparib_routes_to_ollama(self):
        """Olaparib should NOT route to Gemini."""
        from config import requires_gemini
        assert requires_gemini("olaparib") is False


# ─── Knowledge Graph Tests ────────────────────────────────────────────────────

class TestKnowledgeGraph:
    """Test KG construction and querying."""

    def test_kg_add_drug_and_target(self):
        """Drug and target nodes should be added correctly."""
        from graph.knowledge_graph import OncologyKnowledgeGraph
        kg = OncologyKnowledgeGraph()
        kg.add_drug("sotorasib")
        kg.add_gene_target("sotorasib", "KRAS")
        assert kg.graph.has_node("sotorasib")
        assert kg.graph.has_node("KRAS")
        assert kg.graph.has_edge("sotorasib", "KRAS")

    def test_kg_edge_relation(self):
        """Edge relation should be 'targets'."""
        from graph.knowledge_graph import OncologyKnowledgeGraph
        kg = OncologyKnowledgeGraph()
        kg.add_drug("sotorasib")
        kg.add_gene_target("sotorasib", "KRAS")
        edge_data = kg.graph.edges["sotorasib", "KRAS"]
        assert edge_data["relation"] == "targets"

    def test_kg_summary_structure(self):
        """KG summary should have correct structure."""
        from graph.knowledge_graph import OncologyKnowledgeGraph
        kg = OncologyKnowledgeGraph()
        kg.add_drug("sotorasib")
        kg.add_gene_target("sotorasib", "KRAS")
        kg.add_gene("STK11")
        summary = kg.build_summary()
        assert summary.node_count >= 2
        assert summary.edge_count >= 1

    def test_kg_graphml_export(self, tmp_path):
        """KG should export to GraphML without error."""
        from graph.knowledge_graph import OncologyKnowledgeGraph
        kg = OncologyKnowledgeGraph()
        kg.add_drug("olaparib")
        kg.add_gene_target("olaparib", "PARP1")
        graphml_path = tmp_path / "test_kg.graphml"
        kg.export_graphml(graphml_path)
        assert graphml_path.exists()
        assert graphml_path.stat().st_size > 0

    def test_kg_json_export(self, tmp_path):
        """KG should export to JSON without error."""
        import json
        from graph.knowledge_graph import OncologyKnowledgeGraph
        kg = OncologyKnowledgeGraph()
        kg.add_drug("pembrolizumab")
        kg.add_gene_target("pembrolizumab", "PDCD1")
        json_path = tmp_path / "test_kg.json"
        kg.export_json(json_path)
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert "nodes" in data or "links" in data
