"""
OncoMOA — Master Orchestrator Agent
Runs the full 9-agent pipeline in the correct order with progress tracking.
Handles errors gracefully: any single agent failure is logged and skipped.

Example:
    orchestrator = OncologyOrchestrator()
    result = await orchestrator.run("sotorasib", "Covalent KRAS G12C inhibitor", top_n=10)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from models.schemas import AgentOutput, BiomarkerHypothesis, KnowledgeGraphSummary
from agents.drug_agent import DrugAgent
from agents.target_biology_agent import TargetBiologyAgent
from agents.pathway_agent import PathwayAgent
from agents.clinical_evidence_agent import ClinicalEvidenceAgent
from agents.literature_agent import LiteratureAgent
from agents.trial_agent import TrialAgent
from agents.graph_agent import GraphAgent
from agents.ranking_agent import RankingAgent
from agents.biomarker_agent import BiomarkerSynthesisAgent
from llm.backend import get_backend
from config import DEFAULT_TOP_N

logger = logging.getLogger(__name__)


class OncologyOrchestrator:
    """
    Orchestrates the full OncoMOA pipeline from drug input to ranked biomarker output.

    Pipeline order:
      1. DrugAgent         — resolve drug targets
      2. TargetBiologyAgent — enrich gene biology
      3. PathwayAgent      — expand candidates via pathways
      4. ClinicalEvidenceAgent + LiteratureAgent + TrialAgent (parallel)
      5. GraphAgent        — build knowledge graph
      6. RankingAgent      — deterministic scoring
      7. BiomarkerSynthesisAgent — LLM narrative
    """

    def __init__(self, backend_override: str = "") -> None:
        self.backend_override = backend_override

    async def run(
        self,
        drug_name: str,
        moa_description: str,
        top_n: int = DEFAULT_TOP_N,
        progress: Progress | None = None,
    ) -> AgentOutput:
        """
        Execute the full oncology biomarker discovery pipeline.

        Args:
            drug_name: Drug name (e.g., "sotorasib").
            moa_description: Mechanism of action description.
            top_n: Number of biomarker hypotheses to return.
            progress: Optional Rich Progress instance for CLI progress bars.

        Returns:
            AgentOutput with ranked hypotheses, KG summary, and metadata.
        """
        start_time = time.time()
        failed_sources: list[str] = []
        successful_sources: list[str] = []

        def _add_task(description: str) -> Any:
            if progress:
                return progress.add_task(f"[cyan]{description}", total=None)
            return None

        def _complete_task(task_id: Any) -> None:
            if progress and task_id is not None:
                progress.update(task_id, completed=True)

        logger.info("=" * 60)
        logger.info("OncoMOA Pipeline START: drug=%s", drug_name)
        logger.info("=" * 60)

        # ── Step 1: Drug Agent ────────────────────────────────────────────────
        task = _add_task("[1/7] Resolving drug targets...")
        drug_info = None
        try:
            drug_agent = DrugAgent()
            drug_info = await drug_agent.run(drug_name, moa_description)
            if drug_info.sources:
                successful_sources.extend(drug_info.sources)
            else:
                failed_sources.append("DrugTargetResolution")
        except Exception as exc:
            logger.error("[Orchestrator] DrugAgent failed: %s", exc)
            failed_sources.append("DrugAgent")
        _complete_task(task)

        target_genes = drug_info.target_genes if drug_info else []
        if not target_genes:
            logger.warning(
                "[Orchestrator] No drug targets resolved. Continuing without "
                "MOA-derived gene guesses because they are not retrieved evidence."
            )

        # ── Step 2: Target Biology Agent ──────────────────────────────────────
        task = _add_task("[2/7] Enriching target biology...")
        gene_infos = []
        biology_evidence = []
        try:
            bio_agent = TargetBiologyAgent()
            gene_infos, biology_evidence = await bio_agent.run(target_genes)
            if gene_infos:
                successful_sources.append("TargetBiology")
            elif target_genes:
                failed_sources.append("TargetBiology")
        except Exception as exc:
            logger.error("[Orchestrator] TargetBiologyAgent failed: %s", exc)
            failed_sources.append("TargetBiology")
        _complete_task(task)

        # ── Step 3: Pathway Agent ─────────────────────────────────────────────
        task = _add_task("[3/7] Expanding via pathways...")
        pathway_genes = []
        pathways = []
        try:
            pathway_agent = PathwayAgent()
            pathway_genes, pathways = await pathway_agent.run(target_genes, gene_infos)
            if pathway_genes:
                successful_sources.append("PathwayDatabases")
        except Exception as exc:
            logger.error("[Orchestrator] PathwayAgent failed: %s", exc)
            failed_sources.append("PathwayDatabases")
        _complete_task(task)

        # All candidate genes = targets + pathway genes
        all_candidate_genes = list(dict.fromkeys(target_genes + pathway_genes))

        # ── Steps 4a, 4b, 4c: Parallel Evidence Collection ───────────────────
        task = _add_task("[4/7] Gathering evidence (CIViC, PubMed, Trials)...")
        clinical_evidence = []
        lit_evidence = []
        trial_evidence = []
        trials = []
        articles = []
        clinical_stats: dict[str, Any] = {}

        try:
            clinical_agent = ClinicalEvidenceAgent()
            lit_agent = LiteratureAgent()
            trial_agent = TrialAgent()

            (
                (clinical_evidence, clinical_stats),
                (lit_evidence, articles),
                (trial_evidence, trials),
            ) = await asyncio.gather(
                clinical_agent.run(all_candidate_genes, drug_name=drug_name),
                lit_agent.run(all_candidate_genes[:12], drug_name=drug_name),
                trial_agent.run(drug_name, all_candidate_genes),
                return_exceptions=False,
            )

            if clinical_evidence:
                successful_sources.extend(clinical_stats.get("sources", []))
            elif all_candidate_genes:
                failed_sources.append("ClinicalEvidence")
            if lit_evidence:
                successful_sources.append("PubMed/PubTator")
            elif all_candidate_genes:
                failed_sources.append("Literature")
            if trial_evidence:
                successful_sources.append("ClinicalTrials")
            else:
                failed_sources.append("ClinicalTrials")
        except Exception as exc:
            logger.error("[Orchestrator] Parallel evidence collection failed: %s", exc)
            failed_sources.append("EvidenceCollection")
        _complete_task(task)

        # Combine all evidence
        all_evidence = biology_evidence + clinical_evidence + lit_evidence + trial_evidence
        if not all_evidence:
            failed_sources.append("EvidenceRetrieval")
            logger.warning(
                "[Orchestrator] No evidence was retrieved; returning no "
                "biomarker hypotheses unless a later stage has grounded data."
            )

        # ── Step 5: Graph Agent ───────────────────────────────────────────────
        task = _add_task("[5/7] Building knowledge graph...")
        kg = None
        kg_summary = KnowledgeGraphSummary()
        try:
            graph_agent = GraphAgent()
            if drug_info:
                kg, kg_summary = await graph_agent.run(
                    drug_info, pathways, all_evidence, trials, articles
                )
                successful_sources.append("KnowledgeGraph")
        except Exception as exc:
            logger.error("[Orchestrator] GraphAgent failed: %s", exc)
            failed_sources.append("KnowledgeGraph")
        _complete_task(task)

        kg_candidates = kg_summary.candidate_biomarkers if kg_summary else []

        # ── Step 6: Ranking Agent ─────────────────────────────────────────────
        task = _add_task("[6/7] Ranking biomarkers...")
        pre_ranked_hypotheses: list[BiomarkerHypothesis] = []
        try:
            ranking_agent = RankingAgent()
            ranked_candidates = ranking_agent.run(
                all_evidence=all_evidence,
                target_genes=target_genes,
                pathway_genes=pathway_genes,
                kg_candidates=kg_candidates,
                top_n=top_n,
            )
            pre_ranked_hypotheses = ranking_agent.to_hypotheses(ranked_candidates)
            successful_sources.append("RankingEngine")
        except Exception as exc:
            logger.error("[Orchestrator] RankingAgent failed: %s", exc)
            failed_sources.append("RankingEngine")
        _complete_task(task)

        # ── Step 7: LLM Synthesis ─────────────────────────────────────────────
        task = _add_task("[7/7] LLM synthesis...")
        final_hypotheses = pre_ranked_hypotheses
        llm_backend_name = "none"
        synthesis_skipped_reason = ""
        if self.backend_override.lower() == "none":
            synthesis_skipped_reason = "disabled_by_user"
            logger.info("[Orchestrator] LLM synthesis skipped (--no-llm).")
        elif not pre_ranked_hypotheses:
            synthesis_skipped_reason = "no_grounded_candidates"
            logger.warning(
                "[Orchestrator] LLM synthesis skipped: deterministic ranking "
                "produced no evidence-grounded candidates."
            )
        else:
            try:
                backend = get_backend(drug_name=drug_name, override=self.backend_override)
                llm_backend_name = backend.name
                synthesis_agent = BiomarkerSynthesisAgent(backend)
                final_hypotheses = await synthesis_agent.run(
                    drug_name=drug_name,
                    moa_description=moa_description,
                    pre_ranked_hypotheses=pre_ranked_hypotheses,
                    all_evidence=all_evidence,
                    target_genes=target_genes,
                    top_n=top_n,
                )
                successful_sources.append(f"LLM:{llm_backend_name}")
            except Exception as exc:
                logger.error("[Orchestrator] LLM synthesis failed: %s", exc)
                failed_sources.append(f"LLM:{llm_backend_name}")
        _complete_task(task)

        elapsed = time.time() - start_time
        logger.info("OncoMOA Pipeline COMPLETE in %.1fs", elapsed)

        return AgentOutput(
            drug_name=drug_name,
            moa_description=moa_description,
            target_genes=target_genes,
            llm_backend_used=llm_backend_name,
            knowledge_graph_summary=kg_summary,
            hypotheses=final_hypotheses,
            failed_sources=list(set(failed_sources)),
            successful_sources=list(set(successful_sources)),
            total_evidence_items=len(all_evidence),
            run_metadata={
                "elapsed_seconds": round(elapsed, 2),
                "total_evidence": len(all_evidence),
                "target_genes": target_genes,
                "pathway_genes_count": len(pathway_genes),
                "trials_found": len(trials),
                "articles_found": len(articles),
                "civic_stats": clinical_stats,
                "insufficient_evidence": not pre_ranked_hypotheses,
                "llm_synthesis_skipped_reason": synthesis_skipped_reason or None,
            },
        )
