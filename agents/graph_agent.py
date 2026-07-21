"""
OncoMOA Agent — Knowledge Graph Agent
Constructs the oncology knowledge graph from all collected evidence.
Performs neighborhood expansion to discover additional candidate biomarkers.
Exports to GraphML and JSON.

Example:
    agent = GraphAgent()
    kg, summary = await agent.run(drug_info, pathways, all_evidence, trials, articles)
"""

from __future__ import annotations

import logging

from models.schemas import (
    DrugInfo,
    PathwayInfo,
    NormalizedEvidence,
    ClinicalTrialInfo,
    PubMedArticle,
    KnowledgeGraphSummary,
)
from graph.knowledge_graph import OncologyKnowledgeGraph
from config import OUTPUT_KG_GRAPHML, OUTPUT_KG_JSON

logger = logging.getLogger(__name__)


class GraphAgent:
    """
    Builds and queries the OncoMOA knowledge graph.
    Integrates all evidence sources into a unified NetworkX DiGraph.
    """

    async def run(
        self,
        drug_info: DrugInfo,
        pathways: list[PathwayInfo],
        all_evidence: list[NormalizedEvidence],
        trials: list[ClinicalTrialInfo],
        articles: list[PubMedArticle],
    ) -> tuple[OncologyKnowledgeGraph, KnowledgeGraphSummary]:
        """
        Construct the knowledge graph and generate a summary.

        Args:
            drug_info: Resolved drug info with targets and mechanism.
            pathways: Pathway expansion results.
            all_evidence: All normalized evidence from all agents.
            trials: Clinical trial info objects.
            articles: Enriched PubMed articles.

        Returns:
            Tuple of (OncologyKnowledgeGraph, KnowledgeGraphSummary).
        """
        logger.info("[GraphAgent] Building knowledge graph for %s", drug_info.drug_name)

        kg = OncologyKnowledgeGraph()

        # Add drug node
        kg.add_drug(drug_info.drug_name, moa=drug_info.mechanism_of_action)

        # Add target genes
        for gene in drug_info.target_genes:
            kg.add_gene_target(drug_info.drug_name, gene, weight=3.0)

        # Add pathway expansions
        for pw in pathways:
            for gene in drug_info.target_genes:
                for pw_gene in pw.genes:
                    if pw_gene != gene:
                        kg.add_gene(pw_gene)
            # Add pathway node from first target gene perspective
            if drug_info.target_genes and pw.genes:
                primary_gene = drug_info.target_genes[0]
                try:
                    kg.add_pathway(primary_gene, pw)
                except Exception as exc:
                    logger.debug("[GraphAgent] Pathway add error: %s", exc)

        # Add all evidence items
        for ev in all_evidence:
            try:
                kg.add_evidence(ev)
            except Exception as exc:
                logger.debug("[GraphAgent] Evidence add error: %s", exc)

        # Add clinical trials
        all_genes = drug_info.target_genes + [
            ev.gene for ev in all_evidence if ev.gene
        ]
        all_genes = list(dict.fromkeys(all_genes))  # deduplicate
        for trial in trials:
            try:
                kg.add_clinical_trial(trial, all_genes)
            except Exception as exc:
                logger.debug("[GraphAgent] Trial add error: %s", exc)

        # Add literature mentions
        for article in articles[:50]:
            try:
                kg.add_pubmed_article(article, all_genes)
            except Exception as exc:
                logger.debug("[GraphAgent] Article add error: %s", exc)

        # Export graph files
        try:
            kg.export_all(OUTPUT_KG_GRAPHML, OUTPUT_KG_JSON)
        except Exception as exc:
            logger.error("[GraphAgent] KG export failed: %s", exc)

        # Build summary
        summary = kg.build_summary()
        logger.info(
            "[GraphAgent] KG built: %d nodes, %d edges, %d candidate biomarkers",
            summary.node_count,
            summary.edge_count,
            len(summary.candidate_biomarkers),
        )

        return kg, summary
