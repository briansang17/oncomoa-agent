"""
OncoMOA Agent — Clinical Evidence Agent
Aggregates CIViC predictive/prognostic evidence for target and candidate genes.
Computes evidence statistics and enriches with DGIdb and MyVariant data.

Example:
    agent = ClinicalEvidenceAgent()
    evidence, stats = await agent.run(["KRAS", "STK11"], drug_name="sotorasib")
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from models.schemas import NormalizedEvidence, EvidenceType
from tools.civic import fetch_civic_evidence, fetch_civic_summary_stats
from tools.dgidb import fetch_dgidb_interactions
from tools.myvariant import fetch_gene_variants, variants_to_evidence
from tools.cbioportal import fetch_mutation_frequency, cbioportal_to_evidence
from tools.gdc import fetch_gdc_mutations, gdc_to_evidence

logger = logging.getLogger(__name__)


class ClinicalEvidenceAgent:
    """
    Aggregates structured oncology evidence from CIViC, DGIdb, MyVariant,
    cBioPortal, and GDC for all target and candidate genes.
    """

    async def run(
        self,
        gene_symbols: list[str],
        drug_name: str = "",
    ) -> tuple[list[NormalizedEvidence], dict[str, Any]]:
        """
        Gather clinical evidence from all structured oncology databases.

        Args:
            gene_symbols: Combined list of target + candidate genes.
            drug_name: Drug name for relevance filtering.

        Returns:
            Tuple of (evidence_items, stats_dict).
        """
        logger.info(
            "[ClinicalEvidenceAgent] Gathering evidence for %d genes", len(gene_symbols)
        )

        # Cap gene list to prevent API overload
        genes = gene_symbols[:20]

        # Run all evidence sources in parallel
        civic_task = fetch_civic_evidence(genes, drug_name=drug_name)
        dgidb_task = fetch_dgidb_interactions(genes[:10])
        myvariant_tasks = [fetch_gene_variants(g, size=5) for g in genes[:8]]
        cbio_tasks = [fetch_mutation_frequency(g) for g in genes[:5]]
        gdc_tasks = [fetch_gdc_mutations(g, size=10) for g in genes[:5]]

        (
            civic_evidence,
            dgidb_evidence,
            *rest_results,
        ) = await asyncio.gather(
            civic_task,
            dgidb_task,
            *myvariant_tasks,
            *cbio_tasks,
            *gdc_tasks,
            return_exceptions=True,
        )

        all_evidence: list[NormalizedEvidence] = []

        # CIViC
        if isinstance(civic_evidence, list):
            all_evidence.extend(civic_evidence)
            logger.info("[ClinicalEvidenceAgent] CIViC: %d items", len(civic_evidence))
        elif isinstance(civic_evidence, Exception):
            logger.error("[ClinicalEvidenceAgent] CIViC failed: %s", civic_evidence)

        # DGIdb
        if isinstance(dgidb_evidence, list):
            all_evidence.extend(dgidb_evidence)
            logger.info("[ClinicalEvidenceAgent] DGIdb: %d items", len(dgidb_evidence))
        elif isinstance(dgidb_evidence, Exception):
            logger.error("[ClinicalEvidenceAgent] DGIdb failed: %s", dgidb_evidence)

        # MyVariant results
        n_myvariant = len(genes[:8])
        myvariant_raw = rest_results[:n_myvariant]
        cbio_raw = rest_results[n_myvariant : n_myvariant + len(genes[:5])]
        gdc_raw = rest_results[n_myvariant + len(genes[:5]):]

        for gene, mv_result in zip(genes[:8], myvariant_raw):
            if isinstance(mv_result, list):
                mv_evidence = variants_to_evidence(gene, mv_result)
                all_evidence.extend(mv_evidence)
            elif isinstance(mv_result, Exception):
                logger.debug("[ClinicalEvidenceAgent] MyVariant failed for %s: %s", gene, mv_result)

        # cBioPortal
        for gene, cbio_result in zip(genes[:5], cbio_raw):
            if isinstance(cbio_result, list):
                cbio_evidence = cbioportal_to_evidence(gene, cbio_result)
                all_evidence.extend(cbio_evidence)
            elif isinstance(cbio_result, Exception):
                logger.debug("[ClinicalEvidenceAgent] cBioPortal failed for %s: %s", gene, cbio_result)

        # GDC
        for gene, gdc_result in zip(genes[:5], gdc_raw):
            if isinstance(gdc_result, list):
                gdc_ev = gdc_to_evidence(gene, gdc_result)
                all_evidence.extend(gdc_ev)
            elif isinstance(gdc_result, Exception):
                logger.debug("[ClinicalEvidenceAgent] GDC failed for %s: %s", gene, gdc_result)

        # Compute summary stats
        stats = await fetch_civic_summary_stats(all_evidence)
        stats["total_all_sources"] = len(all_evidence)
        stats["sources"] = list({e.source for e in all_evidence})

        logger.info(
            "[ClinicalEvidenceAgent] Total evidence items: %d from sources: %s",
            len(all_evidence),
            stats["sources"],
        )
        return all_evidence, stats
