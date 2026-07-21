"""
OncoMOA Agent — Pathway Expansion Agent
Expands target genes into pathway-connected candidate biomarker genes
using Reactome and WikiPathways. Prioritizes biologically connected genes.

Example:
    agent = PathwayAgent()
    candidates, pathways = await agent.run(["KRAS", "NRAS"])
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter

from models.schemas import PathwayInfo, GeneInfo
from tools.pathways import fetch_gene_pathways, get_pathway_candidate_genes
from tools.mygene import batch_mygene_query, extract_pathways as extract_mygene_pathways

logger = logging.getLogger(__name__)

# Known high-value pathway gene sets for key cancer targets
KNOWN_PATHWAY_GENES: dict[str, list[str]] = {
    "KRAS": ["NRAS", "HRAS", "BRAF", "RAF1", "MAP2K1", "MAPK1", "MAPK3", "PIK3CA", "AKT1", "MTOR",
             "STK11", "KEAP1", "CDKN2A", "TP53", "NF1"],
    "EGFR": ["HER2", "HER3", "HER4", "PIK3CA", "AKT1", "PTEN", "MET", "KRAS", "BRAF", "RB1"],
    "BRAF": ["KRAS", "NRAS", "MEK1", "MAP2K1", "ERK1", "MAPK1", "CDKN2A", "PTEN", "PIK3CA"],
    "PARP1": ["BRCA1", "BRCA2", "PALB2", "ATM", "CHEK2", "RAD51C", "RAD51D", "BRIP1", "CDK12"],
    "PARP2": ["BRCA1", "BRCA2", "PALB2", "ATM", "CHEK2", "RAD51C", "RAD51D"],
    "PDCD1": ["CD274", "HAVCR2", "LAG3", "TIGIT", "CD8A", "FOXP3", "TMB", "MSI", "HLA-A"],
    "CD274": ["PDCD1", "HAVCR2", "CD8A", "FOXP3", "LAG3", "TIGIT", "TMB"],
    "ALK": ["EML4", "NPM1", "KRAS", "EGFR", "MET", "RET", "ROS1"],
    "MET": ["HGF", "EGFR", "KRAS", "RET", "ALK", "PIK3CA", "AKT1"],
    "ERBB2": ["EGFR", "HER3", "PIK3CA", "AKT1", "PTEN", "TP53", "CCND1"],
}


class PathwayAgent:
    """
    Expands target genes into pathway-connected candidate biomarker genes.
    Uses Reactome, WikiPathways, and curated cancer pathway gene sets.
    """

    async def run(
        self,
        target_genes: list[str],
        gene_infos: list[GeneInfo] | None = None,
    ) -> tuple[list[str], list[PathwayInfo]]:
        """
        Generate candidate biomarker genes via pathway expansion.

        Args:
            target_genes: Direct drug targets from DrugAgent.
            gene_infos: Enriched gene data from TargetBiologyAgent (optional).

        Returns:
            Tuple of (candidate_genes, pathway_info_list).
        """
        logger.info("[PathwayAgent] Expanding %d target genes", len(target_genes))

        all_pathways: list[PathwayInfo] = []
        candidate_gene_counts: Counter = Counter()

        # Add known pathway genes from curated sets
        for gene in target_genes:
            known = KNOWN_PATHWAY_GENES.get(gene.upper(), [])
            for kg in known:
                candidate_gene_counts[kg] += 3  # curated = high weight

        # Also add pathway genes from gene_info summaries
        if gene_infos:
            for info in gene_infos:
                for pw_name in info.pathways:
                    # Extract gene-like tokens from pathway names (heuristic)
                    tokens = [t for t in pw_name.split() if t.isupper() and 2 <= len(t) <= 10]
                    for t in tokens:
                        candidate_gene_counts[t] += 1

        # Fetch pathways from Reactome + WikiPathways in parallel
        pathway_tasks = [fetch_gene_pathways(gene) for gene in target_genes[:5]]
        pathway_results = await asyncio.gather(*pathway_tasks, return_exceptions=True)

        for gene, result in zip(target_genes[:5], pathway_results):
            if isinstance(result, Exception):
                logger.warning("[PathwayAgent] Pathway fetch failed for %s: %s", gene, result)
                continue
            all_pathways.extend(result)
            pw_genes = get_pathway_candidate_genes(result)
            for g in pw_genes:
                candidate_gene_counts[g] += 1

        # Build final candidate list (exclude target genes themselves, prioritize by count)
        target_set = {g.upper() for g in target_genes}
        candidates: list[str] = []
        for gene, count in candidate_gene_counts.most_common(50):
            clean = gene.strip()
            if clean.upper() not in target_set and len(clean) <= 15 and clean.isalpha() or "-" in clean:
                candidates.append(clean)
            if len(candidates) >= 30:
                break

        logger.info(
            "[PathwayAgent] Generated %d candidate genes from %d pathways",
            len(candidates),
            len(all_pathways),
        )
        return candidates, all_pathways
