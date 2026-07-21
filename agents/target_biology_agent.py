"""
OncoMOA Agent — Target Biology Agent
Enriches target gene(s) with biological context from Ensembl, MyGene,
UniProt, and Open Targets disease associations. Runs all queries in parallel.

Example:
    agent = TargetBiologyAgent()
    gene_infos = await agent.run(["KRAS", "STK11"])
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from models.schemas import GeneInfo, NormalizedEvidence
from tools.ensembl import fetch_ensembl_gene
from tools.mygene import fetch_mygene_info, extract_aliases, extract_gene_summary, extract_pathways
from tools.uniprot import fetch_uniprot_info, extract_protein_function, extract_disease_associations
from tools.open_targets import fetch_target_disease_associations

logger = logging.getLogger(__name__)


class TargetBiologyAgent:
    """
    Enriches target genes with multi-source biological information.
    All API calls run in parallel for each gene.
    """

    async def _enrich_gene(self, gene_symbol: str) -> tuple[GeneInfo, list[NormalizedEvidence]]:
        """
        Enrich a single gene with Ensembl, MyGene, UniProt, and Open Targets data.

        Returns:
            Tuple of (GeneInfo, list of NormalizedEvidence from OT disease associations).
        """
        ensembl_data, mygene_data, uniprot_data, ot_evidence = await asyncio.gather(
            fetch_ensembl_gene(gene_symbol),
            fetch_mygene_info(gene_symbol),
            fetch_uniprot_info(gene_symbol),
            fetch_target_disease_associations(gene_symbol),
            return_exceptions=True,
        )

        ensembl_id: str | None = None
        uniprot_id: str | None = None
        full_name: str = ""
        summary: str = ""
        pathways: list[str] = []
        diseases: list[str] = []
        aliases: list[str] = []
        evidence_items: list[NormalizedEvidence] = []

        if isinstance(ensembl_data, dict) and ensembl_data:
            ensembl_id = ensembl_data.get("id")
            full_name = ensembl_data.get("description", "").split("[")[0].strip()
        elif isinstance(ensembl_data, Exception):
            logger.debug("[TargetBiologyAgent] Ensembl failed for %s: %s", gene_symbol, ensembl_data)

        if isinstance(mygene_data, dict) and mygene_data:
            summary = extract_gene_summary(mygene_data)
            aliases = extract_aliases(mygene_data)
            mygene_pathways = extract_pathways(mygene_data)
            pathways = [p["name"] for p in mygene_pathways if p.get("name")][:10]
            if not full_name:
                full_name = mygene_data.get("name", "")
        elif isinstance(mygene_data, Exception):
            logger.debug("[TargetBiologyAgent] MyGene failed for %s: %s", gene_symbol, mygene_data)

        if isinstance(uniprot_data, dict) and uniprot_data:
            if not summary:
                summary = extract_protein_function(uniprot_data)
            uniprot_id = uniprot_data.get("primaryAccession")
            diseases = extract_disease_associations(uniprot_data)
        elif isinstance(uniprot_data, Exception):
            logger.debug("[TargetBiologyAgent] UniProt failed for %s: %s", gene_symbol, uniprot_data)

        if isinstance(ot_evidence, list):
            evidence_items = ot_evidence
            for ev in ot_evidence:
                if ev.disease and ev.disease not in diseases:
                    diseases.append(ev.disease)
        elif isinstance(ot_evidence, Exception):
            logger.debug("[TargetBiologyAgent] OT disease assoc failed for %s: %s", gene_symbol, ot_evidence)

        gene_info = GeneInfo(
            symbol=gene_symbol,
            ensembl_id=ensembl_id,
            uniprot_id=uniprot_id,
            full_name=full_name,
            summary=summary[:500] if summary else "",
            pathways=pathways,
            associated_diseases=diseases[:10],
            aliases=aliases[:10],
        )

        logger.info(
            "[TargetBiologyAgent] %s enriched: %d pathways, %d diseases, %d evidence items",
            gene_symbol, len(pathways), len(diseases), len(evidence_items),
        )
        return gene_info, evidence_items

    async def run(
        self, gene_symbols: list[str]
    ) -> tuple[list[GeneInfo], list[NormalizedEvidence]]:
        """
        Enrich all target genes in parallel.

        Args:
            gene_symbols: List of HGNC gene symbols from DrugAgent.

        Returns:
            Tuple of (list of GeneInfo, combined list of NormalizedEvidence).
        """
        if not gene_symbols:
            return [], []

        logger.info("[TargetBiologyAgent] Enriching %d genes: %s", len(gene_symbols), gene_symbols[:5])

        tasks = [self._enrich_gene(gene) for gene in gene_symbols[:10]]  # cap at 10 genes
        results = await asyncio.gather(*tasks, return_exceptions=True)

        gene_infos: list[GeneInfo] = []
        all_evidence: list[NormalizedEvidence] = []

        for gene, result in zip(gene_symbols[:10], results):
            if isinstance(result, Exception):
                logger.error("[TargetBiologyAgent] Enrichment failed for %s: %s", gene, result)
            else:
                info, evidence = result
                gene_infos.append(info)
                all_evidence.extend(evidence)

        return gene_infos, all_evidence
