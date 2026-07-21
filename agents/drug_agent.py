"""
OncoMOA Agent — Drug Agent
Resolves drug name to targets, mechanism, synonyms, and ChEMBL/Open Targets metadata.
Runs Open Targets + ChEMBL queries in parallel.

Example:
    agent = DrugAgent()
    drug_info = await agent.run("sotorasib", "KRAS G12C covalent inhibitor")
"""

from __future__ import annotations

import asyncio
import logging

from models.schemas import DrugInfo
from tools.open_targets import fetch_drug_targets
from tools.chembl import fetch_chembl_drug_targets, extract_gene_symbols_from_chembl, search_chembl_drug

logger = logging.getLogger(__name__)


class DrugAgent:
    """
    Resolves a drug name into structured target and mechanism data
    by querying Open Targets and ChEMBL in parallel.
    """

    async def run(self, drug_name: str, moa_description: str = "") -> DrugInfo:
        """
        Fetch drug information from Open Targets and ChEMBL.

        Args:
            drug_name: Drug name (e.g., "sotorasib").
            moa_description: Human-provided MOA description for context.

        Returns:
            DrugInfo with target_genes, mechanism_of_action, synonyms, chembl_id.
        """
        logger.info("[DrugAgent] Resolving drug: %s", drug_name)

        # Run Open Targets and ChEMBL in parallel
        ot_result, chembl_targets, chembl_mol = await asyncio.gather(
            fetch_drug_targets(drug_name),
            fetch_chembl_drug_targets(drug_name),
            search_chembl_drug(drug_name),
            return_exceptions=True,
        )

        target_genes: list[str] = []
        mechanisms: list[str] = []
        diseases: list[str] = []
        sources: list[str] = []
        chembl_id: str | None = None

        # Open Targets results
        if isinstance(ot_result, dict) and ot_result:
            ot_genes = ot_result.get("target_genes", [])
            target_genes.extend(ot_genes)
            mechanisms.extend(ot_result.get("mechanisms", []))
            diseases.extend(ot_result.get("linked_diseases", []))
            if ot_genes:
                sources.append("OpenTargets")
        elif isinstance(ot_result, Exception):
            logger.warning("[DrugAgent] Open Targets failed: %s", ot_result)

        # ChEMBL results
        if isinstance(chembl_targets, list) and chembl_targets:
            chembl_genes = extract_gene_symbols_from_chembl(chembl_targets)
            for gene in chembl_genes:
                if gene not in target_genes:
                    target_genes.append(gene)
            sources.append("ChEMBL")
        elif isinstance(chembl_targets, Exception):
            logger.warning("[DrugAgent] ChEMBL targets failed: %s", chembl_targets)

        if isinstance(chembl_mol, dict) and chembl_mol:
            chembl_id = chembl_mol.get("molecule_chembl_id")
        elif isinstance(chembl_mol, Exception):
            logger.warning("[DrugAgent] ChEMBL mol search failed: %s", chembl_mol)

        # Use MOA description as fallback mechanism text
        moa_text = mechanisms[0] if mechanisms else moa_description

        drug_info = DrugInfo(
            drug_name=drug_name,
            mechanism_of_action=moa_text,
            target_genes=target_genes,
            chembl_id=chembl_id,
            sources=sources,
        )

        logger.info(
            "[DrugAgent] Resolved %s → %d target genes: %s",
            drug_name,
            len(target_genes),
            target_genes[:5],
        )
        return drug_info
