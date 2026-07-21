"""
OncoMOA Tool — cBioPortal API
Fetches mutation frequencies, copy number alterations, and co-mutation data
from TCGA and other cancer genomics studies.

Example:
    freq = await fetch_mutation_frequency("KRAS", cancer_type="luad")
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from config import CBIOPORTAL_BASE, HTTP_TIMEOUT
from models.schemas import NormalizedEvidence, EvidenceType, EvidenceDirection
from tools.cache import cached_api_call

logger = logging.getLogger(__name__)

# Common TCGA study IDs for major cancer types
TCGA_STUDIES: dict[str, str] = {
    "luad": "luad_tcga_pan_can_atlas_2018",
    "lusc": "lusc_tcga_pan_can_atlas_2018",
    "brca": "brca_tcga_pan_can_atlas_2018",
    "crc": "coadread_tcga_pan_can_atlas_2018",
    "paad": "paad_tcga_pan_can_atlas_2018",
    "prad": "prad_tcga_pan_can_atlas_2018",
    "ov": "ov_tcga_pan_can_atlas_2018",
    "gbm": "gbm_tcga_pan_can_atlas_2018",
    "skcm": "skcm_tcga_pan_can_atlas_2018",
    "blca": "blca_tcga_pan_can_atlas_2018",
    "hnsc": "hnsc_tcga_pan_can_atlas_2018",
    "ucec": "ucec_tcga_pan_can_atlas_2018",
    "stad": "stad_tcga_pan_can_atlas_2018",
    "lihc": "lihc_tcga_pan_can_atlas_2018",
    "kirc": "kirc_tcga_pan_can_atlas_2018",
}


@cached_api_call("cbioportal_mutation_freq")
async def fetch_mutation_frequency(
    gene_symbol: str,
    study_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch mutation frequency for a gene across TCGA studies.

    Args:
        gene_symbol: HGNC gene symbol.
        study_ids: Optional list of study IDs; defaults to all TCGA studies.

    Returns:
        List of dicts with study_id, total_samples, mutated_samples, frequency.
    """
    if study_ids is None:
        study_ids = list(TCGA_STUDIES.values())

    results: list[dict[str, Any]] = []

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)) as session:
            for study_id in study_ids[:5]:  # Limit to 5 studies to stay fast
                url = f"{CBIOPORTAL_BASE}/molecular-profiles/{study_id}_mutations/gene-panel-data"
                # Use the mutations endpoint
                mut_url = f"{CBIOPORTAL_BASE}/studies/{study_id}/mutations"
                params = {"entrezGeneId": gene_symbol, "projection": "SUMMARY"}

                try:
                    async with session.get(mut_url, params=params) as resp:
                        if resp.status != 200:
                            continue
                        mutations = await resp.json()

                    if isinstance(mutations, list) and mutations:
                        results.append({
                            "study_id": study_id,
                            "mutation_count": len(mutations),
                            "gene": gene_symbol,
                            "sample_mutations": mutations[:5],
                        })
                except Exception:
                    continue

        logger.info(
            "cBioPortal: found mutation data in %d studies for %s",
            len(results),
            gene_symbol,
        )
    except Exception as exc:
        logger.error("cBioPortal fetch_mutation_frequency failed for %s: %s", gene_symbol, exc)

    return results


@cached_api_call("cbioportal_gene_panel")
async def fetch_cbioportal_gene_summary(gene_symbol: str) -> dict[str, Any]:
    """
    Fetch gene-level summary from cBioPortal including alteration frequency.

    Args:
        gene_symbol: HGNC gene symbol.

    Returns:
        Dict with gene info and alteration summary.
    """
    url = f"{CBIOPORTAL_BASE}/genes/{gene_symbol}"
    params = {"geneIdType": "HUGO_GENE_SYMBOL"}

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return {}
                data = await resp.json()
        return data
    except Exception as exc:
        logger.error("cBioPortal fetch_gene_summary failed for %s: %s", gene_symbol, exc)
        return {}


def cbioportal_to_evidence(
    gene_symbol: str,
    frequency_data: list[dict[str, Any]],
) -> list[NormalizedEvidence]:
    """
    Convert cBioPortal mutation frequency data to NormalizedEvidence items.

    Args:
        gene_symbol: Gene symbol.
        frequency_data: List of study frequency records.

    Returns:
        List of NormalizedEvidence objects.
    """
    evidence_items: list[NormalizedEvidence] = []

    for study_data in frequency_data:
        study_id = study_data.get("study_id", "")
        count = study_data.get("mutation_count", 0)
        if count == 0:
            continue

        cancer_type = study_id.split("_")[0].upper() if study_id else "CANCER"
        claim = (
            f"{gene_symbol} mutated in {count} samples "
            f"in {cancer_type} (TCGA: {study_id})"
        )

        evidence_items.append(
            NormalizedEvidence(
                source="cBioPortal",
                source_id=f"CBIO_{gene_symbol}_{study_id[:20]}",
                gene=gene_symbol,
                disease=cancer_type,
                evidence_type=EvidenceType.ONCOGENIC,
                evidence_direction=EvidenceDirection.SUPPORTS,
                claim=claim,
                strength=min(count / 50.0, 2.0),
                raw_data=study_data,
            )
        )

    return evidence_items
