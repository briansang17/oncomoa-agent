"""
OncoMOA Tool — MyVariant.info
Fetches variant pathogenicity, ClinVar significance, and oncogenicity annotations.

Example:
    variant_info = await fetch_variant_info("chr12:g.25398284C>A")  # KRAS G12C
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from config import MYVARIANT_BASE, HTTP_TIMEOUT
from models.schemas import NormalizedEvidence, EvidenceType, EvidenceDirection
from tools.cache import cached_api_call

logger = logging.getLogger(__name__)

# Common oncogenic mutations for key cancer genes (HGVS notation)
COMMON_ONCOGENIC_VARIANTS: dict[str, list[str]] = {
    "KRAS": ["chr12:g.25398284C>A"],  # G12C
    "BRAF": ["chr7:g.140453136A>T"],  # V600E
    "EGFR": ["chr7:g.55259515T>G"],   # L858R
    "TP53": ["chr17:g.7674220C>T"],   # R175H
    "PIK3CA": ["chr3:g.179234297A>G"], # H1047R
    "BRCA1": ["chr17:g.43094077G>A"],
    "BRCA2": ["chr13:g.32340300A>T"],
}


@cached_api_call("myvariant_query")
async def fetch_variant_info(hgvs_id: str) -> dict[str, Any]:
    """
    Fetch annotation for a specific variant using HGVS notation.

    Args:
        hgvs_id: HGVS variant identifier (e.g., "chr12:g.25398284C>A").

    Returns:
        Variant annotation dict with ClinVar, OncoKB, and pathogenicity info.
    """
    url = f"{MYVARIANT_BASE}/variant/{hgvs_id}"
    params = {
        "fields": "clinvar,civic,cadd,dbnsfp,gnomad,oncokb",
        "assembly": "hg38",
    }

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)) as session:
            async with session.get(url, params=params) as resp:
                if resp.status == 404:
                    return {}
                resp.raise_for_status()
                return await resp.json()
    except Exception as exc:
        logger.error("MyVariant fetch_variant_info failed for %s: %s", hgvs_id, exc)
        return {}


@cached_api_call("myvariant_gene_search")
async def fetch_gene_variants(gene_symbol: str, size: int = 20) -> list[dict[str, Any]]:
    """
    Search MyVariant.info for cancer-relevant variants in a gene.

    Args:
        gene_symbol: HGNC gene symbol.
        size: Maximum number of variants to return.

    Returns:
        List of variant annotation dicts.
    """
    url = f"{MYVARIANT_BASE}/query"
    params = {
        "q": f"clinvar.gene.symbol:{gene_symbol} AND clinvar.rcv.clinical_significance:Pathogenic",
        "fields": "clinvar,cadd,dbnsfp",
        "size": size,
        "assembly": "hg38",
    }

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)) as session:
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
        hits = data.get("hits", [])
        logger.info("MyVariant: found %d variants for gene %s", len(hits), gene_symbol)
        return hits
    except Exception as exc:
        logger.error("MyVariant fetch_gene_variants failed for %s: %s", gene_symbol, exc)
        return []


def variants_to_evidence(
    gene_symbol: str,
    variants: list[dict[str, Any]],
) -> list[NormalizedEvidence]:
    """
    Convert MyVariant.info hits to NormalizedEvidence items.

    Args:
        gene_symbol: Gene the variants belong to.
        variants: List of raw variant dicts from MyVariant.

    Returns:
        List of NormalizedEvidence objects.
    """
    evidence_items: list[NormalizedEvidence] = []

    for var in variants:
        clinvar = var.get("clinvar", {})
        variant_id = var.get("_id", "")
        rcv = clinvar.get("rcv", {})
        if isinstance(rcv, list):
            rcv = rcv[0] if rcv else {}

        significance = rcv.get("clinical_significance", "")
        conditions = rcv.get("conditions", {})
        disease_name = ""
        if isinstance(conditions, dict):
            disease_name = conditions.get("name", "")
        elif isinstance(conditions, list) and conditions:
            disease_name = conditions[0].get("name", "")

        hgvs = clinvar.get("hgvs", {}).get("coding", variant_id)
        strength = 2.0 if "Pathogenic" in significance else 0.5

        if strength > 0:
            evidence_items.append(
                NormalizedEvidence(
                    source="MyVariant/ClinVar",
                    source_id=f"CLINVAR_{variant_id}",
                    gene=gene_symbol,
                    variant=hgvs,
                    disease=disease_name,
                    evidence_type=EvidenceType.PREDISPOSING,
                    evidence_direction=EvidenceDirection.SUPPORTS,
                    claim=f"{gene_symbol} {hgvs} is {significance} for {disease_name}",
                    strength=strength,
                    raw_data=var,
                )
            )

    return evidence_items
