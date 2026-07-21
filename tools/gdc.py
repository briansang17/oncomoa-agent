"""
OncoMOA Tool — GDC (Genomic Data Commons) API
Fetches TCGA mutation frequencies, gene expression distributions,
and cancer type-level variant summaries.

Example:
    mutations = await fetch_gdc_mutations("KRAS")
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from config import GDC_BASE, HTTP_TIMEOUT
from models.schemas import NormalizedEvidence, EvidenceType, EvidenceDirection
from tools.cache import cached_api_call

logger = logging.getLogger(__name__)


@cached_api_call("gdc_gene_mutations")
async def fetch_gdc_mutations(gene_symbol: str, size: int = 20) -> list[dict[str, Any]]:
    """
    Fetch top somatic mutations for a gene from GDC/TCGA.

    Args:
        gene_symbol: HGNC gene symbol (e.g., "KRAS").
        size: Number of mutations to retrieve.

    Returns:
        List of mutation records with frequency and consequence data.
    """
    url = f"{GDC_BASE}/ssms"
    params = {
        "filters": (
            '{"op":"and","content":['
            f'{{"op":"=","content":{{"field":"consequence.transcript.gene.symbol","value":"{gene_symbol}"}}}},'
            '{"op":"=","content":{"field":"somatic","value":true}}'
            "]}"
        ),
        "fields": (
            "ssm_id,genomic_dna_change,consequence.transcript.consequence_type,"
            "consequence.transcript.gene.symbol,occurrence.case.disease_type,"
            "occurrence.case.primary_site"
        ),
        "format": "json",
        "size": size,
        "sort": "occurrence.case_count:desc",
    }

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.warning("GDC: non-200 response %d for %s", resp.status, gene_symbol)
                    return []
                data = await resp.json()

        hits = data.get("data", {}).get("hits", [])
        logger.info("GDC: found %d mutations for %s", len(hits), gene_symbol)
        return hits
    except Exception as exc:
        logger.error("GDC fetch_gdc_mutations failed for %s: %s", gene_symbol, exc)
        return []


@cached_api_call("gdc_gene_summary")
async def fetch_gdc_gene_summary(gene_symbol: str) -> dict[str, Any]:
    """
    Fetch GDC gene-level summary including case counts and cancer type distribution.

    Args:
        gene_symbol: HGNC gene symbol.

    Returns:
        Gene summary dict with occurrence statistics.
    """
    url = f"{GDC_BASE}/genes"
    params = {
        "filters": f'{{"op":"=","content":{{"field":"symbol","value":"{gene_symbol}"}}}}',
        "fields": (
            "id,symbol,name,biotype,description,"
            "is_cancer_gene_census,numFound"
        ),
        "format": "json",
        "size": 1,
    }

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return {}
                data = await resp.json()

        hits = data.get("data", {}).get("hits", [])
        return hits[0] if hits else {}
    except Exception as exc:
        logger.error("GDC fetch_gdc_gene_summary failed for %s: %s", gene_symbol, exc)
        return {}


def gdc_to_evidence(
    gene_symbol: str,
    mutations: list[dict[str, Any]],
) -> list[NormalizedEvidence]:
    """
    Convert GDC mutation records to NormalizedEvidence items.

    Args:
        gene_symbol: Gene symbol.
        mutations: List of GDC mutation records.

    Returns:
        List of NormalizedEvidence objects.
    """
    evidence_items: list[NormalizedEvidence] = []

    # Aggregate by consequence type
    consequence_counts: dict[str, int] = {}
    disease_set: set[str] = set()

    for mut in mutations:
        consequences = mut.get("consequence", [])
        for csq in consequences:
            csq_type = csq.get("transcript", {}).get("consequence_type", "")
            consequence_counts[csq_type] = consequence_counts.get(csq_type, 0) + 1

        for occurrence in mut.get("occurrence", [])[:3]:
            disease = occurrence.get("case", {}).get("disease_type", "")
            if disease:
                disease_set.add(disease)

    if mutations:
        top_csq = max(consequence_counts, key=consequence_counts.get, default="somatic")
        diseases_str = ", ".join(list(disease_set)[:3])

        evidence_items.append(
            NormalizedEvidence(
                source="GDC/TCGA",
                source_id=f"GDC_{gene_symbol}",
                gene=gene_symbol,
                disease=diseases_str,
                evidence_type=EvidenceType.ONCOGENIC,
                evidence_direction=EvidenceDirection.SUPPORTS,
                claim=(
                    f"{gene_symbol} has {len(mutations)} somatic mutations in TCGA "
                    f"(top consequence: {top_csq}; cancer types: {diseases_str})"
                ),
                strength=min(len(mutations) / 10.0, 2.0),
                raw_data={"mutation_count": len(mutations), "consequence_counts": consequence_counts},
            )
        )

    return evidence_items
