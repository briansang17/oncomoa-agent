"""
OncoMOA Tool — Ensembl REST API
Retrieves gene information, genomic coordinates, and variant data.

Example:
    gene_info = await fetch_ensembl_gene("KRAS")
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from config import ENSEMBL_REST_BASE, HTTP_TIMEOUT
from tools.cache import cached_api_call

logger = logging.getLogger(__name__)

ENSEMBL_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
}


@cached_api_call("ensembl_gene_lookup")
async def fetch_ensembl_gene(gene_symbol: str) -> dict[str, Any]:
    """
    Look up gene information from Ensembl by HGNC symbol.

    Returns gene ID, biotype, description, location, and synonyms.
    """
    url = f"{ENSEMBL_REST_BASE}/lookup/symbol/homo_sapiens/{gene_symbol}"
    params = {"expand": 1, "content-type": "application/json"}

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)) as session:
            async with session.get(url, params=params, headers=ENSEMBL_HEADERS) as resp:
                if resp.status == 400 or resp.status == 404:
                    logger.debug("Ensembl: gene %s not found", gene_symbol)
                    return {}
                resp.raise_for_status()
                data = await resp.json()
        logger.debug("Ensembl: found gene %s (id=%s)", gene_symbol, data.get("id", "?"))
        return data
    except Exception as exc:
        logger.error("Ensembl fetch_ensembl_gene failed for %s: %s", gene_symbol, exc)
        return {}


@cached_api_call("ensembl_variants")
async def fetch_ensembl_variants(gene_symbol: str) -> list[dict[str, Any]]:
    """
    Fetch somatic variants associated with a gene from Ensembl.

    Returns a list of variant dicts with consequence types.
    """
    gene_data = await fetch_ensembl_gene(gene_symbol)
    ensembl_id = gene_data.get("id", "")
    if not ensembl_id:
        return []

    url = f"{ENSEMBL_REST_BASE}/overlap/id/{ensembl_id}"
    params = {
        "feature": "variation",
        "content-type": "application/json",
        "variant_set": "phenotype_associated",
    }

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)) as session:
            async with session.get(url, params=params, headers=ENSEMBL_HEADERS) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
        variants = data if isinstance(data, list) else []
        logger.info("Ensembl: found %d variants for %s", len(variants), gene_symbol)
        return variants[:50]
    except Exception as exc:
        logger.error("Ensembl fetch_ensembl_variants failed for %s: %s", gene_symbol, exc)
        return []


def extract_ensembl_id(gene_data: dict[str, Any]) -> str | None:
    """Extract Ensembl gene ID from gene lookup response."""
    return gene_data.get("id")


def extract_canonical_transcript(gene_data: dict[str, Any]) -> str | None:
    """Extract canonical transcript ID from gene data."""
    transcripts = gene_data.get("Transcript", [])
    for t in transcripts:
        if t.get("is_canonical"):
            return t.get("id")
    return transcripts[0].get("id") if transcripts else None
