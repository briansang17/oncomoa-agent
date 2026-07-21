"""
OncoMOA Tool — UniProt REST API
Retrieves protein function, pathway involvement, and disease associations.

Example:
    protein_info = await fetch_uniprot_info("KRAS")
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from config import UNIPROT_BASE, HTTP_TIMEOUT
from tools.cache import cached_api_call

logger = logging.getLogger(__name__)


@cached_api_call("uniprot_gene")
async def fetch_uniprot_info(gene_symbol: str) -> dict[str, Any]:
    """
    Fetch UniProt protein entry for a human gene symbol.

    Retrieves: protein function, keywords, pathways, disease associations,
    and subcellular localization.

    Args:
        gene_symbol: HGNC gene symbol (e.g., "KRAS").

    Returns:
        UniProt protein entry dict.
    """
    url = f"{UNIPROT_BASE}/uniprotkb/search"
    params = {
        "query": f"gene_exact:{gene_symbol} AND organism_id:9606 AND reviewed:true",
        "fields": (
            "accession,gene_names,protein_name,cc_function,keyword,"
            "cc_pathway,cc_disease,go,cc_subcellular_location"
        ),
        "format": "json",
        "size": 1,
    }

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)) as session:
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()

        results = data.get("results", [])
        if not results:
            logger.debug("UniProt: no entry for gene %s", gene_symbol)
            return {}

        entry = results[0]
        logger.debug(
            "UniProt: found %s (%s)",
            gene_symbol,
            entry.get("primaryAccession", ""),
        )
        return entry
    except Exception as exc:
        logger.error("UniProt fetch_uniprot_info failed for %s: %s", gene_symbol, exc)
        return {}


def extract_protein_function(entry: dict[str, Any]) -> str:
    """Extract plain-text protein function from UniProt entry."""
    comments = entry.get("comments", [])
    for comment in comments:
        if comment.get("commentType") == "FUNCTION":
            texts = comment.get("texts", [])
            if texts:
                return texts[0].get("value", "")
    return ""


def extract_disease_associations(entry: dict[str, Any]) -> list[str]:
    """Extract disease names from UniProt disease associations."""
    diseases: list[str] = []
    comments = entry.get("comments", [])
    for comment in comments:
        if comment.get("commentType") == "DISEASE":
            disease = comment.get("disease", {})
            disease_name = disease.get("diseaseId", disease.get("description", ""))
            if disease_name:
                diseases.append(disease_name)
    return diseases


def extract_pathway_keywords(entry: dict[str, Any]) -> list[str]:
    """Extract pathway-related keywords from UniProt entry."""
    keywords = entry.get("keywords", [])
    pathway_terms: list[str] = []
    for kw in keywords:
        name = kw.get("name", "")
        category = kw.get("category", "")
        if category in ("Biological process", "Molecular function", "Pathway") and name:
            pathway_terms.append(name)
    return pathway_terms
