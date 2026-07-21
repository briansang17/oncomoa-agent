"""
OncoMOA Tool — MyGene.info
Retrieves gene aliases, pathway memberships, summaries, and external IDs.

Example:
    info = await fetch_mygene_info("KRAS")
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from config import MYGENE_BASE, HTTP_TIMEOUT
from tools.cache import cached_api_call

logger = logging.getLogger(__name__)


@cached_api_call("mygene_query")
async def fetch_mygene_info(gene_symbol: str) -> dict[str, Any]:
    """
    Fetch comprehensive gene information from MyGene.info using the /query endpoint.

    Returns aliases, summary, pathways (Reactome, KEGG, GO), and external IDs.
    """
    url = f"{MYGENE_BASE}/query"
    params = {
        "q": f"symbol:{gene_symbol}",
        "species": "human",
        "fields": "symbol,name,summary,alias,pathway,ensembl,uniprot,OMIM,entrezgene",
        "size": 1,
    }

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)) as session:
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()

        hits = data.get("hits", [])
        if not hits:
            logger.debug("MyGene: no result for %s", gene_symbol)
            return {}

        gene_data = hits[0]
        logger.debug("MyGene: found %s (entrez=%s)", gene_symbol, gene_data.get("entrezgene"))
        return gene_data
    except Exception as exc:
        logger.error("MyGene fetch_mygene_info failed for %s: %s", gene_symbol, exc)
        return {}


def extract_aliases(gene_data: dict[str, Any]) -> list[str]:
    """Extract all aliases and synonyms from MyGene data."""
    aliases = gene_data.get("alias", [])
    if isinstance(aliases, str):
        aliases = [aliases]
    return aliases if isinstance(aliases, list) else []


def extract_pathways(gene_data: dict[str, Any]) -> list[dict[str, str]]:
    """
    Extract pathway memberships (Reactome, KEGG, WikiPathways) from MyGene data.

    Returns list of dicts with keys: source, id, name.
    """
    pathways: list[dict[str, str]] = []
    pathway_data = gene_data.get("pathway", {})
    if not isinstance(pathway_data, dict):
        return pathways

    for source in ["reactome", "kegg", "wikipathways", "pid", "biocarta"]:
        entries = pathway_data.get(source, [])
        if isinstance(entries, dict):
            entries = [entries]
        for entry in entries:
            if isinstance(entry, dict):
                pathways.append({
                    "source": source,
                    "id": str(entry.get("id", "")),
                    "name": str(entry.get("name", "")),
                })

    return pathways


def extract_gene_summary(gene_data: dict[str, Any]) -> str:
    """Extract gene summary text from MyGene data."""
    return gene_data.get("summary", "")


async def batch_mygene_query(gene_symbols: list[str]) -> list[dict[str, Any]]:
    """
    Batch query MyGene.info for multiple gene symbols in one POST request.

    Args:
        gene_symbols: List of HGNC gene symbols.

    Returns:
        List of gene info dicts.
    """
    if not gene_symbols:
        return []

    url = f"{MYGENE_BASE}/query"
    body = {
        "q": ",".join(gene_symbols),
        "species": "human",
        "fields": "symbol,name,summary,alias,pathway,ensembl",
        "scopes": "symbol",
    }

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)) as session:
            async with session.post(url, json=body) as resp:
                resp.raise_for_status()
                data = await resp.json()
        logger.info("MyGene batch: retrieved %d gene records", len(data))
        return data if isinstance(data, list) else []
    except Exception as exc:
        logger.error("MyGene batch_mygene_query failed: %s", exc)
        return []
