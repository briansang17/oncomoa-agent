"""
OncoMOA Tool — Pathway Databases
Fetches pathway membership from Reactome and WikiPathways to generate
candidate biomarker genes via pathway expansion.

Example:
    pathways = await fetch_gene_pathways("KRAS")
    candidate_genes = get_pathway_genes(pathways)
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from config import REACTOME_BASE, WIKIPATHWAYS_BASE, HTTP_TIMEOUT
from models.schemas import PathwayInfo
from tools.cache import cached_api_call

logger = logging.getLogger(__name__)


@cached_api_call("reactome_pathways")
async def fetch_reactome_pathways(gene_symbol: str) -> list[PathwayInfo]:
    """
    Fetch Reactome pathways for a gene symbol via the Content Service.

    Args:
        gene_symbol: HGNC gene symbol (e.g., "KRAS").

    Returns:
        List of PathwayInfo objects containing pathway name and member genes.
    """
    pathways: list[PathwayInfo] = []

    # Step 1: Map gene symbol → UniProt → Reactome pathway list
    mapping_url = f"{REACTOME_BASE}/data/mapping/UniProt/{gene_symbol}/pathways"
    # Try direct gene name lookup instead
    lookup_url = f"{REACTOME_BASE}/data/query/enhanced/{gene_symbol}"

    # Use the pathway lookup by gene name
    pathways_url = f"{REACTOME_BASE}/data/pathways/low/entity/{gene_symbol}/allForms"

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)) as session:
            async with session.get(
                f"{REACTOME_BASE}/data/pathways/low/diagram/entity/{gene_symbol}/allForms",
                headers={"Accept": "application/json"},
            ) as resp:
                if resp.status != 200:
                    # Fallback: search by gene symbol
                    async with session.get(
                        f"{REACTOME_BASE}/search/query?query={gene_symbol}&types=Pathway&cluster=true",
                        headers={"Accept": "application/json"},
                    ) as search_resp:
                        if search_resp.status != 200:
                            return []
                        search_data = await search_resp.json()
                        pathway_hits = search_data.get("results", [{}])
                        pathway_hits = pathway_hits[0].get("entries", []) if pathway_hits else []
                else:
                    pathway_hits_raw = await resp.json()
                    pathway_hits = pathway_hits_raw if isinstance(pathway_hits_raw, list) else []

            # Fetch participant genes for top pathways (limit to 5 to stay fast)
            for pathway in pathway_hits[:5]:
                pathway_id = pathway.get("stId", pathway.get("id", ""))
                pathway_name = pathway.get("displayName", pathway.get("name", ""))
                if not pathway_id:
                    continue

                # Get pathway participants
                async with session.get(
                    f"{REACTOME_BASE}/data/pathway/{pathway_id}/containedEvents",
                    headers={"Accept": "application/json"},
                ) as parts_resp:
                    genes_in_pathway: list[str] = [gene_symbol]
                    if parts_resp.status == 200:
                        parts_data = await parts_resp.json()
                        for event in (parts_data if isinstance(parts_data, list) else [])[:50]:
                            display_name = event.get("displayName", "")
                            # Extract gene symbols from display names (heuristic)
                            if display_name and len(display_name) < 20:
                                genes_in_pathway.append(display_name.split(" ")[0])

                pathways.append(
                    PathwayInfo(
                        pathway_id=pathway_id,
                        pathway_name=pathway_name,
                        source="Reactome",
                        genes=list(dict.fromkeys(genes_in_pathway)),
                    )
                )

    except Exception as exc:
        logger.error("Reactome fetch failed for %s: %s", gene_symbol, exc)

    logger.info("Reactome: found %d pathways for %s", len(pathways), gene_symbol)
    return pathways


@cached_api_call("wikipathways_pathways")
async def fetch_wikipathways(gene_symbol: str) -> list[PathwayInfo]:
    """
    Fetch WikiPathways entries for a gene symbol.

    Args:
        gene_symbol: HGNC gene symbol.

    Returns:
        List of PathwayInfo objects.
    """
    pathways: list[PathwayInfo] = []
    url = f"{WIKIPATHWAYS_BASE}/findPathwaysByGene"
    params = {"query": gene_symbol, "species": "Homo sapiens", "format": "json"}

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json(content_type=None)

        results = data.get("result", [])
        for pw in results[:10]:
            pw_id = pw.get("id", "")
            pw_name = pw.get("name", "")
            if pw_id:
                pathways.append(
                    PathwayInfo(
                        pathway_id=pw_id,
                        pathway_name=pw_name,
                        source="WikiPathways",
                        genes=[gene_symbol],
                    )
                )
    except Exception as exc:
        logger.error("WikiPathways fetch failed for %s: %s", gene_symbol, exc)

    logger.info("WikiPathways: found %d pathways for %s", len(pathways), gene_symbol)
    return pathways


async def fetch_gene_pathways(gene_symbol: str) -> list[PathwayInfo]:
    """
    Aggregate pathways for a gene from Reactome and WikiPathways in parallel.

    Args:
        gene_symbol: HGNC gene symbol.

    Returns:
        Combined list of PathwayInfo objects from all sources.
    """
    import asyncio
    reactome_task = fetch_reactome_pathways(gene_symbol)
    wiki_task = fetch_wikipathways(gene_symbol)
    results = await asyncio.gather(reactome_task, wiki_task, return_exceptions=True)

    combined: list[PathwayInfo] = []
    for r in results:
        if isinstance(r, Exception):
            logger.error("Pathway fetch error: %s", r)
        else:
            combined.extend(r)  # type: ignore[arg-type]
    return combined


def get_pathway_candidate_genes(pathways: list[PathwayInfo]) -> list[str]:
    """
    Extract a deduplicated list of candidate biomarker genes from pathway data.

    Args:
        pathways: List of PathwayInfo objects.

    Returns:
        Deduplicated list of gene symbols found in pathways.
    """
    seen: set[str] = set()
    candidates: list[str] = []
    for pw in pathways:
        for gene in pw.genes:
            clean = gene.strip().upper()
            if clean and clean not in seen and len(clean) <= 15:
                seen.add(clean)
                candidates.append(gene.strip())
    return candidates
