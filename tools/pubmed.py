"""
OncoMOA Tool — PubMed E-utilities
Search and fetch biomedical literature for drug-gene-cancer queries.
Rate-limited to stay within NCBI's free-tier limits (3 req/sec).

Example:
    articles = await search_pubmed_for_gene("sotorasib", "KRAS")
"""

from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET
from typing import Any

import aiohttp

from config import PUBMED_EUTILS_BASE, PUBMED_MAX_RESULTS, NCBI_API_KEY, HTTP_TIMEOUT
from models.schemas import PubMedArticle
from tools.cache import cached_api_call

logger = logging.getLogger(__name__)

# Throttle: max 3 concurrent PubMed requests (free tier = 3 req/sec)
# With an API key this can be raised to 10
_PUBMED_SEMAPHORE = asyncio.Semaphore(3)
_PUBMED_DELAY = 0.4  # seconds between requests within a batch


def _build_params(extra: dict[str, Any]) -> dict[str, str]:
    """Build base NCBI params dict, appending API key if available."""
    params: dict[str, Any] = {"retmode": "json", **extra}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    return {k: str(v) for k, v in params.items()}


@cached_api_call("pubmed_search_v2")
async def search_pubmed_ids(query: str, max_results: int = PUBMED_MAX_RESULTS) -> list[str]:
    """
    Search PubMed using esearch and return a list of PMIDs.
    Rate-limited via semaphore to avoid 429 errors.

    Args:
        query: Free-text search query.
        max_results: Maximum number of PMIDs to return.

    Returns:
        List of PMID strings.
    """
    url = f"{PUBMED_EUTILS_BASE}/esearch.fcgi"
    params = _build_params({
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "usehistory": "y",
        "sort": "relevance",
    })
    async with _PUBMED_SEMAPHORE:
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
            ) as session:
                async with session.get(url, params=params) as resp:
                    if resp.status == 429:
                        logger.warning("PubMed rate limit hit — backing off 2s")
                        await asyncio.sleep(2.0)
                        async with session.get(url, params=params) as retry_resp:
                            retry_resp.raise_for_status()
                            data = await retry_resp.json()
                    else:
                        resp.raise_for_status()
                        data = await resp.json()
            await asyncio.sleep(_PUBMED_DELAY)
            pmids = data.get("esearchresult", {}).get("idlist", [])
            logger.info("PubMed esearch: %d PMIDs for query '%s'", len(pmids), query[:60])
            return pmids
        except Exception as exc:
            logger.error("PubMed search_pubmed_ids failed: %s", exc)
            return []


@cached_api_call("pubmed_fetch_v2")
async def fetch_pubmed_abstracts(pmids: tuple[str, ...]) -> list[PubMedArticle]:
    """
    Fetch titles and abstracts for a list of PMIDs using efetch.
    Accepts a tuple for cache-key stability.

    Args:
        pmids: Tuple of PubMed IDs.

    Returns:
        List of PubMedArticle objects.
    """
    if not pmids:
        return []

    url = f"{PUBMED_EUTILS_BASE}/efetch.fcgi"
    params = _build_params({
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    })

    async with _PUBMED_SEMAPHORE:
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
            ) as session:
                async with session.get(url, params=params) as resp:
                    if resp.status == 429:
                        logger.warning("PubMed rate limit on efetch — backing off 3s")
                        await asyncio.sleep(3.0)
                        async with session.get(url, params=params) as retry_resp:
                            retry_resp.raise_for_status()
                            xml_text = await retry_resp.text()
                    else:
                        resp.raise_for_status()
                        xml_text = await resp.text()
            await asyncio.sleep(_PUBMED_DELAY)
            articles = _parse_pubmed_xml(xml_text)
            logger.info("PubMed efetch: parsed %d articles", len(articles))
            return articles
        except Exception as exc:
            logger.error("PubMed fetch_pubmed_abstracts failed: %s", exc)
            return []


def _parse_pubmed_xml(xml_text: str) -> list[PubMedArticle]:
    """Parse PubMed XML response into PubMedArticle objects."""
    articles: list[PubMedArticle] = []
    try:
        root = ET.fromstring(xml_text)
        for article_elem in root.findall(".//PubmedArticle"):
            pmid_elem = article_elem.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else ""

            title_elem = article_elem.find(".//ArticleTitle")
            title = "".join(title_elem.itertext()) if title_elem is not None else ""

            abstract_parts = []
            for ab_text in article_elem.findall(".//AbstractText"):
                label = ab_text.get("Label", "")
                text = "".join(ab_text.itertext())
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            abstract = " ".join(abstract_parts)

            if pmid:
                articles.append(PubMedArticle(pmid=pmid, title=title, abstract=abstract))
    except ET.ParseError as exc:
        logger.error("PubMed XML parse error: %s", exc)
    return articles


async def search_pubmed_for_gene(
    drug_name: str,
    gene_symbol: str,
    max_results: int = PUBMED_MAX_RESULTS,
) -> list[PubMedArticle]:
    """
    Search PubMed for a drug-gene-cancer biomarker query and fetch abstracts.

    Args:
        drug_name: Drug name (e.g., "sotorasib").
        gene_symbol: Gene symbol (e.g., "KRAS").
        max_results: Max results to retrieve.

    Returns:
        List of PubMedArticle objects.
    """
    query = f'"{drug_name}" biomarker "{gene_symbol}" cancer'
    pmids = await search_pubmed_ids(query, max_results=max_results)
    if not pmids:
        query = f"{drug_name} {gene_symbol} cancer biomarker"
        pmids = await search_pubmed_ids(query, max_results=max_results)
    if not pmids:
        return []
    articles = await fetch_pubmed_abstracts(tuple(pmids))
    return articles


async def search_pubmed_bulk(
    drug_name: str,
    gene_symbols: list[str],
    max_per_gene: int = 5,
) -> dict[str, list[PubMedArticle]]:
    """
    Search PubMed for multiple genes sequentially (not parallel) to respect rate limits.

    Returns dict mapping gene_symbol → list of articles.
    """
    output: dict[str, list[PubMedArticle]] = {}
    for gene in gene_symbols:
        try:
            articles = await search_pubmed_for_gene(
                drug_name, gene, max_results=max_per_gene
            )
            output[gene] = articles
            # Small delay between genes to stay under rate limit
            await asyncio.sleep(0.35)
        except Exception as exc:
            logger.error("PubMed bulk search error for %s: %s", gene, exc)
            output[gene] = []
    return output
