"""
OncoMOA Tool — PubTator Central
Extracts structured biomedical entities (genes, mutations, diseases, drugs)
from PubMed abstracts using NLP annotations.

Example:
    annotations = await fetch_pubtator_annotations(["33836569", "34534463"])
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from config import PUBTATOR_BASE, HTTP_TIMEOUT
from models.schemas import PubMedArticle
from tools.cache import cached_api_call

logger = logging.getLogger(__name__)

PUBTATOR_ENTITY_TYPES = {"Gene", "Mutation", "Disease", "Chemical", "Species"}


@cached_api_call("pubtator_biocjson")
async def fetch_pubtator_annotations(pmids: list[str]) -> list[dict[str, Any]]:
    """
    Fetch PubTator3 entity annotations for a list of PMIDs via BioC JSON format.

    Args:
        pmids: List of PubMed IDs to annotate.

    Returns:
        List of raw annotation dicts (one per PMID).

    Example:
        annotations = await fetch_pubtator_annotations(["33836569"])
    """
    if not pmids:
        return []

    results: list[dict[str, Any]] = []
    # PubTator3 supports batch via publications endpoint
    url = f"{PUBTATOR_BASE}/publications/export/biocjson"

    # Process in chunks of 10 to avoid request size limits
    chunk_size = 10
    for i in range(0, len(pmids), chunk_size):
        chunk = pmids[i : i + chunk_size]
        params = {"pmids": ",".join(chunk), "full": "true"}
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
            ) as session:
                async with session.get(url, params=params) as resp:
                    if resp.status == 404:
                        logger.debug("PubTator: no annotations for PMIDs %s", chunk)
                        continue
                    resp.raise_for_status()
                    data = await resp.json(content_type=None)
                    if isinstance(data, list):
                        results.extend(data)
                    elif isinstance(data, dict):
                        results.append(data)
        except Exception as exc:
            logger.error("PubTator fetch failed for chunk %s: %s", chunk, exc)

    logger.info("PubTator: retrieved annotations for %d documents", len(results))
    return results


def parse_pubtator_annotations(
    raw_annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Parse raw PubTator BioC JSON into structured entity records.

    Returns:
        List of dicts with keys: pmid, entity_type, entity_text, entity_id, section.
    """
    entities: list[dict[str, Any]] = []

    for doc in raw_annotations:
        pmid = doc.get("id", "")
        passages = doc.get("passages", [])

        for passage in passages:
            section = passage.get("infons", {}).get("type", "")
            annotations = passage.get("annotations", [])

            for ann in annotations:
                infons = ann.get("infons", {})
                entity_type = infons.get("type", "")
                if entity_type not in PUBTATOR_ENTITY_TYPES:
                    continue

                entity_text = ann.get("text", "")
                entity_id = infons.get("identifier", "")

                entities.append({
                    "pmid": pmid,
                    "entity_type": entity_type,
                    "entity_text": entity_text,
                    "entity_id": entity_id,
                    "section": section,
                })

    return entities


def enrich_articles_with_pubtator(
    articles: list[PubMedArticle],
    annotations: list[dict[str, Any]],
) -> list[PubMedArticle]:
    """
    Enrich PubMedArticle objects with entity annotations from PubTator.

    Args:
        articles: List of PubMedArticle objects.
        annotations: Parsed entity dicts from parse_pubtator_annotations().

    Returns:
        Enriched PubMedArticle objects with genes/mutations/diseases/drugs populated.
    """
    # Build lookup: pmid → entities
    pmid_to_entities: dict[str, list[dict[str, Any]]] = {}
    for ent in annotations:
        pmid = ent.get("pmid", "")
        pmid_to_entities.setdefault(pmid, []).append(ent)

    for article in articles:
        ents = pmid_to_entities.get(article.pmid, [])
        for ent in ents:
            etype = ent.get("entity_type", "")
            text = ent.get("entity_text", "")
            if not text:
                continue
            if etype == "Gene" and text not in article.genes_mentioned:
                article.genes_mentioned.append(text)
            elif etype == "Mutation" and text not in article.mutations_mentioned:
                article.mutations_mentioned.append(text)
            elif etype == "Disease" and text not in article.diseases_mentioned:
                article.diseases_mentioned.append(text)
            elif etype == "Chemical" and text not in article.drugs_mentioned:
                article.drugs_mentioned.append(text)

    return articles


async def get_enriched_articles(
    articles: list[PubMedArticle],
) -> list[PubMedArticle]:
    """
    Convenience wrapper: fetch PubTator annotations and enrich articles in one call.

    Args:
        articles: PubMedArticle objects with PMIDs.

    Returns:
        Enriched articles with entity annotations filled in.
    """
    if not articles:
        return articles
    pmids = [a.pmid for a in articles if a.pmid]
    raw = await fetch_pubtator_annotations(pmids)
    parsed = parse_pubtator_annotations(raw)
    return enrich_articles_with_pubtator(articles, parsed)
