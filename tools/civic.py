"""
OncoMOA Tool — CIViC (Clinical Interpretation of Variants in Cancer)
Uses the CIViC GraphQL API to fetch predictive and prognostic biomarker evidence.
Falls back to REST v2 evidence item search if GraphQL returns no results.

Example:
    evidence = await fetch_civic_evidence(["KRAS", "STK11"], drug_name="sotorasib")
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from config import HTTP_TIMEOUT
from models.schemas import NormalizedEvidence, EvidenceType, EvidenceDirection
from tools.cache import cached_api_call

logger = logging.getLogger(__name__)

CIVIC_GRAPHQL_URL = "https://civicdb.org/api/graphql"
CIVIC_REST_BASE = "https://civicdb.org/api"
CIVIC_EVIDENCE_TYPES = {"PREDICTIVE", "PROGNOSTIC"}

# Gene query using GraphQL — searches by gene symbol
CIVIC_GENE_QUERY = """
query GeneEvidence($symbol: String!) {
  gene(entrezSymbol: $symbol) {
    id
    name
    variants {
      id
      name
      evidenceItems {
        id
        status
        evidenceType
        evidenceLevel
        evidenceDirection
        significance
        disease { name }
        therapies { name }
        source { citation pmid }
      }
    }
  }
}
"""

# Fallback: direct evidence item search via REST
# GET /api/evidence_items?gene_id=...&evidence_type=Predictive&status=accepted
CIVIC_EVIDENCE_URL = f"{CIVIC_REST_BASE}/evidence_items"


def _parse_evidence_type(raw: str) -> EvidenceType:
    mapping = {
        "PREDICTIVE": EvidenceType.PREDICTIVE,
        "PROGNOSTIC": EvidenceType.PROGNOSTIC,
        "DIAGNOSTIC": EvidenceType.DIAGNOSTIC,
        "PREDISPOSING": EvidenceType.PREDISPOSING,
        "ONCOGENIC": EvidenceType.ONCOGENIC,
        "FUNCTIONAL": EvidenceType.FUNCTIONAL,
        "Predictive": EvidenceType.PREDICTIVE,
        "Prognostic": EvidenceType.PROGNOSTIC,
    }
    return mapping.get(raw, EvidenceType.UNKNOWN)


def _parse_direction(raw: str) -> EvidenceDirection:
    raw_up = raw.upper()
    if any(k in raw_up for k in ("DOES_NOT", "NEGATIVE", "WILD_TYPE")):
        return EvidenceDirection.DOES_NOT_SUPPORT
    if any(k in raw_up for k in ("SUPPORTS", "POSITIVE")):
        return EvidenceDirection.SUPPORTS
    return EvidenceDirection.UNKNOWN


def _level_strength(level: str) -> float:
    return {"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0, "E": 1.0}.get(level.upper(), 0.5)


@cached_api_call("civic_graphql_v3")
async def _graphql_fetch(gene_symbol: str) -> list[NormalizedEvidence]:
    """Try CIViC GraphQL using entrezSymbol field."""
    evidence_items: list[NormalizedEvidence] = []
    payload = {"query": CIVIC_GENE_QUERY, "variables": {"symbol": gene_symbol}}

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
        ) as session:
            async with session.post(
                CIVIC_GRAPHQL_URL,
                json=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            ) as resp:
                if resp.status not in (200, 201):
                    logger.debug("CIViC GraphQL %d for %s", resp.status, gene_symbol)
                    return []
                data = await resp.json()

        errors = data.get("errors")
        if errors:
            logger.debug("CIViC GraphQL errors for %s: %s", gene_symbol, errors)

        gene = (data.get("data") or {}).get("gene")
        if not gene:
            return []

        gene_name = gene.get("name", gene_symbol)
        for variant in gene.get("variants", []):
            variant_name = variant.get("name", "")
            for ev in variant.get("evidenceItems", []):
                ev_type = ev.get("evidenceType", "")
                if ev_type.upper() not in CIVIC_EVIDENCE_TYPES:
                    continue
                if ev.get("status", "").upper() not in ("ACCEPTED", "SUBMITTED"):
                    continue

                ev_level = (ev.get("evidenceLevel") or "E").upper()
                disease = (ev.get("disease") or {}).get("name", "")
                therapies = [t.get("name", "") for t in (ev.get("therapies") or [])]
                therapy_str = ", ".join(therapies) if therapies else "unknown therapy"
                ev_id = str(ev.get("id", ""))

                evidence_items.append(NormalizedEvidence(
                    source="CIViC",
                    source_id=f"CIViC_{ev_id}",
                    gene=gene_name,
                    variant=variant_name,
                    disease=disease,
                    drug=therapy_str if therapies else None,
                    evidence_type=_parse_evidence_type(ev_type),
                    evidence_direction=_parse_direction(ev.get("evidenceDirection", "")),
                    evidence_level=ev_level,
                    claim=(
                        f"{gene_name} {variant_name} {ev_type.lower()} "
                        f"for {therapy_str} in {disease} (Level {ev_level})"
                    ),
                    strength=_level_strength(ev_level),
                    raw_data=ev,
                ))
    except Exception as exc:
        logger.debug("CIViC GraphQL failed for %s: %s", gene_symbol, exc)

    return evidence_items


@cached_api_call("civic_rest_evidence_v2")
async def _rest_fetch(gene_symbol: str) -> list[NormalizedEvidence]:
    """
    Fallback: query CIViC REST v1 evidence_items endpoint directly.
    Searches by gene_entrez_symbol query parameter.
    """
    evidence_items: list[NormalizedEvidence] = []
    params = {
        "gene_entrez_symbol": gene_symbol,
        "count": 25,
        "status": "accepted",
    }

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
        ) as session:
            async with session.get(
                CIVIC_EVIDENCE_URL,
                params=params,
                headers={"Accept": "application/json"},
            ) as resp:
                if resp.status != 200:
                    logger.debug("CIViC REST evidence_items %d for %s", resp.status, gene_symbol)
                    return []
                data = await resp.json()

        records = data.get("records", [])
        if not records and isinstance(data, list):
            records = data

        for ev in records:
            ev_type = ev.get("evidence_type", "")
            if ev_type not in ("Predictive", "Prognostic"):
                continue

            ev_level = (ev.get("evidence_level") or "E").upper()
            gene_name = ev.get("gene", {}).get("name", gene_symbol) if ev.get("gene") else gene_symbol
            variant_name = ev.get("variant", {}).get("name", "") if ev.get("variant") else ""
            disease = ev.get("disease", {}).get("name", "") if ev.get("disease") else ""
            drugs = [d.get("name", "") for d in (ev.get("drugs") or [])]
            therapy_str = ", ".join(drugs) if drugs else "unknown therapy"
            ev_id = str(ev.get("id", ""))

            evidence_items.append(NormalizedEvidence(
                source="CIViC",
                source_id=f"CIViC_{ev_id}",
                gene=gene_name,
                variant=variant_name,
                disease=disease,
                drug=therapy_str if drugs else None,
                evidence_type=_parse_evidence_type(ev_type),
                evidence_direction=_parse_direction(ev.get("evidence_direction", "")),
                evidence_level=ev_level,
                claim=(
                    f"{gene_name} {variant_name} {ev_type.lower()} "
                    f"for {therapy_str} in {disease} (Level {ev_level})"
                ),
                strength=_level_strength(ev_level),
                raw_data=ev,
            ))
    except Exception as exc:
        logger.debug("CIViC REST fallback failed for %s: %s", gene_symbol, exc)

    return evidence_items


@cached_api_call("civic_evidence_combined")
async def fetch_civic_evidence(
    gene_symbols: list[str],
    drug_name: str = "",
) -> list[NormalizedEvidence]:
    """
    Fetch CIViC Predictive and Prognostic evidence for a list of gene symbols.
    Tries GraphQL first, then falls back to REST evidence_items endpoint.

    Args:
        gene_symbols: List of HGNC gene symbols.
        drug_name: Optional drug name for context.

    Returns:
        List of NormalizedEvidence items with CIViC-level strength scores.
    """
    all_evidence: list[NormalizedEvidence] = []

    for gene in gene_symbols:
        # Try GraphQL first
        items = await _graphql_fetch(gene)

        # Fallback to REST if GraphQL returns nothing
        if not items:
            logger.debug("CIViC GraphQL empty for %s — trying REST fallback", gene)
            items = await _rest_fetch(gene)

        all_evidence.extend(items)
        if items:
            logger.info("CIViC: %d items for %s", len(items), gene)

    logger.info(
        "CIViC: retrieved %d total evidence items for %d genes",
        len(all_evidence),
        len(gene_symbols),
    )
    return all_evidence


async def fetch_civic_summary_stats(
    evidence_items: list[NormalizedEvidence],
) -> dict[str, Any]:
    """Compute summary statistics from CIViC evidence items."""
    level_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    gene_counts: dict[str, int] = {}

    for item in evidence_items:
        if item.source != "CIViC":
            continue
        level = item.evidence_level or "E"
        level_counts[level] = level_counts.get(level, 0) + 1
        etype = item.evidence_type.value
        type_counts[etype] = type_counts.get(etype, 0) + 1
        if item.gene:
            gene_counts[item.gene] = gene_counts.get(item.gene, 0) + 1

    return {
        "total": len([e for e in evidence_items if e.source == "CIViC"]),
        "by_level": level_counts,
        "by_type": type_counts,
        "by_gene": gene_counts,
    }
