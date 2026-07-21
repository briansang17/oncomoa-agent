"""
OncoMOA Tool — Open Targets Platform
GraphQL queries for drug-target associations, disease links, and MOA data.

Example:
    targets = await fetch_drug_targets("sotorasib")
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from config import OPEN_TARGETS_GRAPHQL_URL, HTTP_TIMEOUT
from models.schemas import NormalizedEvidence, EvidenceType, EvidenceDirection
from tools.cache import cached_api_call

logger = logging.getLogger(__name__)

# Step 1: Search for the drug by name → get ChEMBL ID
DRUG_SEARCH_QUERY = """
query DrugSearch($name: String!) {
  search(queryString: $name, entityNames: ["drug"], page: {index: 0, size: 5}) {
    total
    hits {
      id
      entity
      name
      description
    }
  }
}
"""

# Step 2: Fetch full drug info by ChEMBL ID
DRUG_DETAIL_QUERY = """
query DrugDetail($chemblId: String!) {
  drug(chemblId: $chemblId) {
    id
    name
    maximumClinicalTrialPhase
    mechanismsOfAction {
      uniqueActionTypes
      rows {
        mechanismOfAction
        actionType
        targets {
          id
          approvedSymbol
          approvedName
        }
      }
    }
    linkedTargets {
      count
      rows {
        id
        approvedSymbol
        approvedName
      }
    }
    linkedDiseases {
      count
      rows {
        id
        name
      }
    }
  }
}
"""

# Target disease associations
TARGET_DISEASE_QUERY = """
query TargetDiseases($targetId: String!) {
  target(ensemblId: $targetId) {
    id
    approvedSymbol
    approvedName
    associatedDiseases(page: {index: 0, size: 20}) {
      rows {
        disease { id name }
        score
      }
    }
  }
}
"""

TARGET_SEARCH_QUERY = """
query SearchTarget($symbol: String!) {
  search(queryString: $symbol, entityNames: ["target"], page: {index: 0, size: 3}) {
    hits { id name entity }
  }
}
"""


async def _post_graphql(session: aiohttp.ClientSession, query: str, variables: dict) -> dict:
    """Execute a GraphQL query and return the data field."""
    async with session.post(
        OPEN_TARGETS_GRAPHQL_URL,
        json={"query": query, "variables": variables},
        headers={"Content-Type": "application/json"},
    ) as resp:
        resp.raise_for_status()
        result = await resp.json()
    errors = result.get("errors")
    if errors:
        logger.warning("Open Targets GraphQL errors: %s", errors)
    return result.get("data", {})


@cached_api_call("open_targets_drug_v2")
async def fetch_drug_targets(drug_name: str) -> dict[str, Any]:
    """
    Query Open Targets for drug targets, mechanisms of action, and linked diseases.

    Returns a dict with keys: target_genes, mechanisms, linked_diseases.
    """
    result: dict[str, Any] = {
        "target_genes": [],
        "mechanisms": [],
        "linked_diseases": [],
    }

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
        ) as session:
            # Step 1: search for drug to get ChEMBL ID
            search_data = await _post_graphql(
                session, DRUG_SEARCH_QUERY, {"name": drug_name}
            )
            hits = search_data.get("search", {}).get("hits", [])
            drug_hits = [h for h in hits if h.get("entity") == "drug"]

            if not drug_hits:
                logger.warning("Open Targets: no drug hits for '%s'", drug_name)
                return result

            chembl_id = drug_hits[0]["id"]
            logger.debug("Open Targets: found drug %s (ChEMBL ID: %s)", drug_name, chembl_id)

            # Step 2: fetch full drug detail
            detail_data = await _post_graphql(
                session, DRUG_DETAIL_QUERY, {"chemblId": chembl_id}
            )
            drug = detail_data.get("drug", {})
            if not drug:
                return result

            # Mechanisms of action → target genes
            for row in drug.get("mechanismsOfAction", {}).get("rows", []):
                moa = row.get("mechanismOfAction", "")
                if moa:
                    result["mechanisms"].append(moa)
                for tgt in row.get("targets", []):
                    symbol = tgt.get("approvedSymbol", "")
                    if symbol and symbol not in result["target_genes"]:
                        result["target_genes"].append(symbol)

            # Linked targets (broader coverage)
            for row in drug.get("linkedTargets", {}).get("rows", []):
                symbol = row.get("approvedSymbol", "")
                if symbol and symbol not in result["target_genes"]:
                    result["target_genes"].append(symbol)

            # Linked diseases
            for row in drug.get("linkedDiseases", {}).get("rows", []):
                name = row.get("name", "")
                if name:
                    result["linked_diseases"].append(name)

    except Exception as exc:
        logger.error("Open Targets fetch_drug_targets failed for '%s': %s", drug_name, exc)

    logger.info(
        "Open Targets: %d target genes, %d diseases for '%s'",
        len(result["target_genes"]),
        len(result["linked_diseases"]),
        drug_name,
    )
    return result


@cached_api_call("open_targets_target_disease_v2")
async def fetch_target_disease_associations(
    target_symbol: str,
) -> list[NormalizedEvidence]:
    """
    Fetch disease associations for a gene target from Open Targets.

    Args:
        target_symbol: HGNC gene symbol.

    Returns:
        List of NormalizedEvidence objects for each disease association.
    """
    evidence_items: list[NormalizedEvidence] = []
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
        ) as session:
            # Resolve symbol → Ensembl ID
            search_data = await _post_graphql(
                session, TARGET_SEARCH_QUERY, {"symbol": target_symbol}
            )
            hits = search_data.get("search", {}).get("hits", [])
            if not hits:
                return []
            target_id = hits[0]["id"]

            # Fetch disease associations
            assoc_data = await _post_graphql(
                session, TARGET_DISEASE_QUERY, {"targetId": target_id}
            )

        rows = (
            assoc_data.get("target", {})
            .get("associatedDiseases", {})
            .get("rows", [])
        )

        for row in rows:
            disease_name = row.get("disease", {}).get("name", "")
            score = float(row.get("score", 0.0))
            if disease_name and score > 0.1:
                evidence_items.append(
                    NormalizedEvidence(
                        source="OpenTargets",
                        source_id=f"OT_{target_symbol}_{disease_name[:20]}",
                        gene=target_symbol,
                        disease=disease_name,
                        evidence_type=EvidenceType.ONCOGENIC,
                        evidence_direction=EvidenceDirection.SUPPORTS,
                        claim=(
                            f"{target_symbol} associated with {disease_name} "
                            f"(Open Targets score: {score:.3f})"
                        ),
                        strength=score * 2,
                        raw_data=row,
                    )
                )

        logger.info(
            "Open Targets: %d disease associations for %s",
            len(evidence_items),
            target_symbol,
        )
    except Exception as exc:
        logger.error(
            "Open Targets fetch_target_disease_associations failed for %s: %s",
            target_symbol,
            exc,
        )

    return evidence_items
