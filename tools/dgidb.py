"""
OncoMOA Tool — DGIdb (Drug-Gene Interaction Database)
Fetches drug-gene interaction evidence to generate indirect biomarker hypotheses.

Example:
    interactions = await fetch_dgidb_interactions(["KRAS", "STK11"])
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from config import DGIDB_BASE, HTTP_TIMEOUT
from models.schemas import NormalizedEvidence, EvidenceType, EvidenceDirection
from tools.cache import cached_api_call

logger = logging.getLogger(__name__)

# GraphQL query for drug-gene interactions
DGIDB_QUERY = """
query GetInteractions($genes: [String!]!) {
  genes(names: $genes) {
    nodes {
      name
      interactions {
        drug {
          name
          approved
        }
        interactionScore
        interactionTypes {
          type
          directionality
        }
        publications {
          pmid
        }
        sources {
          sourceDbName
        }
      }
    }
  }
}
"""


@cached_api_call("dgidb_interactions")
async def fetch_dgidb_interactions(
    gene_symbols: list[str],
) -> list[NormalizedEvidence]:
    """
    Fetch drug-gene interactions from DGIdb for candidate genes.

    Used to generate indirect biomarker hypotheses: if gene X interacts with
    drug Y which shares a pathway with the primary target, gene X is a candidate.

    Args:
        gene_symbols: List of HGNC gene symbols.

    Returns:
        List of NormalizedEvidence objects representing drug-gene interactions.
    """
    if not gene_symbols:
        return []

    evidence_items: list[NormalizedEvidence] = []
    payload = {"query": DGIDB_QUERY, "variables": {"genes": gene_symbols}}

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)) as session:
            async with session.post(
                DGIDB_BASE,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

        nodes = data.get("data", {}).get("genes", {}).get("nodes", [])
        for node in nodes:
            gene = node.get("name", "")
            for interaction in node.get("interactions", []):
                drug_name = interaction.get("drug", {}).get("name", "")
                score = interaction.get("interactionScore", 0.0) or 0.0
                itypes = [
                    it.get("type", "") for it in interaction.get("interactionTypes", [])
                ]
                pmids = [str(p.get("pmid", "")) for p in interaction.get("publications", [])]
                sources = [s.get("sourceDbName", "") for s in interaction.get("sources", [])]

                if not drug_name or score < 1.0:
                    continue

                claim = (
                    f"{gene} interacts with {drug_name} "
                    f"({', '.join(itypes) if itypes else 'interaction'}) "
                    f"[Sources: {', '.join(sources[:3])}]"
                )

                evidence_items.append(
                    NormalizedEvidence(
                        source="DGIdb",
                        source_id=f"DGIDB_{gene}_{drug_name[:20]}",
                        gene=gene,
                        drug=drug_name,
                        evidence_type=EvidenceType.PREDICTIVE,
                        evidence_direction=EvidenceDirection.SUPPORTS,
                        claim=claim,
                        strength=min(score / 5.0, 3.0),
                        raw_data=interaction,
                    )
                )

        logger.info(
            "DGIdb: found %d interactions for %d genes",
            len(evidence_items),
            len(gene_symbols),
        )
    except Exception as exc:
        logger.error("DGIdb fetch_dgidb_interactions failed: %s", exc)

    return evidence_items
