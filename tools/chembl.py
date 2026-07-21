"""
OncoMOA Tool — ChEMBL API
Fetches drug targets, bioactivity data, and mechanism annotations.

Example:
    targets = await fetch_chembl_drug_targets("sotorasib")
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from config import CHEMBL_BASE, HTTP_TIMEOUT
from tools.cache import cached_api_call

logger = logging.getLogger(__name__)


@cached_api_call("chembl_drug_search")
async def search_chembl_drug(drug_name: str) -> dict[str, Any]:
    """
    Search ChEMBL for a drug by name and return the top matching molecule.

    Args:
        drug_name: Drug name to search (e.g., "sotorasib").

    Returns:
        ChEMBL molecule dict with chembl_id, preferred_name, etc.
    """
    url = f"{CHEMBL_BASE}/molecule.json"
    params = {
        "pref_name__iexact": drug_name,
        "format": "json",
        "limit": 1,
    }

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)) as session:
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()

        molecules = data.get("molecules", [])
        if not molecules:
            # Try fuzzy search
            params2 = {"pref_name__icontains": drug_name, "format": "json", "limit": 1}
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)) as session:
                async with session.get(url, params=params2) as resp:
                    resp.raise_for_status()
                    data2 = await resp.json()
            molecules = data2.get("molecules", [])

        if molecules:
            mol = molecules[0]
            logger.info("ChEMBL: found %s (id=%s)", drug_name, mol.get("molecule_chembl_id"))
            return mol
        return {}
    except Exception as exc:
        logger.error("ChEMBL search_chembl_drug failed for %s: %s", drug_name, exc)
        return {}


@cached_api_call("chembl_drug_targets")
async def fetch_chembl_drug_targets(drug_name: str) -> list[dict[str, Any]]:
    """
    Fetch target proteins for a drug from ChEMBL via mechanism of action data.

    Args:
        drug_name: Drug name (e.g., "sotorasib").

    Returns:
        List of target dicts with target_chembl_id, pref_name, target_type, gene_names.
    """
    mol = await search_chembl_drug(drug_name)
    chembl_id = mol.get("molecule_chembl_id", "")
    if not chembl_id:
        return []

    url = f"{CHEMBL_BASE}/mechanism.json"
    params = {
        "molecule_chembl_id": chembl_id,
        "format": "json",
        "limit": 50,
    }

    targets: list[dict[str, Any]] = []
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)) as session:
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()

        mechanisms = data.get("mechanisms", [])
        for mech in mechanisms:
            target_chembl_id = mech.get("target_chembl_id", "")
            if not target_chembl_id:
                continue

            # Fetch target details
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)) as session:
                async with session.get(
                    f"{CHEMBL_BASE}/target/{target_chembl_id}.json"
                ) as tresp:
                    if tresp.status != 200:
                        continue
                    target_data = await tresp.json()

            gene_names = []
            for comp in target_data.get("target_components", []):
                for syn in comp.get("target_component_synonyms", []):
                    if syn.get("syn_type") == "GENE_SYMBOL":
                        gene_names.append(syn.get("component_synonym", ""))

            targets.append({
                "target_chembl_id": target_chembl_id,
                "pref_name": target_data.get("pref_name", ""),
                "target_type": target_data.get("target_type", ""),
                "gene_names": gene_names,
                "mechanism": mech.get("mechanism_of_action", ""),
                "action_type": mech.get("action_type", ""),
            })

        logger.info(
            "ChEMBL: found %d targets for %s", len(targets), drug_name
        )
    except Exception as exc:
        logger.error("ChEMBL fetch_chembl_drug_targets failed for %s: %s", drug_name, exc)

    return targets


def extract_gene_symbols_from_chembl(targets: list[dict[str, Any]]) -> list[str]:
    """Extract HGNC gene symbols from ChEMBL target records."""
    genes: list[str] = []
    for target in targets:
        for gene in target.get("gene_names", []):
            if gene and gene not in genes:
                genes.append(gene)
    return genes
