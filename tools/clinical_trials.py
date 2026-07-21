"""
OncoMOA Tool — ClinicalTrials.gov v2 API
Fetches oncology trials for a drug and extracts biomarker mentions,
stratification factors, and exploratory endpoints.

Example:
    trials = await fetch_trials("sotorasib", max_results=20)
"""

from __future__ import annotations

import logging
import re
from typing import Any

import aiohttp

from config import CLINICAL_TRIALS_BASE, CLINICAL_TRIALS_MAX_RESULTS, HTTP_TIMEOUT
from models.schemas import ClinicalTrialInfo, NormalizedEvidence, EvidenceType, EvidenceDirection
from tools.cache import cached_api_call

logger = logging.getLogger(__name__)

# Keywords that signal biomarker-related content in trial criteria text
BIOMARKER_KEYWORDS = [
    "biomarker", "mutation", "expression", "amplification", "deletion",
    "fusion", "rearrangement", "overexpression", "positive", "negative",
    "high", "low", "score", "status", "MSI", "TMB", "HER2", "KRAS",
    "BRAF", "EGFR", "ALK", "RET", "MET", "NTRK", "BRCA", "PD-L1",
    "PDL1", "CD274", "HRD", "CPS", "TPS", "IHC", "FISH", "NGS",
    "sequencing", "genotyping", "ctDNA", "cfDNA",
]

_BIOMARKER_RE = re.compile(
    r"\b(" + "|".join(re.escape(kw) for kw in BIOMARKER_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def _extract_biomarker_mentions(text: str) -> list[str]:
    """Extract potential biomarker keyword mentions from free text."""
    if not text:
        return []
    matches = _BIOMARKER_RE.findall(text)
    return list(dict.fromkeys(m.upper() for m in matches))  # deduplicated, order-preserving


def _parse_phase(phases: list[str] | str) -> str | None:
    """Normalize trial phase to a readable string."""
    if isinstance(phases, list):
        phases = " ".join(phases)
    phases = phases.upper()
    if "3" in phases or "III" in phases:
        return "Phase 3"
    if "2" in phases or "II" in phases:
        return "Phase 2"
    if "1" in phases or "I" in phases:
        return "Phase 1"
    if "4" in phases or "IV" in phases:
        return "Phase 4"
    return phases or None


@cached_api_call("clinical_trials_search")
async def fetch_trials(
    drug_name: str, max_results: int = CLINICAL_TRIALS_MAX_RESULTS
) -> list[ClinicalTrialInfo]:
    """
    Search ClinicalTrials.gov v2 API for trials involving a drug.

    Extracts: trial ID, phase, condition, biomarker mentions,
    inclusion/exclusion criteria, and exploratory endpoints.

    Args:
        drug_name: Drug name to search (e.g., "sotorasib").
        max_results: Max number of trials to retrieve.

    Returns:
        List of ClinicalTrialInfo objects.
    """
    url = f"{CLINICAL_TRIALS_BASE}/studies"
    params = {
        "query.intr": drug_name,
        "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING,COMPLETED",
        "pageSize": max_results,
        "format": "json",
        "fields": (
            "NCTId,BriefTitle,Phase,Condition,InterventionName,"
            "EligibilityCriteria,PrimaryOutcomeMeasure,SecondaryOutcomeMeasure,"
            "OverallStatus,BriefSummary"
        ),
    }

    trials: list[ClinicalTrialInfo] = []
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT)) as session:
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()

        studies = data.get("studies", [])
        for study in studies:
            proto = study.get("protocolSection", {})
            id_mod = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            design_mod = proto.get("designModule", {})
            eligibility_mod = proto.get("eligibilityModule", {})
            outcomes_mod = proto.get("outcomesModule", {})
            conditions_mod = proto.get("conditionsModule", {})
            interventions_mod = proto.get("armsInterventionsModule", {})

            nct_id = id_mod.get("nctId", "")
            title = id_mod.get("briefTitle", "")
            status = status_mod.get("overallStatus", "")

            phase_list = design_mod.get("phases", [])
            phase = _parse_phase(phase_list)

            condition = ", ".join(conditions_mod.get("conditions", [])[:3])
            eligibility_text = eligibility_mod.get("eligibilityCriteria", "")

            # Extract inclusion/exclusion from combined criteria text
            inclusion = ""
            exclusion = ""
            if "Exclusion" in eligibility_text:
                parts = eligibility_text.split("Exclusion")
                inclusion = parts[0].replace("Inclusion Criteria:", "").strip()
                exclusion = ("Exclusion" + parts[1]).strip()
            else:
                inclusion = eligibility_text.strip()

            # Biomarker mentions from criteria + outcomes
            primary_outcomes = [
                o.get("measure", "") for o in outcomes_mod.get("primaryOutcomes", [])
            ]
            secondary_outcomes = [
                o.get("measure", "") for o in outcomes_mod.get("secondaryOutcomes", [])
            ]
            all_text = " ".join([eligibility_text] + primary_outcomes + secondary_outcomes)
            biomarker_mentions = _extract_biomarker_mentions(all_text)

            # Exploratory endpoints from secondary outcomes
            exploratory = [
                o for o in secondary_outcomes
                if any(kw in o.lower() for kw in ["biomarker", "exploratory", "correlative", "translational"])
            ]

            # Interventions: look for drug match
            drug_matched = drug_name.lower()
            for arm in interventions_mod.get("interventions", []):
                arm_name = arm.get("name", "").lower()
                if drug_matched in arm_name:
                    drug_matched = arm.get("name", drug_name)
                    break

            trials.append(
                ClinicalTrialInfo(
                    trial_id=nct_id,
                    title=title,
                    phase=phase,
                    condition=condition,
                    drug=drug_name,
                    status=status,
                    biomarker_mentions=biomarker_mentions,
                    inclusion_criteria=inclusion[:2000] if inclusion else None,
                    exclusion_criteria=exclusion[:2000] if exclusion else None,
                    exploratory_endpoints=exploratory[:10],
                    stratification_factors=[],
                )
            )

        logger.info(
            "ClinicalTrials: fetched %d trials for '%s'", len(trials), drug_name
        )
    except Exception as exc:
        logger.error("ClinicalTrials fetch_trials failed for '%s': %s", drug_name, exc)

    return trials


def trials_to_evidence(
    trials: list[ClinicalTrialInfo],
    target_genes: list[str],
) -> list[NormalizedEvidence]:
    """
    Convert ClinicalTrialInfo objects to NormalizedEvidence items.

    Links trials to genes based on biomarker mention overlap.
    """
    evidence_items: list[NormalizedEvidence] = []
    gene_set = {g.upper() for g in target_genes}

    # Phase → strength mapping
    phase_strength = {"Phase 3": 3.0, "Phase 2": 2.0, "Phase 1": 1.0, "Phase 4": 3.5}

    for trial in trials:
        strength = phase_strength.get(trial.phase or "", 1.0)

        # Find genes mentioned in the trial biomarkers
        matched_genes = [
            g for g in target_genes
            if g.upper() in [m.upper() for m in trial.biomarker_mentions]
        ]
        if not matched_genes:
            matched_genes = list(target_genes[:1])  # associate with primary target

        for gene in matched_genes:
            evidence_items.append(
                NormalizedEvidence(
                    source="ClinicalTrials",
                    source_id=trial.trial_id,
                    gene=gene,
                    disease=trial.condition,
                    drug=trial.drug,
                    evidence_type=EvidenceType.PREDICTIVE,
                    evidence_direction=EvidenceDirection.SUPPORTS,
                    claim=(
                        f"{trial.title} ({trial.phase}, {trial.status}) "
                        f"includes biomarker mentions: {', '.join(trial.biomarker_mentions[:5])}"
                    ),
                    strength=strength,
                    raw_data=trial.model_dump(),
                )
            )

    return evidence_items
