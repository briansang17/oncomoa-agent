"""
OncoMOA Agent — Clinical Trial Agent
Fetches clinical trials for the drug and extracts biomarker mentions,
stratification factors, and exploratory endpoints as structured evidence.

Example:
    agent = TrialAgent()
    evidence, trials = await agent.run("sotorasib", ["KRAS", "STK11"])
"""

from __future__ import annotations

import logging

from models.schemas import NormalizedEvidence, ClinicalTrialInfo
from tools.clinical_trials import fetch_trials, trials_to_evidence

logger = logging.getLogger(__name__)


class TrialAgent:
    """
    Retrieves clinical trial data and converts it to structured biomarker evidence.
    """

    async def run(
        self,
        drug_name: str,
        target_genes: list[str],
        max_results: int = 20,
    ) -> tuple[list[NormalizedEvidence], list[ClinicalTrialInfo]]:
        """
        Fetch trials for the drug and generate NormalizedEvidence items.

        Args:
            drug_name: Drug name to search (e.g., "sotorasib").
            target_genes: Target/candidate genes to match against trial biomarker mentions.
            max_results: Max number of trials to retrieve.

        Returns:
            Tuple of (NormalizedEvidence list, ClinicalTrialInfo list).
        """
        logger.info("[TrialAgent] Fetching trials for drug: %s", drug_name)

        trials = await fetch_trials(drug_name, max_results=max_results)

        if not trials:
            logger.warning("[TrialAgent] No trials found for %s", drug_name)
            return [], []

        evidence_items = trials_to_evidence(trials, target_genes)

        # Log phase distribution
        phase_counts: dict[str, int] = {}
        for trial in trials:
            phase = trial.phase or "Unknown"
            phase_counts[phase] = phase_counts.get(phase, 0) + 1

        logger.info(
            "[TrialAgent] Found %d trials, %d evidence items. Phase distribution: %s",
            len(trials),
            len(evidence_items),
            phase_counts,
        )

        # Log top biomarker mentions across all trials
        all_biomarker_mentions: dict[str, int] = {}
        for trial in trials:
            for bm in trial.biomarker_mentions:
                all_biomarker_mentions[bm] = all_biomarker_mentions.get(bm, 0) + 1
        top_mentions = sorted(all_biomarker_mentions.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("[TrialAgent] Top biomarker mentions in trials: %s", top_mentions)

        return evidence_items, trials
