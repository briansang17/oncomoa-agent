"""
OncoMOA Agent — Biomarker Synthesis Agent (LLM Step)
Takes the pre-ranked BiomarkerHypothesis list and evidence context,
invokes the LLM to generate biological narrative and refine hypotheses,
then validates output with Pydantic and retries once on failure.

Example:
    agent = BiomarkerSynthesisAgent(backend)
    hypotheses = await agent.run(drug_name, moa, ranked_hypotheses, all_evidence)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import ValidationError

from llm.backend import (
    LLMBackend, ONCOLOGY_SYSTEM_PROMPT, build_evidence_prompt, build_narrative_prompt
)
from models.schemas import BiomarkerHypothesis, SupportingEvidence, RankingRationale

logger = logging.getLogger(__name__)

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    """
    Extract and parse the first JSON array from LLM output.
    Handles: markdown fences, <thinking> blocks, trailing commas, truncation.
    """
    # Strip thinking / reasoning tags (Gemini 3.x models)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    text = text.replace("```", "")

    match = _JSON_ARRAY_RE.search(text)
    if not match:
        raise ValueError("No JSON array found in LLM response")

    raw = match.group(0)

    # Try parsing directly
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Fix trailing commas before ] or } (common model mistake)
    cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Truncated JSON — try to recover complete objects
    recovered = []
    depth = 0
    start = None
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(cleaned[start : i + 1])
                    recovered.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None

    if recovered:
        return recovered

    raise ValueError("Could not parse JSON array from LLM response")


def _extract_narratives(text: str, biomarkers: list[str]) -> dict[str, str]:
    """
    Parse a numbered plain-text narrative response from a local model.

    Expects lines like:  "1. PDCD1 is the direct target..."
    Returns a dict mapping biomarker name → narrative sentence.
    """
    narratives: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        # Match lines starting with a digit and a dot/paren
        m = re.match(r"^(\d+)[.)]\s*(.+)", line)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        sentence = m.group(2).strip()
        if 0 <= idx < len(biomarkers):
            narratives[biomarkers[idx]] = sentence
    return narratives


def _build_evidence_context(
    all_evidence: list[Any],
    target_genes: list[str],
    max_items: int = 80,
) -> str:
    """
    Build a compact evidence context string from normalized evidence.
    Groups by source type for LLM readability.
    """
    from models.schemas import NormalizedEvidence

    by_source: dict[str, list[NormalizedEvidence]] = {}
    for ev in all_evidence:
        if not isinstance(ev, NormalizedEvidence):
            continue
        by_source.setdefault(ev.source, []).append(ev)

    lines: list[str] = []
    lines.append(f"TARGET GENES: {', '.join(target_genes)}")
    lines.append("")

    for source, items in by_source.items():
        lines.append(f"[{source.upper()}] ({len(items)} records)")
        for item in items[:max_items // max(len(by_source), 1)]:
            gene_tag = f"{item.gene} " if item.gene else ""
            variant_tag = f"({item.variant}) " if item.variant else ""
            level_tag = f"[Level {item.evidence_level}] " if item.evidence_level else ""
            lines.append(
                f"  - {gene_tag}{variant_tag}{level_tag}"
                f"[{item.evidence_type.value}] {item.claim[:200]}"
                f" | Source: {item.source_id}"
            )
        lines.append("")

    return "\n".join(lines)


def _validate_hypothesis(raw: dict[str, Any]) -> BiomarkerHypothesis | None:
    """Attempt to parse and validate a single hypothesis dict."""
    try:
        # Normalize supporting_evidence
        raw_se = raw.get("supporting_evidence", [])
        if isinstance(raw_se, list):
            normalized_se = []
            for item in raw_se:
                if isinstance(item, dict):
                    normalized_se.append(
                        SupportingEvidence(
                            source=str(item.get("source", "")),
                            id=str(item.get("id", "")),
                            claim=str(item.get("claim", ""))[:300],
                        )
                    )
            raw["supporting_evidence"] = normalized_se

        # Normalize ranking_rationale
        raw_rr = raw.get("ranking_rationale", {})
        if isinstance(raw_rr, dict):
            raw["ranking_rationale"] = RankingRationale(
                direct_target=bool(raw_rr.get("direct_target", False)),
                civic_level=raw_rr.get("civic_level"),
                pubmed_hits=int(raw_rr.get("pubmed_hits", 0)),
                clinical_trials=int(raw_rr.get("clinical_trials", 0)),
                pathway_support=bool(raw_rr.get("pathway_support", False)),
            )

        return BiomarkerHypothesis.model_validate(raw)
    except (ValidationError, Exception) as exc:
        logger.debug("Hypothesis validation failed: %s", exc)
        return None


class BiomarkerSynthesisAgent:
    """
    Self-healing LLM synthesis agent.

    Execution order
    ───────────────
    1. Try Gemini (if primary) with its built-in retry-after logic.
    2. On Gemini hard failure (auth / quota exhausted) → switch to Ollama.
    3. If Ollama also fails → wait HEAL_SLEEP seconds, then retry Gemini.
    4. After MAX_ROUNDS total rounds without success → return deterministic ranking.

    Ollama uses a simple numbered-sentence prompt (local models can't reliably
    output nested JSON).  Gemini gets the full structured JSON schema prompt.
    """

    MAX_ROUNDS  = 3     # total self-healing rounds before giving up
    HEAL_SLEEP  = 45    # seconds to wait before retrying after all backends fail

    def __init__(self, backend: LLMBackend) -> None:
        self.backend = backend

    async def run(
        self,
        drug_name: str,
        moa_description: str,
        pre_ranked_hypotheses: list[BiomarkerHypothesis],
        all_evidence: list[Any],
        target_genes: list[str],
        top_n: int = 10,
    ) -> list[BiomarkerHypothesis]:
        """
        Self-healing LLM synthesis loop.

        Each round tries Gemini then Ollama.  If both fail the loop sleeps
        HEAL_SLEEP seconds and retries (up to MAX_ROUNDS total).
        Always returns a result — worst case is the deterministic ranking.

        Args:
            drug_name: Drug name.
            moa_description: Mechanism of action description.
            pre_ranked_hypotheses: Deterministically ranked hypotheses from RankingAgent.
            all_evidence: Full evidence list for context building.
            target_genes: Direct drug targets.
            top_n: Number of final hypotheses to return.

        Returns:
            List of evidence-backed BiomarkerHypothesis items. Returns an empty
            list when deterministic ranking found no eligible candidates.
        """
        import asyncio
        from llm.backend import OllamaBackend, LOCAL_SYSTEM_PROMPT

        if not pre_ranked_hypotheses:
            logger.warning(
                "[BiomarkerSynthesisAgent] Skipping synthesis for %s: "
                "no deterministically ranked, evidence-backed hypotheses.",
                drug_name,
            )
            return []

        logger.info(
            "[BiomarkerSynthesisAgent] Starting self-healing LLM synthesis "
            "with %s for %s (max %d rounds)",
            self.backend.name, drug_name, self.MAX_ROUNDS,
        )

        evidence_context  = _build_evidence_context(all_evidence, target_genes)
        top_biomarkers    = [h.biomarker for h in pre_ranked_hypotheses[:top_n]]
        is_gemini_primary = "gemini" in self.backend.name.lower()

        gemini_prompt = build_evidence_prompt(
            drug_name=drug_name,
            moa_description=moa_description,
            evidence_summary=evidence_context,
            top_n=top_n,
        )
        local_prompt = build_narrative_prompt(
            drug_name=drug_name,
            moa_description=moa_description,
            biomarkers=top_biomarkers,
            evidence_summary=evidence_context,
        )

        for round_num in range(1, self.MAX_ROUNDS + 1):
            logger.info("[BiomarkerSynthesisAgent] Round %d/%d", round_num, self.MAX_ROUNDS)

            # ── Gemini attempt ────────────────────────────────────────────────
            if is_gemini_primary:
                result = await self._call_backend(
                    self.backend, gemini_prompt, ONCOLOGY_SYSTEM_PROMPT,
                    pre_ranked_hypotheses, top_biomarkers, drug_name, is_local=False,
                )
                if result is not None:
                    logger.info("[BiomarkerSynthesisAgent] Gemini succeeded in round %d", round_num)
                    return result

            # ── Ollama / meditron attempt ─────────────────────────────────────
            ollama = OllamaBackend()
            result = await self._call_backend(
                ollama, local_prompt, LOCAL_SYSTEM_PROMPT,
                pre_ranked_hypotheses, top_biomarkers, drug_name, is_local=True,
            )
            if result is not None:
                logger.info("[BiomarkerSynthesisAgent] Ollama succeeded in round %d", round_num)
                return result

            if round_num < self.MAX_ROUNDS:
                logger.warning(
                    "[BiomarkerSynthesisAgent] Both backends failed in round %d. "
                    "Sleeping %ds before retry…",
                    round_num, self.HEAL_SLEEP,
                )
                await asyncio.sleep(self.HEAL_SLEEP)

        # ── All rounds exhausted — return deterministic ranking ───────────────
        logger.error(
            "[BiomarkerSynthesisAgent] All %d rounds failed. "
            "Returning deterministic ranking without LLM narrative.",
            self.MAX_ROUNDS,
        )
        for h in pre_ranked_hypotheses:
            if not h.hypothesis:
                h.hypothesis = (
                    f"Deterministic evidence-based ranking for {drug_name}. "
                    f"Supported by {len(h.supporting_sources)} evidence source(s)."
                )
        return pre_ranked_hypotheses[:top_n]

    # ─────────────────────────────────────────────────────────────────────────

    async def _call_backend(
        self,
        backend: Any,
        user_prompt: str,
        system_prompt: str,
        pre_ranked: list[BiomarkerHypothesis],
        top_biomarkers: list[str],
        drug_name: str,
        is_local: bool,
    ) -> list[BiomarkerHypothesis] | None:
        """
        Single backend call with one retry on parse failure.
        Returns enriched hypotheses on success, None on any unrecoverable error.
        """
        _HARD_FAIL = (
            "API_KEY_INVALID", "API key not valid", "INVALID_ARGUMENT",
            "NOT_FOUND", "not found for API version",
            "Unauthorized", "authentication",
        )

        for attempt in range(2):
            try:
                raw = await backend.generate(system_prompt, user_prompt)
                logger.info(
                    "[BiomarkerSynthesisAgent] %s returned %d chars (local=%s)",
                    backend.name, len(raw), is_local,
                )
                logger.debug("Raw response preview: %s", raw[:400])

                if is_local:
                    result = self._inject_narratives(raw, pre_ranked, top_biomarkers, drug_name)
                else:
                    result = self._parse_and_validate(raw, pre_ranked)

                if result:
                    return result

                # Got a response but couldn't parse it — retry with nudge
                if attempt == 0:
                    nudge = (
                        "\n\nREMINDER: Number every answer line as '1.', '2.', etc."
                        if is_local
                        else "\n\nReturn ONLY the JSON array starting with [ and ending with ]."
                    )
                    user_prompt = user_prompt + nudge
                    logger.warning(
                        "[BiomarkerSynthesisAgent] %s parse failed — retrying with nudge",
                        backend.name,
                    )

            except Exception as exc:
                exc_str = str(exc)
                if any(sig in exc_str for sig in _HARD_FAIL):
                    logger.warning(
                        "[BiomarkerSynthesisAgent] %s hard failure: %s — giving up on this backend",
                        backend.name, exc_str[:120],
                    )
                    return None
                logger.warning(
                    "[BiomarkerSynthesisAgent] %s attempt %d error: %s",
                    backend.name, attempt + 1, exc_str[:120],
                )
                if attempt == 1:
                    return None

        return None

    def _inject_narratives(
        self,
        raw_response: str,
        pre_ranked: list[BiomarkerHypothesis],
        top_biomarkers: list[str],
        drug_name: str,
    ) -> list[BiomarkerHypothesis] | None:
        """
        Parse numbered lines from a local model's response and inject them
        as the hypothesis field of pre-ranked hypotheses.

        Accepts:
          "1. PDCD1 is the direct target…"
          "1) PDCD1 is the direct target…"
          "PDCD1: PDCD1 is…"  (fallback: scan for biomarker name in any line)
        """
        narratives = _extract_narratives(raw_response, top_biomarkers)

        # Fallback: search any line containing the biomarker name
        if not narratives:
            for line in raw_response.splitlines():
                stripped = line.strip()
                if not stripped or len(stripped) < 10:
                    continue
                for bm in top_biomarkers:
                    if bm.lower() in stripped.lower() and bm not in narratives:
                        # Strip leading "Biomarker:" or numbering if present
                        sentence = re.sub(r"^[\d.):]+\s*", "", stripped).strip()
                        narratives[bm] = sentence
                        break

        if not narratives:
            logger.warning(
                "[BiomarkerSynthesisAgent] Local model produced no parseable narrative. "
                "First 400 chars: %s", raw_response[:400],
            )
            return None

        injected = 0
        result = list(pre_ranked)
        for hyp in result:
            sentence = narratives.get(hyp.biomarker)
            if sentence:
                hyp.hypothesis = (
                    f"{sentence} "
                    f"[{len(hyp.supporting_sources)} evidence source(s) · meditron]"
                )
                injected += 1
            elif not hyp.hypothesis:
                hyp.hypothesis = (
                    f"Evidence-based candidate for {drug_name}. "
                    f"Supported by {len(hyp.supporting_sources)} source(s)."
                )

        logger.info(
            "[BiomarkerSynthesisAgent] Injected %d/%d narratives from local model",
            injected, len(result),
        )
        return result if injected > 0 else None

    def _parse_and_validate(
        self,
        raw_response: str,
        fallback_hypotheses: list[BiomarkerHypothesis],
    ) -> list[BiomarkerHypothesis]:
        """
        Parse LLM JSON response and validate each hypothesis.
        Merges LLM narrative into pre-ranked deterministic hypotheses.
        """
        raw_list = _extract_json_array(raw_response)
        validated: list[BiomarkerHypothesis] = []

        # Build lookup from pre-ranked by biomarker name. LLM output may enrich
        # narrative text only; it may not introduce candidates or evidence.
        ranked_lookup: dict[str, BiomarkerHypothesis] = {
            h.biomarker.lower(): h for h in fallback_hypotheses
        }
        valid_evidence_ids = {
            evidence.id
            for hypothesis in fallback_hypotheses
            for evidence in hypothesis.supporting_evidence
        }

        for raw_item in raw_list:
            hyp = _validate_hypothesis(raw_item)
            if hyp is None:
                continue

            bm_key = hyp.biomarker.lower()
            if bm_key not in ranked_lookup:
                logger.warning(
                    "[BiomarkerSynthesisAgent] Ignoring LLM-only biomarker %s.",
                    hyp.biomarker,
                )
                continue
            if any(item.id not in valid_evidence_ids for item in hyp.supporting_evidence):
                logger.warning(
                    "[BiomarkerSynthesisAgent] Ignoring %s due to unrecognized evidence IDs.",
                    hyp.biomarker,
                )
                continue

            # Preserve all deterministic values and citations. The model is
            # limited to a narrative refinement of an already grounded result.
            pre = ranked_lookup[bm_key].model_copy(deep=True)
            pre.hypothesis = hyp.hypothesis
            validated.append(pre)

        return validated
