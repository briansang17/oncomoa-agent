"""
OncoMOA — LLM Backend Abstraction
Provides a unified async interface for Gemini and Ollama (meditron/llama3.2),
with smart drug-class routing and fully self-healing retry/fallback logic.

Retry strategy (all free-tier safe):
  Gemini 429 (rate limit)      → wait retry-after + buffer, retry up to 3x
  Gemini 429 (quota exhausted) → immediately fall back to Ollama
  Gemini auth / 404            → immediately fall back to Ollama
  Ollama empty / error         → retry with simpler prompt, then give up

Usage:
    backend = get_backend(drug_name="pembrolizumab")
    response = await backend.generate(system_prompt, user_prompt)
"""

from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod

import aiohttp

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_BACKEND,
    OLLAMA_BASE_URL,
    OLLAMA_PRIMARY_MODEL,
    OLLAMA_FALLBACK_MODEL,
    requires_gemini,
)

logger = logging.getLogger(__name__)

# Signals that mean the API key / model is broken — switch backend immediately
_GEMINI_HARD_FAIL = (
    "API_KEY_INVALID", "API key not valid", "INVALID_ARGUMENT",
    "NOT_FOUND", "not found for API version",
    "authentication", "Unauthorized",
)

# Signals that mean the daily/project quota is exhausted — also switch backend
_GEMINI_QUOTA_EXHAUSTED = ("limit: 0",)


def _parse_retry_delay(exc_str: str, default: float = 10.0) -> float:
    """Extract the suggested retry-after seconds from a Gemini error string."""
    m = re.search(r"retry[_ ]in\s+([\d.]+)s", exc_str, re.I)
    raw = float(m.group(1)) if m else default
    return min(raw + 2.0, 60.0)   # add 2 s buffer, cap at 60 s


# ─── Abstract Base ─────────────────────────────────────────────────────────────

class LLMBackend(ABC):
    """Abstract LLM backend interface."""

    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate a response given system and user prompts.

        Args:
            system_prompt: Role/instruction context for the model.
            user_prompt: The actual query or task.

        Returns:
            Model response as a string.

        Raises:
            RuntimeError: on unrecoverable failure after all internal retries.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name for logging."""
        ...


# ─── Gemini Backend ───────────────────────────────────────────────────────────

class GeminiBackend(LLMBackend):
    """
    Google Gemini backend (google-genai SDK).
    Self-heals on rate limits: retries up to MAX_RETRIES times, honouring
    the retry-after delay the API returns.  Hard-fails immediately on auth
    errors or daily quota exhaustion so the caller can switch to Ollama.
    """

    MAX_RETRIES = 3

    def __init__(self, api_key: str = GEMINI_API_KEY, model: str = GEMINI_MODEL) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set. Add it to your .env file.")
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        return f"Gemini/{self._model}"

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call Gemini and return the text response.
        Retries automatically on transient rate-limits (429) with the
        server-suggested delay.  Raises immediately on auth / quota errors.
        """
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._api_key)

        last_exc: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = client.models.generate_content(
                    model=self._model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.1,
                        max_output_tokens=8192,
                        # Disable extended thinking so the full token budget
                        # goes to the JSON output, not reasoning chains
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                text = response.text
                if attempt > 0:
                    logger.info("Gemini succeeded on attempt %d", attempt + 1)
                logger.debug("Gemini response: %d chars", len(text))
                return text

            except Exception as exc:
                exc_str = str(exc)
                last_exc = exc
                logger.error("GeminiBackend attempt %d/%d failed: %s",
                             attempt + 1, self.MAX_RETRIES, exc_str[:200])

                # Hard failures — caller must switch backend, no point retrying
                if any(sig in exc_str for sig in _GEMINI_HARD_FAIL):
                    raise

                # Quota fully exhausted (limit:0) — switch backend immediately
                if any(sig in exc_str for sig in _GEMINI_QUOTA_EXHAUSTED):
                    logger.warning("Gemini daily quota exhausted — switching backend")
                    raise

                # Transient rate limit — wait and retry
                if "RESOURCE_EXHAUSTED" in exc_str and attempt < self.MAX_RETRIES - 1:
                    wait = _parse_retry_delay(exc_str)
                    logger.info(
                        "Gemini rate-limited. Waiting %.1f s then retrying (attempt %d/%d)…",
                        wait, attempt + 2, self.MAX_RETRIES,
                    )
                    await asyncio.sleep(wait)
                    continue

                # Unknown error on last attempt
                if attempt == self.MAX_RETRIES - 1:
                    raise

        raise last_exc or RuntimeError("Gemini: all retries exhausted")


# ─── Ollama Backend ───────────────────────────────────────────────────────────

class OllamaBackend(LLMBackend):
    """
    Ollama local LLM backend using the /api/chat messages interface.
    Tries primary model (meditron) then falls back to llama3.2.
    Prompt is truncated to keep inference time under ~3 minutes on CPU.
    """

    # 7B models on Apple Silicon can do ~3 tok/s; 1024 out = ~6 min worst case
    _TIMEOUT   = 300    # 5 min total
    _MAX_CHARS = 3500   # ~875 tokens — fits any 7B context safely

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        primary_model: str = OLLAMA_PRIMARY_MODEL,
        fallback_model: str = OLLAMA_FALLBACK_MODEL,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._primary = primary_model
        self._fallback = fallback_model
        self._active: str | None = None

    @property
    def name(self) -> str:
        return f"Ollama/{self._active or self._primary}"

    async def _available_models(self) -> list[str]:
        """Return list of short model names available in the local Ollama server."""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as s:
                async with s.get(f"{self._base_url}/api/tags") as r:
                    if r.status != 200:
                        return []
                    data = await r.json()
                    return [m.get("name", "").split(":")[0]
                            for m in data.get("models", [])]
        except Exception:
            return []

    async def _chat(self, model: str, system: str, user: str) -> str:
        """
        Send a chat request to the Ollama /api/chat endpoint and return
        the assistant's reply text.  Raises on HTTP error.
        """
        # Truncate user prompt to keep inference fast
        if len(user) > self._MAX_CHARS:
            user = (
                user[: self._MAX_CHARS]
                + "\n\n[Evidence truncated for local model. Use the evidence above.]"
            )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": 0.15,
                "num_predict": 800,
                "num_ctx": 4096,
            },
        }

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._TIMEOUT)
        ) as s:
            async with s.post(f"{self._base_url}/api/chat", json=payload) as r:
                r.raise_for_status()
                data = await r.json()

        text = data.get("message", {}).get("content", "")
        logger.debug("Ollama [%s] response: %d chars — first 200: %s",
                     model, len(text), text[:200])
        return text

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate using the primary model (meditron), fall back to llama3.2
        if meditron is unavailable or errors.
        """
        available = await self._available_models()
        logger.info("Ollama available models: %s", available)

        for model in (self._primary, self._fallback):
            base = model.split(":")[0]
            if base not in available and model not in available:
                logger.info("Ollama model %s not available — skipping", model)
                continue
            try:
                self._active = model
                text = await self._chat(model, system_prompt, user_prompt)
                if text.strip():
                    return text
                logger.warning("Ollama [%s] returned empty response", model)
            except Exception as exc:
                logger.warning("Ollama [%s] error: %s", model, exc)

        raise RuntimeError(
            f"No usable Ollama model. Run: ollama pull {self._primary}"
        )


# ─── Backend Factory ──────────────────────────────────────────────────────────

def get_backend(drug_name: str = "", override: str = "") -> LLMBackend:
    """
    Resolve the appropriate LLM backend using smart routing rules.

    Priority order:
    1. CLI override (--backend flag)
    2. LLM_BACKEND env var (if not "auto")
    3. Drug-class auto-routing:
       Merck drugs / bispecific PD-L1+VEGF / ADCs → Gemini
       Everything else → Ollama (meditron → llama3.2)

    Args:
        drug_name: Drug being analysed (used for auto-routing).
        override: Explicit backend choice: "gemini" | "ollama" | "meditron" | "auto".
            The special value "none" disables synthesis and must be handled by
            the caller before requesting a backend.

    Returns:
        An instantiated LLMBackend.
    """
    choice = (override or LLM_BACKEND).lower()

    if choice == "none":
        raise ValueError("LLM synthesis is disabled (backend override is 'none').")

    if choice == "gemini":
        logger.info("LLM backend: Gemini (explicit)")
        return GeminiBackend()

    if choice in ("ollama", "meditron"):
        logger.info("LLM backend: Ollama/%s (explicit)", OLLAMA_PRIMARY_MODEL)
        return OllamaBackend()

    # Auto routing
    if drug_name and requires_gemini(drug_name):
        if GEMINI_API_KEY:
            logger.info(
                "LLM backend: Gemini (auto-routed — drug class match for '%s')", drug_name
            )
            return GeminiBackend()
        logger.warning(
            "Gemini auto-routed for '%s' but no API key — falling back to Ollama", drug_name
        )

    logger.info("LLM backend: Ollama (auto-routed → %s → %s)",
                OLLAMA_PRIMARY_MODEL, OLLAMA_FALLBACK_MODEL)
    return OllamaBackend()


# ─── Oncology System Prompts ──────────────────────────────────────────────────

ONCOLOGY_SYSTEM_PROMPT = """You are a senior translational oncologist and precision medicine researcher.

RULES (follow without exception):
1. Use ONLY the evidence provided. Do NOT use external knowledge.
2. Do NOT hallucinate biomarkers, genes, studies, or references.
3. Return ONLY valid JSON — no markdown, no code fences, no explanation text.
4. Rank biomarkers by evidence strength, not theoretical plausibility.
5. Omit a biomarker if evidence is insufficient rather than speculate."""


LOCAL_SYSTEM_PROMPT = """You are a medical oncology expert. Answer concisely and factually using only the evidence provided."""


def build_evidence_prompt(
    drug_name: str,
    moa_description: str,
    evidence_summary: str,
    top_n: int = 10,
) -> str:
    """
    Full structured JSON prompt for large models (Gemini).
    Expects a JSON array in response.
    """
    return f"""DRUG: {drug_name}
MECHANISM OF ACTION: {moa_description}

=== EVIDENCE (use ONLY this) ===
{evidence_summary}
================================

Generate up to {top_n} ranked biomarker hypotheses for {drug_name} based SOLELY on the evidence above.
Return [] when the evidence is insufficient. Do not treat the TARGET GENES line
as supporting evidence; every returned item must cite a record ID from the
evidence section.

Return a JSON array — each item MUST have ALL these fields:
{{
  "rank": <int>,
  "biomarker": "<name>",
  "biomarker_category": "<mutation|expression|copy_number|fusion|pathway_signature|immune_signature|protein|other>",
  "biomarker_type": "<predictive|prognostic|both|unknown>",
  "direction": "<positive|negative|resistance|unknown>",
  "confidence_score": <0-100>,
  "predictive_score": <0-100>,
  "prognostic_score": <0-100>,
  "evidence_level": "<A|B|C|D|E|null>",
  "drug_relevance": "<one sentence>",
  "supporting_sources": ["<id>"],
  "supporting_evidence": [{{"source": "<name>", "id": "<id>", "claim": "<text>"}}],
  "ranking_rationale": {{
    "direct_target": <true|false>,
    "civic_level": "<A-E|null>",
    "pubmed_hits": <int>,
    "clinical_trials": <int>,
    "pathway_support": <true|false>
  }},
  "hypothesis": "<2-3 sentence rationale grounded in the provided evidence>"
}}

Return ONLY the JSON array starting with [ and ending with ]. No other text."""


def build_narrative_prompt(
    drug_name: str,
    moa_description: str,
    biomarkers: list[str],
    evidence_summary: str,
) -> str:
    """
    Minimal numbered-sentence prompt for local models (meditron / llama3.2).
    Asks only for plain-English sentences, one per biomarker.
    These are injected into the deterministic pre-ranked hypotheses.
    """
    numbered = "\n".join(f"{i+1}. {b}" for i, b in enumerate(biomarkers))
    return f"""Drug: {drug_name}. MOA: {moa_description}.

Evidence summary:
{evidence_summary[:2000]}

For each biomarker below, write ONE sentence (max 40 words) explaining its relevance to {drug_name} response using ONLY the evidence above. Number each answer to match the list.

{numbered}

Start your answer with "1." and number every line."""
