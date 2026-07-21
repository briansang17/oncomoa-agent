"""
OncoMOA Biomarker Agent — Central Configuration
All parameters, weights, API URLs, and drug routing rules live here.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Project Paths ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
OUTPUT_DIR = ROOT_DIR / "output"
LOG_DIR = ROOT_DIR / "logs"
CACHE_DIR = Path(os.getenv("CACHE_DIR", str(ROOT_DIR / ".cache" / "oncomoa")))

OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_RESULTS_JSON = OUTPUT_DIR / "results.json"
OUTPUT_KG_GRAPHML = OUTPUT_DIR / "knowledge_graph.graphml"
OUTPUT_KG_JSON = OUTPUT_DIR / "knowledge_graph.json"
OUTPUT_EVIDENCE_CSV = OUTPUT_DIR / "evidence_summary.csv"
LOG_FILE = LOG_DIR / "oncomoa.log"

# ─── Cache ────────────────────────────────────────────────────────────────────
CACHE_TTL: int = int(os.getenv("CACHE_TTL", 604800))  # 7 days

# ─── LLM Configuration ────────────────────────────────────────────────────────
LLM_BACKEND: str = os.getenv("LLM_BACKEND", "auto")  # auto | gemini | ollama | meditron
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = "gemini-3.5-flash"

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_PRIMARY_MODEL: str = "meditron"
OLLAMA_FALLBACK_MODEL: str = "llama3.2"

# ─── Smart LLM Routing — Drug Class Rules ─────────────────────────────────────

# Merck oncology drugs → Gemini (complex IO biomarker landscape)
MERCK_DRUGS: list[str] = [
    "pembrolizumab", "keytruda",
    "belzutifan", "welireg",
    "tepotinib", "tepmetko",
    "quizartinib", "vanflyta",
    "lenvatinib", "lenvima",
    "ertumaxomab",
    "upifitamab rilsodotin",
]

# Bispecific PD-L1/VEGF antibodies → Gemini (dual-target biology)
BISPECIFIC_PDLVEGF_DRUGS: list[str] = [
    "ivonescimab", "ak112",
    "cadonilimab", "aq154",
    "faricimab", "vabysmo",
    "kn046",
    "ly3434172",
    "pm8002",
    "hz31",
    "syd985",
    "lq036",
]

# ADC name-suffix patterns → Gemini (3 independent biomarker axes: target, linker, payload)
ADC_SUFFIXES: list[str] = [
    "vedotin",
    "emtansine",
    "deruxtecan",
    "govitecan",
    "ozogamicin",
    "mafodotin",
    "tesirine",
    "ravtansine",
    "duocarmazine",
    "pasudotox",
    "rilsodotin",
    "ixtecan",
    "topotecan",  # some ADC payloads
]


def requires_gemini(drug_name: str) -> bool:
    """Return True if the drug class warrants Gemini backend routing."""
    name = drug_name.lower().strip()
    if name in MERCK_DRUGS:
        return True
    if name in BISPECIFIC_PDLVEGF_DRUGS:
        return True
    if any(name.endswith(suffix) for suffix in ADC_SUFFIXES):
        return True
    return False


# ─── Evidence Scoring Weights ─────────────────────────────────────────────────
CIVIC_EVIDENCE_WEIGHTS: dict[str, float] = {
    "A": 5.0,
    "B": 4.0,
    "C": 3.0,
    "D": 2.0,
    "E": 1.0,
}

TRIAL_PHASE_WEIGHTS: dict[str, float] = {
    "Phase 3": 3.0,
    "Phase III": 3.0,
    "PHASE3": 3.0,
    "Phase 2": 2.0,
    "Phase II": 2.0,
    "PHASE2": 2.0,
    "Phase 1": 1.0,
    "Phase I": 1.0,
    "PHASE1": 1.0,
}

PUBMED_HIT_WEIGHT: float = 0.1
DIRECT_TARGET_WEIGHT: float = 3.0
PATHWAY_CONNECTED_WEIGHT: float = 2.0

# Thresholds for predictive/prognostic classification
PREDICTIVE_THRESHOLD: float = 20.0
PROGNOSTIC_THRESHOLD: float = 20.0

# Minimum evidence gate: a hypothesis must have at least one DB source AND one pub
MIN_DB_SOURCES: int = 1
MIN_PUB_SOURCES: int = 1

# Top-N biomarkers to return (overridable via CLI)
DEFAULT_TOP_N: int = 10

# ─── Biomarker Categories ────────────────────────────────────────────────────
BIOMARKER_CATEGORIES: list[str] = [
    "mutation",
    "expression",
    "copy_number",
    "fusion",
    "pathway_signature",
    "immune_signature",
    "protein",
    "other",
]

# ─── API Base URLs ────────────────────────────────────────────────────────────

# Tier 1
OPEN_TARGETS_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
CIVIC_API_BASE = "https://civicdb.org/api"
PUBMED_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBTATOR_BASE = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"
CLINICAL_TRIALS_BASE = "https://clinicaltrials.gov/api/v2"

# Tier 2
ENSEMBL_REST_BASE = "https://rest.ensembl.org"
MYGENE_BASE = "https://mygene.info/v3"
MYVARIANT_BASE = "https://myvariant.info/v1"
DGIDB_BASE = "https://dgidb.org/api/graphql"
CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
UNIPROT_BASE = "https://rest.uniprot.org"

# Tier 3 — Pathway
REACTOME_BASE = "https://reactome.org/ContentService"
WIKIPATHWAYS_BASE = "https://webservice.wikipathways.org"

# Tier 4 — Cancer-specific
CBIOPORTAL_BASE = "https://www.cbioportal.org/api"
GDC_BASE = "https://api.gdc.cancer.gov"

# ─── PubMed Search Config ─────────────────────────────────────────────────────
PUBMED_MAX_RESULTS: int = 20
NCBI_API_KEY: str = os.getenv("NCBI_API_KEY", "")

# ─── Clinical Trials Config ───────────────────────────────────────────────────
CLINICAL_TRIALS_MAX_RESULTS: int = 20

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ─── HTTP Timeouts (seconds) ──────────────────────────────────────────────────
HTTP_TIMEOUT: int = 30
HTTP_RETRIES: int = 3
HTTP_RETRY_WAIT: float = 1.0
