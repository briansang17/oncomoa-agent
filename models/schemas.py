"""
OncoMOA Biomarker Agent — Pydantic v2 Schemas
All data models for evidence normalization, biomarker hypotheses, and agent outputs.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


# ─── Enumerations ─────────────────────────────────────────────────────────────

class EvidenceType(str, Enum):
    PREDICTIVE = "Predictive"
    PROGNOSTIC = "Prognostic"
    DIAGNOSTIC = "Diagnostic"
    PREDISPOSING = "Predisposing"
    ONCOGENIC = "Oncogenic"
    FUNCTIONAL = "Functional"
    UNKNOWN = "Unknown"


class EvidenceDirection(str, Enum):
    SUPPORTS = "Supports"
    DOES_NOT_SUPPORT = "Does Not Support"
    UNKNOWN = "Unknown"


class BiomarkerCategory(str, Enum):
    MUTATION = "mutation"
    EXPRESSION = "expression"
    COPY_NUMBER = "copy_number"
    FUSION = "fusion"
    PATHWAY_SIGNATURE = "pathway_signature"
    IMMUNE_SIGNATURE = "immune_signature"
    PROTEIN = "protein"
    OTHER = "other"


class BiomarkerType(str, Enum):
    PREDICTIVE = "predictive"
    PROGNOSTIC = "prognostic"
    BOTH = "both"
    UNKNOWN = "unknown"


class ResponseDirection(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    RESISTANCE = "resistance"
    UNKNOWN = "unknown"


class NodeType(str, Enum):
    DRUG = "Drug"
    GENE = "Gene"
    MUTATION = "Mutation"
    DISEASE = "Disease"
    PATHWAY = "Pathway"
    CLINICAL_TRIAL = "ClinicalTrial"
    BIOMARKER = "Biomarker"


# ─── Normalized Evidence (unified across all APIs) ────────────────────────────

class NormalizedEvidence(BaseModel):
    """Unified evidence record normalized from any data source."""
    source: str = Field(..., description="Source name: CIViC, PubMed, ClinicalTrials, etc.")
    source_id: str = Field(..., description="Record ID within the source")
    gene: Optional[str] = Field(None, description="Gene symbol")
    variant: Optional[str] = Field(None, description="Variant description, e.g. G12C")
    disease: Optional[str] = Field(None, description="Cancer type or disease name")
    drug: Optional[str] = Field(None, description="Drug name if applicable")
    evidence_type: EvidenceType = Field(EvidenceType.UNKNOWN)
    evidence_direction: EvidenceDirection = Field(EvidenceDirection.UNKNOWN)
    evidence_level: Optional[str] = Field(None, description="CIViC level A-E or equivalent")
    claim: str = Field("", description="Human-readable summary of the evidence")
    strength: float = Field(0.0, description="Numeric strength score (computed from weights)")
    raw_data: dict[str, Any] = Field(default_factory=dict, description="Raw API payload")


# ─── Drug Agent Output ────────────────────────────────────────────────────────

class DrugInfo(BaseModel):
    """Structured drug information resolved from APIs."""
    drug_name: str
    synonyms: list[str] = Field(default_factory=list)
    mechanism_of_action: str = ""
    target_genes: list[str] = Field(default_factory=list)
    drug_class: Optional[str] = None
    chembl_id: Optional[str] = None
    sources: list[str] = Field(default_factory=list)


# ─── Target Biology ───────────────────────────────────────────────────────────

class GeneInfo(BaseModel):
    """Biological information about a target gene."""
    symbol: str
    ensembl_id: Optional[str] = None
    uniprot_id: Optional[str] = None
    full_name: Optional[str] = None
    summary: Optional[str] = None
    pathways: list[str] = Field(default_factory=list)
    associated_diseases: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)


# ─── Pathway ──────────────────────────────────────────────────────────────────

class PathwayInfo(BaseModel):
    """A biological pathway with its member genes."""
    pathway_id: str
    pathway_name: str
    source: str  # Reactome, WikiPathways, MSigDB
    genes: list[str] = Field(default_factory=list)
    description: Optional[str] = None


# ─── Clinical Trial ───────────────────────────────────────────────────────────

class ClinicalTrialInfo(BaseModel):
    """Structured clinical trial record."""
    trial_id: str
    title: str = ""
    phase: Optional[str] = None
    condition: Optional[str] = None
    drug: Optional[str] = None
    status: Optional[str] = None
    biomarker_mentions: list[str] = Field(default_factory=list)
    inclusion_criteria: Optional[str] = None
    exclusion_criteria: Optional[str] = None
    exploratory_endpoints: list[str] = Field(default_factory=list)
    stratification_factors: list[str] = Field(default_factory=list)


# ─── Literature Evidence ──────────────────────────────────────────────────────

class PubMedArticle(BaseModel):
    """A PubMed article with optional PubTator annotations."""
    pmid: str
    title: str = ""
    abstract: str = ""
    genes_mentioned: list[str] = Field(default_factory=list)
    mutations_mentioned: list[str] = Field(default_factory=list)
    diseases_mentioned: list[str] = Field(default_factory=list)
    drugs_mentioned: list[str] = Field(default_factory=list)


# ─── Knowledge Graph Summary ──────────────────────────────────────────────────

class KGEdge(BaseModel):
    source: str
    target: str
    relation: str
    weight: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeGraphSummary(BaseModel):
    """Summary statistics and key paths from the knowledge graph."""
    node_count: int = 0
    edge_count: int = 0
    drug_target_chains: list[list[str]] = Field(default_factory=list)
    candidate_biomarkers: list[str] = Field(default_factory=list)
    top_connected_genes: list[str] = Field(default_factory=list)


# ─── Ranking Rationale ────────────────────────────────────────────────────────

class RankingRationale(BaseModel):
    """Explains how a biomarker's score was computed."""
    direct_target: bool = False
    civic_level: Optional[str] = None
    civic_evidence_count: int = 0
    pubmed_hits: int = 0
    clinical_trials: int = 0
    pathway_support: bool = False
    raw_score: float = 0.0


# ─── Supporting Evidence Item ─────────────────────────────────────────────────

class SupportingEvidence(BaseModel):
    """A single piece of supporting evidence for a hypothesis."""
    source: str
    id: str
    claim: str


# ─── Biomarker Hypothesis (primary output object) ────────────────────────────

class BiomarkerHypothesis(BaseModel):
    """A ranked predictive/prognostic biomarker hypothesis with full evidence."""
    rank: int
    biomarker: str
    biomarker_category: BiomarkerCategory = BiomarkerCategory.OTHER
    biomarker_type: BiomarkerType = BiomarkerType.UNKNOWN
    direction: ResponseDirection = ResponseDirection.UNKNOWN
    confidence_score: float = Field(ge=0.0, le=100.0)
    predictive_score: float = Field(ge=0.0, le=100.0)
    prognostic_score: float = Field(ge=0.0, le=100.0)
    evidence_level: Optional[str] = None
    drug_relevance: str = ""
    supporting_sources: list[str] = Field(default_factory=list)
    supporting_evidence: list[SupportingEvidence] = Field(default_factory=list)
    ranking_rationale: RankingRationale = Field(default_factory=RankingRationale)
    hypothesis: str = ""

    @field_validator("confidence_score", "predictive_score", "prognostic_score", mode="before")
    @classmethod
    def clamp_score(cls, v: Any) -> float:
        """Clamp scores to [0, 100]."""
        return max(0.0, min(100.0, float(v)))


# ─── Top-Level Agent Output ───────────────────────────────────────────────────

class AgentOutput(BaseModel):
    """Top-level output object for the OncoMOA pipeline."""
    drug_name: str
    moa_description: str
    target_genes: list[str] = Field(default_factory=list)
    llm_backend_used: str = ""
    knowledge_graph_summary: KnowledgeGraphSummary = Field(default_factory=KnowledgeGraphSummary)
    hypotheses: list[BiomarkerHypothesis] = Field(default_factory=list)
    failed_sources: list[str] = Field(default_factory=list)
    successful_sources: list[str] = Field(default_factory=list)
    total_evidence_items: int = 0
    run_metadata: dict[str, Any] = Field(default_factory=dict)


# ─── Internal Scoring Container ───────────────────────────────────────────────

class CandidateBiomarker(BaseModel):
    """Internal scoring object before final ranking."""
    gene: str
    variant: Optional[str] = None
    biomarker_label: str = ""  # e.g. "KRAS G12C"
    category: BiomarkerCategory = BiomarkerCategory.OTHER

    # Raw score accumulators
    civic_score: float = 0.0
    trial_score: float = 0.0
    pubmed_score: float = 0.0
    target_score: float = 0.0
    pathway_score: float = 0.0

    # Independent predictive vs prognostic scores
    predictive_raw: float = 0.0
    prognostic_raw: float = 0.0

    # Evidence references
    evidence_items: list[NormalizedEvidence] = Field(default_factory=list)
    pubmed_ids: list[str] = Field(default_factory=list)
    trial_ids: list[str] = Field(default_factory=list)
    civic_ids: list[str] = Field(default_factory=list)

    # Rationale flags
    is_direct_target: bool = False
    is_pathway_connected: bool = False
    best_civic_level: Optional[str] = None

    @property
    def total_raw_score(self) -> float:
        return (
            self.civic_score
            + self.trial_score
            + self.pubmed_score
            + self.target_score
            + self.pathway_score
        )

    @property
    def has_minimum_evidence(self) -> bool:
        """Return whether evidence meets the configured structured/publication gate."""
        from config import MIN_DB_SOURCES, MIN_PUB_SOURCES

        db_sources = [e for e in self.evidence_items if e.source not in ("PubMed",)]
        pub_sources = self.pubmed_ids
        return len(db_sources) >= MIN_DB_SOURCES and len(pub_sources) >= MIN_PUB_SOURCES
