# OncoMOA Biomarker Agent

**What it does in one sentence:**
You give it a cancer drug name and how it works — it goes out to 15 scientific databases, builds a knowledge graph, scores the evidence mathematically, and returns a ranked list of biomarkers that predict whether a patient will respond to that drug, with explanations grounded entirely in retrieved evidence.

---

## Structured Graph-RAG

OncoMOA is **structured Graph-RAG**, not vector Graph-RAG. It retrieves
authoritative oncology records from public APIs, normalizes them into a common
evidence schema, builds a directed NetworkX knowledge graph, and ranks
evidence-grounded candidates before optional LLM narrative synthesis.

```mermaid
flowchart LR
    drug[DrugAndMOA] --> targets[DrugTargetRetrieval]
    targets --> biology[TargetBiology]
    biology --> pathway[PathwayExpansion]
    pathway --> evidence[ClinicalLiteratureTrials]
    evidence --> graph[KnowledgeGraph]
    graph --> ranking[EvidenceGatedRanking]
    ranking --> synthesis[OptionalLLMSynthesis]
    synthesis --> results[RankedBiomarkers]
```

The graph is an auditable reasoning layer, not a graph database or an LLM
graph-traversal system:

- **Nodes:** drug, gene, mutation, disease, pathway, clinical trial, and biomarker.
- **Edges:** target, pathway, disease, publication, clinical-trial, and response
  relationships derived from retrieved records.
- **Graph expansion:** graph neighbors can enter the candidate set, but they
  still need at least one structured database record and one PubMed record to
  become a hypothesis.
- **LLM boundary:** the optional LLM sees only pre-ranked, evidence-backed
  candidates. It cannot introduce new biomarkers or citations.

### Graph-RAG compared with vector RAG

| Aspect | Typical vector RAG | OncoMOA structured Graph-RAG |
|---|---|---|
| Retrieval | Similarity search over text chunks | 15 biomedical APIs normalized into evidence records |
| Structure | Flat context window | Directed knowledge graph plus scored relationships |
| Candidate discovery | Top-k retrieved passages | Target biology, pathways, and graph neighborhoods |
| Generation | LLM synthesizes retrieved chunks | LLM optionally writes narratives for pre-ranked candidates |
| Audit trail | Chunk provenance | Source records, deterministic scores, GraphML, and JSON |

---

## Setup

Requires Python 3.10+ and internet access to the public data sources.

```bash
git clone https://github.com/briansang17/oncomoa-agent.git
cd oncomoa-agent

python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

For local synthesis, install Ollama and pull the configured model:

```bash
ollama pull meditron
```

Set `GEMINI_API_KEY` in `.env` only when you want Gemini synthesis. API keys
for the biomedical sources are not required; `NCBI_API_KEY` is optional and
increases PubMed rate limits.

---

## Safety and fail-closed behavior

OncoMOA is a research hypothesis-generation tool, not clinical decision
support. Every returned hypothesis must pass the evidence gate: at least one
structured database source and one PubMed source. If target or evidence
retrieval fails, the system returns an empty hypothesis list, marks failed
sources in `results.json`, and skips LLM synthesis rather than generating
unsupported biomarkers.

---

## Why This Exists

In oncology clinical development, figuring out which patients will respond to a drug — and which won't — is one of the hardest and most expensive problems in medicine. This is the biomarker question.

Traditionally, a translational scientist spends weeks manually:
- Searching PubMed for relevant papers
- Reviewing clinical trial inclusion criteria
- Querying variant databases like CIViC
- Building pathway hypotheses by hand
- Writing up a ranked list of predictive biomarkers

This system automates that entire workflow in minutes. It is designed to think the way a translational oncologist thinks: start with the drug target, expand through biology, gather clinical evidence, rank by strength, then synthesize.

---

## Why So Many Agents? (The Most Important Question)

This is a multi-agent system — nine separate agents, each with a single defined job. That architecture is a deliberate choice, not complexity for its own sake.

**The core problem with a single LLM approach:**
If you just ask an LLM "what are the biomarkers for sotorasib?", it will hallucinate. It will confidently list biomarkers that are not supported by current clinical evidence. It may confuse trial phases, invent citations, or miss recent findings entirely. For clinical work, this is dangerous.

**Why multiple specialized agents solve this:**

Each agent is a domain expert. Breaking the workflow into stages means:

1. Each stage can be verified independently
2. If one stage fails (e.g., CIViC is down), the rest continue
3. Each agent is testable in isolation
4. The system is extensible — you can add a new data source by adding one agent
5. The LLM never sees raw API responses — it only sees pre-filtered, pre-scored evidence

The agents enforce a strict separation between **retrieval** (finding facts) and **generation** (writing narratives). The LLM only ever synthesizes — it never discovers.

This is what makes the system scientifically defensible.

---

## The Nine Agents — What Each One Does and Why It Exists

### 1. Drug Agent
**What it does:**
Resolves the drug name into molecular targets by querying Open Targets (European drug-target database) and ChEMBL (bioactivity database) in parallel.

**Why it exists:**
You cannot rank biomarkers without knowing what the drug actually hits at the molecular level. The drug name alone is insufficient — "sotorasib" needs to map to KRAS. "Olaparib" needs to map to PARP1 and PARP2. ChEMBL also provides mechanism of action annotations and bioactivity data that inform which gene interactions are pharmacologically relevant.

**Why two sources:**
Open Targets has strong disease-association data. ChEMBL has stronger drug-target bioactivity data. Together they cover cases where one source is incomplete.

---

### 2. Target Biology Agent
**What it does:**
For every target gene found by the Drug Agent, fetches its full biological profile from Ensembl, MyGene.info, UniProt, and Open Targets — in parallel.

**Why it exists:**
The drug target is the anchor of the entire hypothesis space. If you know KRAS is the target, you need to know: What pathways does KRAS sit in? What diseases is it associated with? What is the protein's function? What are its aliases (because the same gene appears under different names in different databases)?

This information seeds the pathway expansion step and helps the LLM write biologically accurate rationales later.

**Why four sources:**
Each source has a different specialty. Ensembl has genomic coordinates and transcripts. MyGene has pathway memberships and aliases. UniProt has protein-level function and disease annotations. Open Targets has quantitative disease association scores derived from multiple evidence types. No single source provides all of this.

---

### 3. Pathway Expansion Agent
**What it does:**
Expands from the drug target outward into the biological pathway, generating a list of candidate biomarker genes — genes that are biologically connected to the target even if they are not directly targeted by the drug.

**Why it exists:**
This is one of the most scientifically important steps. In oncology, the most clinically actionable biomarkers are often not the drug target itself — they are other genes in the same pathway whose mutations predict whether the target will respond.

For sotorasib (KRAS G12C inhibitor):
The direct target is KRAS. But the real biomarker landscape includes STK11 (LKB1), which co-occurs with KRAS mutations and predicts resistance. KEAP1 mutations also co-occur and predict poor response. Neither of these is the drug target — they are pathway neighbors.

Without this step, you would only ever discover the obvious biomarker (the target itself). The pathway expansion is what generates non-obvious, biologically plausible hypotheses.

Uses Reactome, WikiPathways, and curated cancer-specific gene sets for key targets.

---

### 4. Clinical Evidence Agent
**What it does:**
For every target gene AND every pathway-expanded candidate gene, searches five structured databases for clinical evidence.

Queries: CIViC, DGIdb, MyVariant/ClinVar, cBioPortal, GDC/TCGA — all in parallel.

**Why it exists:**
Biological plausibility is not enough. A biomarker needs clinical evidence — actual patient data showing that this gene variant changes the response to this drug (or changes prognosis). CIViC provides exactly this, with evidence grades from A (multiple clinical trials) down to E (expert opinion). This is the highest-quality signal in the entire system.

cBioPortal and GDC provide population-level mutation frequency data from TCGA, telling us how common a variant is across cancer types — which matters for clinical utility.

**Why this agent runs in parallel with Literature and Trial agents:**
These three evidence-gathering steps are completely independent. Running them simultaneously cuts total runtime by roughly 60%.

---

### 5. Literature Agent
**What it does:**
Searches PubMed for papers about the drug + each candidate gene + cancer. Fetches the top 20 abstracts per gene. Then sends all PMIDs to PubTator Central, which uses NLP to extract structured entities (genes, mutations, diseases, drugs) from each abstract.

**Why it exists:**
The structured databases (CIViC, cBioPortal) contain curated, vetted findings — but they lag behind the primary literature by months to years. PubMed catches recent findings that have not yet been curated into databases. For a new drug like sotorasib or a recently approved ADC, some of the most important evidence exists only in preprints and recent publications.

PubTator is what converts those unstructured abstracts into structured data that the ranking engine can use. Without PubTator, abstracts are just text. With it, you know which genes and mutations were actually discussed in the paper, which lets you link papers to biomarker candidates systematically.

---

### 6. Trial Agent
**What it does:**
Queries ClinicalTrials.gov for all trials involving the drug. Parses inclusion/exclusion criteria and exploratory endpoints to extract biomarker mentions and stratification factors.

**Why it exists:**
Clinical trial eligibility criteria are one of the most underutilized signals in biomarker discovery. When a Phase III trial says "patients must have KRAS G12C mutation" — that is strong real-world evidence that KRAS G12C is a predictive biomarker. When a trial stratifies patients by STK11 status, that tells you researchers believe STK11 may affect outcomes.

Most researchers search CIViC or PubMed but overlook trial criteria text. This agent systematically mines that signal. It also notes when a biomarker appears in exploratory endpoints — which signals emerging hypothesis-generating intent even before results are published.

---

### 7. Knowledge Graph Agent
**What it does:**
Takes all output from all previous agents and builds a directed knowledge graph using NetworkX. Exports to GraphML (for Cytoscape/Gephi) and JSON.

Nodes: Drug, Gene, Mutation, Disease, Pathway, Clinical Trial, Biomarker
Edges: targets, participates_in, associated_with, predicts_response, predicts_resistance, mentioned_in, tested_in

**Why it exists:**
The knowledge graph serves two purposes.

First, it makes the evidence structure transparent and auditable. You can open the GraphML file in Cytoscape and visually trace the path from a drug to a biomarker hypothesis. This is important for scientific credibility — you can show a reviewer exactly why a biomarker was included.

Second, it enables non-obvious discovery. The graph supports neighborhood expansion — finding genes that are not directly linked to the drug but are reachable within 2–3 hops through pathway or disease connections. These become additional candidate biomarkers that no single database search would have returned.

This is the component that most closely resembles how a scientist thinks when they say "follow the biology."

---

### 8. Ranking Agent
**What it does:**
Scores every candidate biomarker using a deterministic, evidence-weighted formula. No AI involved — pure mathematics. Normalizes all scores to 0–100. Computes predictive and prognostic scores independently.

**Why it exists:**
Before the LLM sees anything, there needs to be a ground-truth ranking that is reproducible, auditable, and bias-free. The scoring weights are explicitly defined (CIViC Level A = 5 points, Phase III trial = 3 points, PubMed hit = 0.1 points, etc.) and can be reviewed and adjusted by a domain expert.

This is what makes the system defensible in a clinical or regulatory context. The ranking is not based on what the AI "thinks" is important — it is based on explicit, weighted, verifiable evidence.

The predictive/prognostic separation is also done here, using independent score accumulators. A biomarker that appears mainly in prognostic studies gets a different classification than one that appears mainly in drug-response studies — even if both accumulate similar total evidence counts.

---

### 9. Biomarker Synthesis Agent (LLM Step)
**What it does:**
Receives the pre-ranked biomarker list and the full evidence context. Invokes the LLM with a strict system prompt. Asks it to write a 2–3 sentence biological rationale for each ranked biomarker, confirm the ranking direction, and return structured JSON.

Validates the output with Pydantic. Retries once with a correction prompt if validation fails.

**Why it exists:**
The deterministic ranking tells you WHAT the evidence supports and HOW STRONGLY. The LLM's job is to explain WHY — to write the biological narrative that connects the evidence to the molecular mechanism. This is what makes the output readable by a scientist who needs to evaluate it quickly.

The LLM is given strict grounding rules: use only the evidence provided, cite the sources, do not speculate, return JSON only. If the LLM cannot generate a grounded hypothesis for a biomarker, it is removed rather than fabricated.

The retry logic handles the common case where a model returns malformed JSON on the first attempt — a one-shot correction prompt resolves this in most cases.

---

## How RAG Works in This System

RAG stands for Retrieval-Augmented Generation. The idea is that instead of asking an AI to answer from memory (where it will hallucinate), you first retrieve relevant evidence and then ask the AI to synthesize only what was retrieved.

**How most RAG systems work:**
1. User asks a question
2. System searches a vector database for similar text chunks
3. Top-k chunks are passed to the LLM
4. LLM writes an answer based on those chunks

**How OncoMOA's RAG works:**
1. Drug + MOA are provided as input
2. System queries 15 specialized scientific APIs in parallel
3. Raw API responses are parsed into structured evidence objects (NormalizedEvidence)
4. Evidence objects are scored mathematically and ranked
5. Ranked evidence is formatted into a structured context
6. LLM writes hypotheses and rationales based only on that context

**Why this is stronger than standard RAG for science:**

Standard RAG retrieves text chunks from a generic corpus. The quality of the output depends entirely on whether the right chunks were retrieved, which depends on how well your vector similarity search worked. Irrelevant chunks contaminate the context. The LLM may blend them together incorrectly.

OncoMOA uses purpose-built retrieval from authoritative oncology databases. The evidence is structured before the LLM sees it. There is no vector similarity ambiguity — either a gene has a CIViC Level A predictive evidence record or it does not. The LLM cannot confuse different genes' evidence because they arrive as separate labeled records.

This makes the system significantly more reliable for high-stakes scientific use.

---

## Smart LLM Routing — Why Different Drugs Get Different Models

Not every drug gets the same AI model. The system routes based on drug complexity.

**Ollama / meditron → for standard targeted therapies**
meditron is a medical domain-specific instruction model. It runs locally on your machine, costs nothing, has no rate limits, and is fine-tuned on biomedical literature. For drugs with a single clear target (KRAS inhibitors, PARP inhibitors, EGFR inhibitors), this is sufficient.

Examples: sotorasib, olaparib, erlotinib, vemurafenib

**Gemini 1.5 Flash → for complex multi-target drugs**
Used for three drug classes where the biomarker landscape is fundamentally more complex:

**Merck IO drugs** (pembrolizumab, belzutifan, tepotinib): Immune checkpoint inhibitors operate through the tumor microenvironment. The biomarker landscape includes TMB (tumor mutational burden), MSI (microsatellite instability), PD-L1 expression, TIL (tumor infiltrating lymphocytes), HLA diversity, and co-mutation patterns. These interact non-linearly. Stronger reasoning is warranted.

**Bispecific antibodies** (ivonescimab targeting PD-1 + VEGF): These drugs hit two independent targets. The biomarker question becomes combinatorial — you need to consider angiogenesis biomarkers (VEGF, CD31) AND immune biomarkers (PD-L1, CD8) AND their interaction. This requires reasoning across two biological axes simultaneously.

**ADCs — Antibody-Drug Conjugates** (trastuzumab deruxtecan, sacituzumab govitecan, enfortumab vedotin): ADCs have three independent biomarker axes:
- Target antigen expression (HER2, TROP2, Nectin-4) — does the cancer have enough of the target for the antibody to bind?
- Linker stability — are there drug efflux pumps or enzymatic conditions that affect payload release?
- Payload toxicity — is there resistance to the chemotherapy payload?

Biomarker discovery for an ADC requires reasoning across all three axes. ADCs are detected automatically by drug name suffix (vedotin, deruxtecan, govitecan, emtansine, ozogamicin, mafodotin).

---

## The Evidence Grounding Gate

Before any hypothesis reaches the LLM, it must pass a gate:
- At least one database source (CIViC, cBioPortal, ClinicalTrials, etc.)
- At least one publication source (PubMed)

A biomarker with only database evidence and no publications is not allowed. A biomarker with only publications and no database evidence is not allowed. Both types of corroboration are required.

This rule eliminates low-confidence candidates before the LLM even sees them, keeping the synthesis step focused on defensible hypotheses.

---

## The 15 Data Sources — Full Explanations

Every database used in this system is free, public, and maintained by a major research institution. Here is what each one is, who runs it, what data it holds, and exactly what this system pulls from it.

---

### Tier 1 — Core Oncology Evidence

---

#### Open Targets Platform
**Who runs it:** European Bioinformatics Institute (EMBL-EBI), Wellcome Sanger Institute, GSK, and Pfizer — a public-private partnership.

**What it is:** A comprehensive drug-target-disease database that integrates evidence from genetics, genomics, clinical trials, literature, and pathways to quantify how strongly a gene is associated with a disease. It is one of the most authoritative resources in translational medicine.

**What data it holds:** Drug mechanisms of action, target gene lists per drug, disease-target association scores (0–1, higher = stronger evidence), linked diseases for each drug, genetic variant associations, pathway data, and safety evidence.

**What this system uses:**
- Drug name → target gene symbols (e.g., sotorasib → KRAS)
- Mechanism of action annotations
- Disease association scores for each target gene across hundreds of cancer types
- Linked diseases (which cancers a drug is approved or studied in)

**Why this specific database:** Open Targets aggregates 20+ evidence sources into a single score, which saves querying each source individually. Its drug-target data is curated at pharmaceutical grade — it is used internally by GSK and Pfizer for drug discovery decisions.

**API used:** GraphQL endpoint at `api.platform.opentargets.org/api/v4/graphql`

---

#### CIViC — Clinical Interpretation of Variants in Cancer
**Who runs it:** Washington University in St. Louis, with global expert curation community.

**What it is:** The only database built specifically for the clinical interpretation of cancer variants. Each entry is manually reviewed and graded by oncologists and clinical molecular pathologists. Think of it as the peer-reviewed authority on "does this mutation matter for this drug in this cancer."

**What data it holds:** Gene variants (e.g., KRAS G12C), evidence type (Predictive, Prognostic, Diagnostic), evidence direction (Supports, Does Not Support), drug, disease, evidence level (A through E), clinical significance, and supporting publications.

Evidence levels:
- **A** — Validated association from multiple clinical trials
- **B** — Evidence from clinical trials or well-powered studies
- **C** — Evidence from a small study or case reports
- **D** — Preclinical evidence (cell lines, animal models)
- **E** — Expert opinion or inferential

**What this system uses:**
- All Predictive and Prognostic evidence records for every candidate gene
- Evidence level (used as the primary scoring weight: A=5, B=4, C=3, D=2, E=1)
- Drug name and disease name from each record
- Supporting citation IDs

**Why this specific database:** CIViC is the gold standard for clinical biomarker evidence. A Level A CIViC entry means oncologists have accepted this biomarker in clinical practice. No other database provides this level of human expert curation for clinical variant interpretation.

**API used:** REST at `civicdb.org/api`

---

#### PubMed / NCBI E-utilities
**Who runs it:** National Center for Biotechnology Information (NCBI), part of the US National Institutes of Health (NIH).

**What it is:** The world's largest biomedical literature database. Over 37 million abstracts from journals in medicine, biology, chemistry, and life sciences. If a biomarker finding was published anywhere in the peer-reviewed literature, it is indexed here.

**What data it holds:** Article titles, abstracts, author lists, journal names, publication dates, MeSH terms, and citation links.

**What this system uses:**
- Search query: `"[drug name]" biomarker "[gene symbol]" cancer`
- Top 20 most relevant PMIDs per gene-drug pair
- Title and full abstract text for each PMID
- Used as a publication count signal in scoring (+0.1 per hit)

**Why this specific database:** CIViC and other curated databases lag behind the literature. A biomarker discovered in 2024 may not appear in CIViC until 2025 or 2026. PubMed catches those recent findings. For a new drug like sotorasib (approved 2021) or a next-generation ADC, the most important evidence may exist only in recent publications.

**API used:** NCBI E-utilities (esearch + efetch) at `eutils.ncbi.nlm.nih.gov/entrez/eutils`

---

#### PubTator Central
**Who runs it:** NCBI (same as PubMed).

**What it is:** A text mining system that runs NLP over PubMed abstracts and full-text articles to automatically identify and tag biomedical entities — genes, mutations, diseases, drugs, species. It converts unstructured text into structured annotations.

**What data it holds:** For each PMID, a list of entity mentions with type (Gene, Mutation, Disease, Chemical) and normalized identifier (e.g., NCBI Gene ID, OMIM ID).

**What this system uses:**
- Given the PMIDs from PubMed search, fetches PubTator annotations for each
- Extracts: genes mentioned, mutations mentioned, diseases mentioned, drugs mentioned
- Uses this to link papers to specific biomarker candidates
- Enriches each PubMedArticle object with structured entity lists

**Why this specific database:** Without PubTator, abstracts are just text. You cannot systematically link a paper to a gene without reading it. PubTator lets the system say "this paper mentions KRAS G12C and STK11" without any NLP code of its own. It turns 400 raw abstracts into 400 structured evidence records automatically.

**API used:** PubTator3 BioC JSON endpoint at `ncbi.nlm.nih.gov/research/pubtator3-api`

---

#### ClinicalTrials.gov
**Who runs it:** US National Library of Medicine (NLM), part of NIH. Registration is mandatory by law for most clinical trials conducted in the US.

**What it is:** The official registry for clinical trials. Every drug approved by the FDA went through trials that are listed here. The registry contains study protocols, eligibility criteria, primary and secondary endpoints, and results.

**What data it holds:** Trial ID (NCT number), title, phase, disease condition, interventions (drugs), eligibility criteria (inclusion and exclusion text), primary outcomes, secondary outcomes, status (recruiting, completed, etc.), and sponsor.

**What this system uses:**
- Searches for all trials involving the drug name
- Extracts full eligibility criteria text and scans for biomarker keywords (KRAS, PD-L1, TMB, BRCA, MSI, etc.)
- Identifies stratification factors and exploratory endpoints from secondary outcomes
- Phase is used in scoring: Phase III = +3, Phase II = +2, Phase I = +1

**Why this specific database:** Trial inclusion criteria are one of the most underused evidence signals in biomarker discovery. When a Phase III trial enrolls only patients with KRAS G12C mutations, that is the strongest possible real-world signal that KRAS G12C is the predictive biomarker for that drug — stronger than most published papers because it reflects regulatory-grade evidence.

**API used:** ClinicalTrials.gov v2 API at `clinicaltrials.gov/api/v2`

---

### Tier 2 — Gene and Variant Biology

---

#### Ensembl REST API
**Who runs it:** EMBL-EBI (same institution that co-runs Open Targets).

**What it is:** The canonical reference database for vertebrate genome annotation. Every human gene has an Ensembl ID (e.g., ENSG00000133703 for KRAS), which is the stable unique identifier used across genomics databases to avoid synonym confusion.

**What data it holds:** Gene coordinates (chromosome, start, end, strand), gene biotype (protein-coding, lncRNA, pseudogene), transcripts, exons, protein translations, and cross-references to other databases (HGNC, RefSeq, UniProt).

**What this system uses:**
- Gene symbol → Ensembl ID resolution (important for cross-database linking)
- Gene full name and description
- Canonical transcript ID
- Overlapping variants in phenotype-associated regions

**Why this specific database:** Genes have many names. KRAS is also called RASK, RASK2, K-RAS, and Ki-ras depending on the era and database. Ensembl provides the authoritative identifier that resolves these aliases. Without it, the system might miss evidence stored under a different gene name.

**API used:** REST at `rest.ensembl.org`

---

#### MyGene.info
**Who runs it:** Scripps Research Institute, funded by NIH.

**What it is:** A fast gene information API that aggregates gene metadata from dozens of sources (NCBI Gene, Ensembl, UniProt, KEGG, Reactome, GO, etc.) into a single queryable endpoint. Think of it as a gene metadata search engine.

**What data it holds:** Gene symbol, full name, aliases, summary text, pathway memberships (Reactome, KEGG, WikiPathways, BioCarta), Ensembl ID, UniProt accession, OMIM ID, Entrez Gene ID, GO terms.

**What this system uses:**
- Gene aliases — critical for resolving PDCD1 = PD-1 = CD279 and avoiding missed matches
- Pathway memberships — used to seed pathway expansion (which pathways does this gene belong to?)
- Gene summary text — used in target biology enrichment
- Batch query mode — can query 20+ genes in a single request, much faster than individual lookups

**Why this specific database:** Its batch query capability is uniquely valuable. Instead of 20 separate API calls to get pathway data for 20 genes, one batch call returns everything. This is critical for keeping the async pipeline fast.

**API used:** REST at `mygene.info/v3`

---

#### MyVariant.info
**Who runs it:** Scripps Research Institute (same team as MyGene.info).

**What it is:** A variant annotation API that aggregates information about specific genetic variants from ClinVar, CADD, DANN, dbNSFP, gnomAD, and other databases into a single queryable endpoint.

**What data it holds:** For any given variant (in HGVS notation): ClinVar clinical significance (Pathogenic, Likely Pathogenic, VUS, etc.), CADD score (computational deleteriousness), population frequency from gnomAD, conservation scores, and functional consequence predictions.

**What this system uses:**
- Queries ClinVar pathogenicity for variants in candidate genes
- Pathogenic variants in cancer-associated genes get added as evidence items
- Used to distinguish passenger mutations (not clinically relevant) from driver mutations (clinically relevant)

**Why this specific database:** Not all mutations in a gene are equal. A variant classified as Pathogenic in ClinVar is evidence that this specific change in the gene is known to affect protein function in a disease context. A VUS (variant of uncertain significance) is not. MyVariant lets the system make that distinction automatically.

**API used:** REST at `myvariant.info/v1`

---

#### DGIdb — Drug-Gene Interaction Database
**Who runs it:** The Griffith Lab at Washington University in St. Louis (same institution as CIViC).

**What it is:** A database of drug-gene interactions that aggregates evidence from PharmGKB, DrugBank, TTD (Therapeutic Target Database), ChEMBL, CIViC, and 20+ other sources. It focuses on interactions that are pharmacologically meaningful: inhibition, activation, binding, expression changes.

**What data it holds:** Drug name, gene name, interaction type (inhibitor, agonist, activator, inducer, etc.), interaction score, supporting publications, and source databases.

**What this system uses:**
- For every candidate gene, queries which drugs interact with it
- Used to find indirect pharmacological relationships — genes that interact with drugs in the same class as the target drug
- Interaction score used as a fractional evidence weight

**Why this specific database:** It surfaces genes that are pharmacologically connected to the drug class, even if not the direct target. For example, EGFR and MET both interact with tyrosine kinase inhibitors. If the drug is an EGFR inhibitor, MET's known interactions with similar drugs suggest it may be a resistance biomarker — which DGIdb would surface.

**API used:** GraphQL at `dgidb.org/api/graphql`

---

#### ChEMBL
**Who runs it:** EMBL-EBI (same institution as Ensembl and Open Targets).

**What it is:** A manually curated database of bioactive molecules and their biological targets. It contains data from drug discovery experiments — IC50 values, Ki values, binding assays — extracted from the scientific literature. It is used by pharmaceutical companies worldwide for target identification and lead optimization.

**What data it holds:** Drug compound information (structure, molecular formula, synonyms), bioactivity measurements (IC50 against specific targets), mechanism of action annotations, and approved drug indication data.

**What this system uses:**
- Drug name → ChEMBL molecule ID
- Molecule ID → mechanism of action records → target proteins → gene symbols
- Provides a second independent drug-target resolution path (alongside Open Targets)
- Action type (inhibitor, agonist, modulator) to characterize the drug-target relationship

**Why this specific database:** ChEMBL has the deepest bioactivity data of any public database. It is especially valuable for drugs with multiple targets (e.g., multi-kinase inhibitors) where Open Targets may only list the primary target but ChEMBL shows all targets with measurable binding affinity.

**API used:** REST at `ebi.ac.uk/chembl/api/data`

---

#### UniProt
**Who runs it:** Universal Protein Resource — a consortium of EMBL-EBI, SIB Swiss Institute of Bioinformatics, and PIR.

**What it is:** The world's most comprehensive protein database. The Swiss-Prot section (reviewed) contains manually curated entries for every well-characterized human protein, with detailed annotations of function, structure, interactions, and disease relevance. It is considered the gold standard for protein-level information.

**What data it holds:** Protein function description, subcellular localization, post-translational modifications, known interaction partners, pathway involvement, disease associations (linked to OMIM), tissue expression, and structural features.

**What this system uses:**
- Protein function text — provides mechanistic language for the target's role in cancer biology
- Disease associations — which diseases are linked to mutations in this protein
- Pathway keywords — supplements pathway expansion with GO biological process terms
- Protein accession (P01116 for KRAS) — used for cross-database linking

**Why this specific database:** MyGene provides pathway names. UniProt provides the biological explanation of why a protein is in that pathway — what it actually does, what domains it has, what it binds, where it sits in the cell. This is the information the LLM needs to write accurate mechanistic rationales.

**API used:** REST at `rest.uniprot.org`

---

### Tier 3 — Pathway Biology

---

#### Reactome Content Service
**Who runs it:** Ontario Institute for Cancer Research (OICR) and EMBL-EBI, in collaboration with New York University.

**What it is:** A peer-reviewed, expert-authored database of biological pathways. Unlike automated pathway databases, every Reactome pathway is manually reviewed by domain experts and published in peer-reviewed journals. It covers signaling, metabolism, gene expression, DNA repair, cell cycle, and immune responses in human biology.

**What data it holds:** Hierarchical pathway descriptions from high-level (e.g., "Signal Transduction") to granular reactions (e.g., "GTP binds to RAS-GDP"). Each pathway includes participant genes, upstream/downstream relationships, and literature support.

**What this system uses:**
- Given a target gene, finds all pathways it participates in
- For each pathway, retrieves the member genes (other genes in the same biological process)
- These member genes become candidate biomarker genes via pathway expansion
- Pathway name and ID are added as nodes in the knowledge graph

**Why this specific database:** Reactome is the highest quality pathway database available publicly. The expert curation means the pathway gene lists are biologically accurate and conserved — if KRAS participates in the MAPK pathway, Reactome will list exactly the right downstream effectors.

**API used:** REST at `reactome.org/ContentService`

---

#### WikiPathways
**Who runs it:** Maastricht University, Gladstone Institutes, and a global community of biologists. Originally inspired by the Wikipedia model — open, community-contributed, peer-reviewed.

**What it is:** A community-curated biological pathway database with a focus on human diseases. While Reactome emphasizes reaction-level detail, WikiPathways emphasizes pathway-level gene lists and disease relevance. It has particularly strong coverage of cancer signaling pathways.

**What data it holds:** Pathway diagrams (as structured data), gene participants, pathway descriptions, disease annotations, and links to literature.

**What this system uses:**
- Searches for pathways containing a target gene
- Returns pathway member genes as additional candidates
- Complements Reactome — some pathways are better curated in WikiPathways (e.g., specific oncogenic signaling cascades)

**Why this specific database:** WikiPathways and Reactome have different coverage. For some cancer-specific pathways (e.g., the full KRAS → RAS → RAF → MEK → ERK cascade with all known feedback loops), WikiPathways may have more complete gene lists than Reactome. Using both maximizes coverage.

**API used:** REST at `webservice.wikipathways.org`

---

### Tier 4 — Cancer Genomics Frequency Data

---

#### cBioPortal
**Who runs it:** Memorial Sloan Kettering Cancer Center (MSK) and Dana-Farber Cancer Institute, funded by NCI.

**What it is:** A web platform for exploring multidimensional cancer genomics data from TCGA, AACR GENIE, and dozens of institutional cohorts. It is the standard tool used by oncology researchers worldwide to look up mutation frequencies, co-occurrence patterns, and survival correlations.

**What data it holds:** Somatic mutation frequencies by gene and cancer type, copy number alterations, structural variants, mRNA expression data, protein expression, co-mutation patterns, and overall survival correlations — across 700+ studies and 150,000+ tumor samples.

**What this system uses:**
- For each candidate gene, queries mutation frequency across major TCGA cancer types (LUAD, BRCA, COADREAD, PAAD, etc.)
- Mutation count per study used as evidence strength (proportional to how many patients carry this alteration)
- Converted to NormalizedEvidence items with source_id linking back to the specific TCGA study

**Why this specific database:** CIViC tells you whether a mutation matters clinically. cBioPortal tells you how common it is. A mutation that is clinically meaningful but occurs in only 0.1% of patients has very limited utility for patient selection. A mutation that occurs in 40% of LUAD patients (like KRAS in lung adenocarcinoma) has high clinical utility. Frequency data is essential for prioritization.

**API used:** REST at `cbioportal.org/api`

---

#### GDC — Genomic Data Commons / TCGA
**Who runs it:** National Cancer Institute (NCI), part of NIH.

**What it is:** The central repository for TCGA (The Cancer Genome Atlas) data — the largest coordinated effort to characterize the genomic landscape of human cancer. TCGA profiled ~11,000 tumor samples across 33 cancer types using WGS, RNA-seq, methylation arrays, and protein expression assays.

**What data it holds:** Somatic mutation calls (MAF files), mRNA expression data, copy number profiles, methylation data, miRNA data, protein expression (RPPA), and clinical outcome data — all harmonized to a common pipeline and reference genome.

**What this system uses:**
- Queries somatic mutation frequency for each candidate gene across TCGA studies
- Retrieves consequence type distribution (missense, nonsense, frameshift, etc.)
- Cancer type distribution of mutations — which cancer types have the most mutations in this gene
- Used as independent corroboration of cBioPortal frequency data

**Why this specific database:** GDC is the primary source; cBioPortal uses GDC data in its backend. Querying both provides independent verification and sometimes returns different granularity. GDC's direct API also allows custom filters (e.g., somatic-only, specific consequence types) that are not always available through cBioPortal's interface.

**API used:** REST at `api.gdc.cancer.gov`

---

## How the Databases Work Together

Here is how all 15 sources combine for a single drug query. Using sotorasib as an example:

**Step 1 — Drug resolution:**
Open Targets + ChEMBL both confirm: sotorasib targets KRAS. Open Targets also links it to NSCLC, PDAC, CRC.

**Step 2 — Target enrichment:**
Ensembl gives KRAS its canonical ID (ENSG00000133703). MyGene confirms aliases (RASK, KRAS2) and lists Reactome pathway memberships. UniProt explains KRAS is a GTPase in the RAS superfamily that activates MAPK and PI3K signaling.

**Step 3 — Pathway expansion:**
Reactome: KRAS participates in RAS signaling, MAPK cascade, PI3K signaling → yields NRAS, BRAF, RAF1, MAP2K1, MAPK1, PIK3CA, AKT1. WikiPathways: same pathway, additionally includes STK11, KEAP1, CDKN2A from cancer-specific pathway annotations.

**Step 4 — Clinical evidence:**
CIViC: KRAS G12C has Level A predictive evidence for sotorasib in NSCLC. STK11 has Level B evidence as a resistance biomarker. KEAP1 has Level C evidence.
cBioPortal: KRAS mutated in 32% of LUAD, 88% of PDAC. STK11 mutated in 17% of LUAD (co-occurring with KRAS in 10% of cases).
GDC: Confirms KRAS mutation frequency, adds consequence type distribution.
DGIdb: BRAF interacts with multiple kinase inhibitors in the same class — surfaced as candidate.
MyVariant: KRAS G12C is ClinVar Pathogenic. STK11 frameshift variants are Pathogenic.
ClinicalTrials.gov: 8 trials for sotorasib include KRAS G12C as mandatory inclusion criterion. 3 trials stratify by STK11 status.

**Step 5 — Literature:**
PubMed: 14 papers for sotorasib + KRAS, 6 for sotorasib + STK11. PubTator extracts: STK11 and KEAP1 both mentioned in 3 of the STK11 papers as co-occurring resistance factors.

**Step 6 — Knowledge graph:**
Graph connects: Sotorasib → targets → KRAS → participates_in → MAPK Pathway → STK11 → tested_in → NCT04303780 → associated_with → NSCLC.

**Step 7 — Scoring:**
KRAS G12C: CIViC A (5) + Phase III trial (3) + 14 PubMed hits (1.4) + direct target (3) = 12.4 → normalized to 96/100.
STK11: CIViC B (4) + Phase II stratification (2) + 6 PubMed hits (0.6) + pathway connected (2) = 8.6 → normalized to 72/100.

**Step 8 — LLM:**
Receives the scored, structured context. Writes: "STK11 loss-of-function mutations co-occur with KRAS G12C in approximately 10% of NSCLC cases and are associated with immune exclusion and resistance to sotorasib. CIViC Level B evidence and stratification in two Phase II trials support STK11 as a predictive resistance biomarker."

---

## Output Files

| File | What It Contains |
|---|---|
| output/results.json | Complete ranked hypotheses — every field, every score, every citation |
| output/evidence_summary.csv | Spreadsheet version — easy to open in Excel or share |
| output/knowledge_graph.graphml | Graph file — open in Cytoscape or Gephi for visualization |
| output/knowledge_graph.json | JSON graph — for web visualization (D3.js, etc.) |
| logs/oncomoa.log | Full structured log: every API call, latency, success/failure |

---

## What Makes This Production-Grade

**Async parallel execution:** All API calls within each stage run concurrently using asyncio and aiohttp. Without this, a single run querying 15 sources for 20+ genes would take 10+ minutes. With it, most evidence collection completes in under 60 seconds.

**Graceful degradation:** Every API call is wrapped in error handling. If CIViC returns a 500 error, the system logs it, marks CIViC as a failed source, and continues with the remaining 14 sources. No single failure terminates the pipeline.

**Persistent caching:** All API responses are cached for 7 days. Running the same drug twice in a week costs zero API calls. This also makes the system reproducible — the same input produces the same output within the cache window.

**Pydantic validation throughout:** Every data object — evidence items, biomarker hypotheses, agent outputs — is validated against a strict schema. Type errors, missing fields, and out-of-range scores are caught at the point of entry, not silently passed downstream.

**Retry logic:** The LLM synthesis step retries once with a correction prompt if the output fails JSON parsing or Pydantic validation. If both attempts fail, the system returns the deterministic ranking without narrative — which is still scientifically valid output.

**Source attribution:** Every hypothesis lists exactly which databases and publications support it. This is mandatory for scientific credibility. A scientist reviewing the output can immediately trace each hypothesis back to its evidence.

---

## Test Coverage — 42 Tests

| Test Group | What It Covers |
|---|---|
| TestCivicWeights | CIViC level A–E weights are correct and descending |
| TestTrialWeights | Phase III/II/I weights are correct, aliases work |
| TestBiomarkerClassification | KRAS G12C → mutation, TMB → immune signature, EML4-ALK → fusion |
| TestResponseDirection | Positive vs resistance direction inferred from evidence text |
| TestRankingEngine | Direct target bonus, score normalization, top-N cap, predictive/prognostic separation |
| TestBiomarkerHypothesisSchema | Score clamping, defaults, enum validation |
| TestNormalizedEvidenceSchema | All fields, default evidence type |
| TestAgentOutputSchema | Valid output, empty hypotheses, source tracking |
| TestCandidateBiomarkerGate | DB + pub required, DB-only fails, pub-only fails |
| TestDrugRouting | Merck → Gemini, Amgen → Ollama, ADC suffixes → Gemini |
| TestKnowledgeGraph | Node/edge creation, edge relations, GraphML export, JSON export |

---

## Interview Talking Points

**On the architecture:**
"The multi-agent design enforces a strict separation between retrieval and generation. Each agent is independently testable and replaceable. The LLM only synthesizes — it never discovers. This is what makes the output scientifically defensible."

**On RAG:**
"This is domain-specific RAG. Instead of generic vector similarity search, we do purpose-built retrieval from authoritative oncology databases. The evidence arrives structured and scored before the LLM sees it. There is no ambiguity about which gene's evidence belongs to which record."

**On the knowledge graph:**
"The graph is not just a visualization tool — it is a reasoning layer. It allows neighborhood expansion: discovering genes that are biologically connected to the drug target but would not appear in a simple keyword search. This is what generates non-obvious biomarker hypotheses."

**On LLM routing:**
"We route to different models based on drug complexity. ADCs have three independent biomarker axes — target antigen, linker stability, and payload toxicity. Bispecifics require reasoning across two targets simultaneously. These need stronger reasoning than a standard KRAS inhibitor."

**On deterministic scoring:**
"The ranking engine has no AI in it. It is a mathematical formula with explicit, auditable weights. A biomarker's confidence score can be fully explained: it came from a Level A CIViC finding (5 points), two Phase II trials (4 points), and 14 PubMed hits (1.4 points). This matters in clinical settings where you need to justify your reasoning."

**On cost:**
"The system operates at near-zero cost. All data sources are free public APIs. All responses are cached for 7 days. For most drugs, Ollama runs the LLM locally for free. Gemini is only invoked for specific complex drug classes, and even then it is called exactly once per run."

---

## Outputs and graph visualization

Each run writes these artifacts to `./output/` unless `--output` is provided:

| File | Purpose |
|---|---|
| `results.json` | Ranked hypotheses, source status, and knowledge-graph summary |
| `evidence_summary.csv` | Tabular evidence and ranking export |
| `knowledge_graph.graphml` | Graph export for Cytoscape or Gephi |
| `knowledge_graph.json` | Portable graph nodes and edges for programmatic inspection |

To inspect the graph in Cytoscape, import `knowledge_graph.graphml` using
**File → Import → Network from File**. A graph with zero evidence may contain
only the drug node; this is expected fail-closed output, not a biomarker result.

## Quick Reference Commands

```bash
# Standard targeted therapy (Ollama/meditron, local, free)
python3.10 main.py \
  --drug "sotorasib" \
  --moa "Covalent KRAS G12C inhibitor"

# IO checkpoint inhibitor (auto-routes to Gemini)
python3.10 main.py \
  --drug "pembrolizumab" \
  --moa "PD-1 checkpoint inhibitor" \
  --top-n 15

# ADC (auto-routes to Gemini by 'deruxtecan' suffix)
python3.10 main.py \
  --drug "trastuzumab deruxtecan" \
  --moa "HER2-directed antibody-drug conjugate"

# Skip LLM entirely — deterministic ranking only (fastest)
python3.10 main.py \
  --drug "olaparib" \
  --moa "PARP1/2 inhibitor" \
  --no-llm

# Write a run to an isolated directory
python3.10 main.py \
  --drug "pembrolizumab" \
  --moa "PD-1 checkpoint inhibitor" \
  --no-llm \
  --output output/pembrolizumab_run

# Force a specific backend regardless of drug routing
python3.10 main.py \
  --drug "sotorasib" \
  --moa "KRAS G12C inhibitor" \
  --backend gemini

# Run all unit tests
python3.10 -m pytest tests/test_scoring.py tests/test_schemas.py tests/test_pipeline_safety.py -v

# Run full integration tests (requires network)
python3.10 -m pytest tests/test_integration.py -v
```
