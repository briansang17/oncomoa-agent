"""
OncoMOA Agent — Literature Agent
Aggregates PubMed literature and PubTator entity annotations for
target and candidate genes. Converts to NormalizedEvidence.

Example:
    agent = LiteratureAgent()
    evidence, articles = await agent.run(["KRAS", "STK11"], drug_name="sotorasib")
"""

from __future__ import annotations

import asyncio
import logging

from models.schemas import NormalizedEvidence, PubMedArticle, EvidenceType, EvidenceDirection
from tools.pubmed import search_pubmed_bulk
from tools.pubtator import get_enriched_articles

logger = logging.getLogger(__name__)


class LiteratureAgent:
    """
    Retrieves and structures literature evidence from PubMed + PubTator.
    """

    async def run(
        self,
        gene_symbols: list[str],
        drug_name: str,
        max_per_gene: int = 5,
    ) -> tuple[list[NormalizedEvidence], list[PubMedArticle]]:
        """
        Search PubMed for drug-gene-cancer biomarker papers and enrich with PubTator.

        Args:
            gene_symbols: Target and candidate genes.
            drug_name: Drug name for search queries.
            max_per_gene: Max articles per gene.

        Returns:
            Tuple of (NormalizedEvidence list, PubMedArticle list with entity annotations).
        """
        logger.info(
            "[LiteratureAgent] Searching PubMed for %d genes + drug '%s'",
            len(gene_symbols),
            drug_name,
        )

        # Parallel PubMed searches per gene
        gene_articles_map = await search_pubmed_bulk(
            drug_name=drug_name,
            gene_symbols=gene_symbols[:15],
            max_per_gene=max_per_gene,
        )

        # Flatten all articles
        all_articles: list[PubMedArticle] = []
        gene_article_counts: dict[str, int] = {}
        article_query_genes: dict[str, list[str]] = {}

        for gene, articles in gene_articles_map.items():
            gene_article_counts[gene] = len(articles)
            all_articles.extend(articles)
            for article in articles:
                article_query_genes.setdefault(article.pmid, []).append(gene)

        # Deduplicate by PMID
        seen_pmids: set[str] = set()
        unique_articles: list[PubMedArticle] = []
        for article in all_articles:
            if article.pmid not in seen_pmids:
                seen_pmids.add(article.pmid)
                unique_articles.append(article)

        logger.info("[LiteratureAgent] %d unique articles; enriching with PubTator...", len(unique_articles))

        # Enrich with PubTator entity annotations
        try:
            enriched_articles = await get_enriched_articles(unique_articles)
        except Exception as exc:
            logger.warning("[LiteratureAgent] PubTator enrichment failed: %s", exc)
            enriched_articles = unique_articles

        # Convert to NormalizedEvidence
        evidence_items = self._articles_to_evidence(
            enriched_articles,
            article_query_genes,
            drug_name,
        )

        logger.info(
            "[LiteratureAgent] Generated %d evidence items from %d articles",
            len(evidence_items),
            len(enriched_articles),
        )
        return evidence_items, enriched_articles

    def _articles_to_evidence(
        self,
        articles: list[PubMedArticle],
        article_query_genes: dict[str, list[str]],
        drug_name: str,
    ) -> list[NormalizedEvidence]:
        """
        Convert PubMedArticle objects to NormalizedEvidence.

        The PubMed query already links each article to a candidate gene. Preserve
        that association when PubTator has no annotations instead of emitting an
        unlinked record that the evidence gate cannot evaluate.
        """
        evidence_items: list[NormalizedEvidence] = []

        for article in articles:
            query_genes = list(dict.fromkeys(article_query_genes.get(article.pmid, [])))
            genes = article.genes_mentioned[:5] if article.genes_mentioned else []
            attributed_genes = query_genes or genes[:1]
            mutations = article.mutations_mentioned[:3]

            variant = mutations[0] if mutations else None

            claim = article.title or f"PMID:{article.pmid}"
            if article.abstract:
                claim = f"{claim}. {article.abstract[:200]}..."

            for gene in attributed_genes:
                evidence_items.append(
                    NormalizedEvidence(
                        source="PubMed",
                        source_id=f"PMID:{article.pmid}",
                        gene=gene,
                        variant=variant,
                        drug=drug_name,
                        evidence_type=EvidenceType.PREDICTIVE,
                        evidence_direction=EvidenceDirection.SUPPORTS,
                        claim=claim[:500],
                        strength=0.1,
                        raw_data={
                            "pmid": article.pmid,
                            "title": article.title,
                            "query_genes": query_genes,
                            "genes": genes,
                            "mutations": mutations,
                            "diseases": article.diseases_mentioned[:3],
                        },
                    )
                )

        return evidence_items
