"""
OncoMOA — Knowledge Graph Layer
Constructs a directed NetworkX graph from multi-source evidence.
Supports neighborhood expansion, drug→target→pathway→biomarker chain discovery,
and export to GraphML and JSON.

Example:
    kg = OncologyKnowledgeGraph()
    kg.add_drug("sotorasib")
    kg.add_gene_target("sotorasib", "KRAS")
    kg.export_graphml("output/knowledge_graph.graphml")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx

from models.schemas import (
    NormalizedEvidence,
    PathwayInfo,
    ClinicalTrialInfo,
    PubMedArticle,
    KnowledgeGraphSummary,
    NodeType,
)

logger = logging.getLogger(__name__)


class OncologyKnowledgeGraph:
    """
    Directed knowledge graph for oncology biomarker discovery.

    Node types: Drug, Gene, Mutation, Disease, Pathway, ClinicalTrial, Biomarker
    Edge types: targets, participates_in, associated_with, predicts_response,
                predicts_resistance, mentioned_in, tested_in
    """

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()

    # ─── Node Helpers ────────────────────────────────────────────────────────

    def _add_node(self, node_id: str, node_type: NodeType, **attrs: Any) -> None:
        """Add a node to the graph with type metadata."""
        self.graph.add_node(node_id, node_type=node_type.value, **attrs)

    def _add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        weight: float = 1.0,
        **attrs: Any,
    ) -> None:
        """Add a directed edge between two nodes."""
        self.graph.add_edge(source, target, relation=relation, weight=weight, **attrs)

    # ─── Domain Add Methods ───────────────────────────────────────────────────

    def add_drug(self, drug_name: str, moa: str = "", **attrs: Any) -> None:
        """Add a drug node."""
        self._add_node(drug_name, NodeType.DRUG, label=drug_name, moa=moa, **attrs)

    def add_gene_target(self, drug_name: str, gene_symbol: str, weight: float = 3.0) -> None:
        """Add a gene node and a drug→gene 'targets' edge."""
        self._add_node(gene_symbol, NodeType.GENE, label=gene_symbol)
        self._add_edge(drug_name, gene_symbol, relation="targets", weight=weight)

    def add_gene(self, gene_symbol: str, summary: str = "", **attrs: Any) -> None:
        """Add or update a gene node without creating drug-target edges."""
        if not self.graph.has_node(gene_symbol):
            self._add_node(gene_symbol, NodeType.GENE, label=gene_symbol, summary=summary, **attrs)
        elif summary:
            self.graph.nodes[gene_symbol]["summary"] = summary

    def add_mutation(self, gene_symbol: str, variant: str, weight: float = 2.0) -> None:
        """Add a mutation node and a gene→mutation edge."""
        mut_id = f"{gene_symbol}:{variant}"
        self._add_node(mut_id, NodeType.MUTATION, label=variant, gene=gene_symbol)
        self.add_gene(gene_symbol)
        self._add_edge(gene_symbol, mut_id, relation="has_variant", weight=weight)

    def add_pathway(self, gene_symbol: str, pathway: PathwayInfo) -> None:
        """Add a pathway node and link the gene to it."""
        pw_id = f"PW:{pathway.pathway_id}"
        self._add_node(
            pw_id,
            NodeType.PATHWAY,
            label=pathway.pathway_name,
            source=pathway.source,
        )
        self.add_gene(gene_symbol)
        self._add_edge(gene_symbol, pw_id, relation="participates_in", weight=1.0)

        # Add all pathway member genes and link to pathway
        for member_gene in pathway.genes:
            if member_gene and member_gene != gene_symbol:
                self.add_gene(member_gene)
                self._add_edge(member_gene, pw_id, relation="participates_in", weight=0.5)

    def add_disease_association(
        self, gene_symbol: str, disease: str, score: float = 1.0
    ) -> None:
        """Add a disease node and gene→disease 'associated_with' edge."""
        self._add_node(disease, NodeType.DISEASE, label=disease)
        self.add_gene(gene_symbol)
        self._add_edge(gene_symbol, disease, relation="associated_with", weight=score)

    def add_clinical_trial(self, trial: ClinicalTrialInfo, target_genes: list[str]) -> None:
        """Add a clinical trial node and link it to genes mentioned."""
        trial_id = trial.trial_id
        self._add_node(
            trial_id,
            NodeType.CLINICAL_TRIAL,
            label=trial.title[:80] if trial.title else trial_id,
            phase=trial.phase or "",
            drug=trial.drug or "",
        )
        for gene in target_genes:
            if gene in trial.biomarker_mentions or any(
                gene.upper() in m.upper() for m in trial.biomarker_mentions
            ):
                self.add_gene(gene)
                self._add_edge(gene, trial_id, relation="tested_in", weight=1.0)

    def add_evidence(self, evidence: NormalizedEvidence) -> None:
        """
        Add a normalized evidence item to the graph.
        Creates appropriate nodes and edges based on evidence type.
        """
        gene = evidence.gene
        if not gene:
            return

        self.add_gene(gene)

        if evidence.variant:
            self.add_mutation(gene, evidence.variant, weight=evidence.strength)

        if evidence.disease:
            self.add_disease_association(gene, evidence.disease, score=evidence.strength)

        # For predictive evidence, add biomarker node
        from models.schemas import EvidenceType
        if evidence.evidence_type in (EvidenceType.PREDICTIVE, EvidenceType.PROGNOSTIC):
            bm_label = f"{gene} {evidence.variant}" if evidence.variant else gene
            bm_id = f"BM:{bm_label}"
            self._add_node(
                bm_id,
                NodeType.BIOMARKER,
                label=bm_label,
                evidence_type=evidence.evidence_type.value,
                source=evidence.source,
            )
            relation = (
                "predicts_response"
                if evidence.evidence_type == EvidenceType.PREDICTIVE
                else "predicts_prognosis"
            )
            self._add_edge(gene, bm_id, relation=relation, weight=evidence.strength)

    def add_pubmed_article(
        self, article: PubMedArticle, target_genes: list[str]
    ) -> None:
        """Add literature mentions linking genes to PubMed articles."""
        pmid_node = f"PMID:{article.pmid}"
        self._add_node(
            pmid_node,
            NodeType.BIOMARKER,
            label=article.title[:60] if article.title else article.pmid,
        )
        all_genes = set(article.genes_mentioned) & set(target_genes)
        for gene in all_genes:
            self.add_gene(gene)
            self._add_edge(gene, pmid_node, relation="mentioned_in", weight=0.1)

    # ─── Graph Analysis ────────────────────────────────────────────────────────

    def get_drug_target_chains(self, drug_name: str, max_depth: int = 4) -> list[list[str]]:
        """
        Find drug → target → pathway → mutation chains up to max_depth hops.

        Args:
            drug_name: Starting drug node.
            max_depth: Maximum path length.

        Returns:
            List of node-id paths.
        """
        if not self.graph.has_node(drug_name):
            return []

        chains: list[list[str]] = []
        for target in list(self.graph.successors(drug_name)):
            edge_data = self.graph.edges[drug_name, target]
            if edge_data.get("relation") != "targets":
                continue
            # BFS from target: find paths to any successor node
            successors = list(self.graph.successors(target))
            for dest in successors[:10]:
                try:
                    paths = list(
                        nx.all_simple_paths(self.graph, target, dest, cutoff=max_depth - 1)
                    )
                    for path in paths[:5]:
                        chains.append([drug_name] + path)
                except Exception:
                    continue

        return chains[:50]

    def get_candidate_biomarkers_from_graph(
        self, drug_name: str, min_degree: int = 1
    ) -> list[str]:
        """
        Extract candidate biomarker gene symbols via neighborhood expansion from drug.

        Returns genes reachable from the drug node within 3 hops.
        """
        if not self.graph.has_node(drug_name):
            return []

        candidates: list[str] = []
        try:
            neighbors = nx.single_source_shortest_path_length(
                self.graph, drug_name, cutoff=3
            )
            for node, distance in neighbors.items():
                if (
                    distance >= 1
                    and self.graph.nodes[node].get("node_type") == NodeType.GENE.value
                    and node != drug_name
                ):
                    candidates.append(node)
        except Exception as exc:
            logger.error("KG candidate extraction failed: %s", exc)

        return candidates

    def get_top_connected_genes(self, top_n: int = 10) -> list[str]:
        """Return gene nodes sorted by in-degree (most evidence connections)."""
        gene_degrees = [
            (node, self.graph.in_degree(node))
            for node, data in self.graph.nodes(data=True)
            if data.get("node_type") == NodeType.GENE.value
        ]
        gene_degrees.sort(key=lambda x: x[1], reverse=True)
        return [g for g, _ in gene_degrees[:top_n]]

    def build_summary(self) -> KnowledgeGraphSummary:
        """Generate a summary of the knowledge graph."""
        chains = []
        drug_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == NodeType.DRUG.value
        ]
        for drug in drug_nodes[:3]:
            chains.extend(self.get_drug_target_chains(drug, max_depth=3))

        candidates: list[str] = []
        for drug in drug_nodes:
            candidates.extend(self.get_candidate_biomarkers_from_graph(drug))
        candidates = list(dict.fromkeys(candidates))

        return KnowledgeGraphSummary(
            node_count=self.graph.number_of_nodes(),
            edge_count=self.graph.number_of_edges(),
            drug_target_chains=[c[:6] for c in chains[:10]],
            candidate_biomarkers=candidates[:20],
            top_connected_genes=self.get_top_connected_genes(top_n=10),
        )

    # ─── Export ───────────────────────────────────────────────────────────────

    def export_graphml(self, path: str | Path) -> None:
        """Export the knowledge graph to GraphML format."""
        try:
            nx.write_graphml(self.graph, str(path))
            logger.info("KG exported to GraphML: %s (%d nodes, %d edges)",
                        path, self.graph.number_of_nodes(), self.graph.number_of_edges())
        except Exception as exc:
            logger.error("KG GraphML export failed: %s", exc)

    def export_json(self, path: str | Path) -> None:
        """Export the knowledge graph to node-link JSON format."""
        try:
            data = nx.node_link_data(self.graph, edges="links")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info("KG exported to JSON: %s", path)
        except Exception as exc:
            logger.error("KG JSON export failed: %s", exc)

    def export_all(
        self,
        graphml_path: str | Path,
        json_path: str | Path,
    ) -> None:
        """Export to both GraphML and JSON."""
        self.export_graphml(graphml_path)
        self.export_json(json_path)
