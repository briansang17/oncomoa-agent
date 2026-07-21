"""
OncoMOA Biomarker Agent — Main CLI Entrypoint

Runs the full oncology biomarker discovery pipeline and outputs:
  - Rich table display in terminal
  - output/results.json
  - output/knowledge_graph.graphml
  - output/knowledge_graph.json
  - output/evidence_summary.csv
  - logs/oncomoa.log

Example:
    python main.py --drug "sotorasib" --moa "Covalent KRAS G12C inhibitor"
    python main.py --drug "pembrolizumab" --moa "PD-1 checkpoint inhibitor" --top-n 15
    python main.py --drug "olaparib" --moa "PARP1/2 inhibitor" --backend ollama
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich import print as rprint

from config import (
    LOG_FILE,
    LOG_LEVEL,
    OUTPUT_RESULTS_JSON,
    OUTPUT_EVIDENCE_CSV,
    DEFAULT_TOP_N,
)
from models.schemas import AgentOutput, BiomarkerHypothesis

console = Console()


def setup_logging() -> None:
    """Configure structured logging to both file and Rich console."""
    log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    # File handler (structured JSON-like)
    LOG_FILE.parent.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    # Rich console handler (INFO+)
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_path=False,
        markup=True,
    )
    rich_handler.setLevel(log_level)

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[file_handler, rich_handler],
        force=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="oncomoa",
        description="OncoMOA Biomarker Agent — Agentic oncology biomarker discovery platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --drug "sotorasib" --moa "Covalent KRAS G12C inhibitor"
  python main.py --drug "pembrolizumab" --moa "PD-1 checkpoint inhibitor" --top-n 15
  python main.py --drug "trastuzumab deruxtecan" --moa "HER2-directed ADC" --backend gemini
  python main.py --drug "olaparib" --moa "PARP1/2 inhibitor" --backend ollama --top-n 8
        """,
    )
    parser.add_argument(
        "--drug", "-d", required=True, help="Drug name (e.g., 'sotorasib')"
    )
    parser.add_argument(
        "--moa", "-m", required=True, help="Mechanism of action description"
    )
    parser.add_argument(
        "--top-n", "-n", type=int, default=DEFAULT_TOP_N,
        help=f"Number of biomarker hypotheses to return (default: {DEFAULT_TOP_N})"
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["auto", "gemini", "ollama", "meditron"],
        default="auto",
        help="LLM backend override (default: auto — uses drug-class routing)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Custom output directory (default: ./output/)"
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM synthesis step, return deterministic ranking only"
    )
    return parser.parse_args()


def display_results_table(output: AgentOutput) -> None:
    """Render biomarker hypotheses as a Rich table in the terminal."""
    console.print()
    console.print(Panel(
        f"[bold green]OncoMOA Results[/bold green]\n"
        f"Drug: [cyan]{output.drug_name}[/cyan] | "
        f"Targets: [yellow]{', '.join(output.target_genes[:5])}[/yellow] | "
        f"Evidence items: [magenta]{output.total_evidence_items}[/magenta] | "
        f"LLM: [blue]{output.llm_backend_used}[/blue]",
        expand=False,
    ))

    if not output.hypotheses:
        console.print("[red]No biomarker hypotheses generated.[/red]")
        return

    table = Table(
        show_header=True,
        header_style="bold white on dark_blue",
        border_style="blue",
        title=f"[bold]Top {len(output.hypotheses)} Biomarker Hypotheses[/bold]",
        title_style="bold cyan",
    )

    table.add_column("#", style="bold white", width=3)
    table.add_column("Biomarker", style="bold yellow", width=20)
    table.add_column("Type", style="cyan", width=12)
    table.add_column("Category", style="green", width=16)
    table.add_column("Direction", style="magenta", width=10)
    table.add_column("Conf.", style="bold", width=6)
    table.add_column("Level", style="red", width=6)
    table.add_column("Evidence", style="dim", width=30)

    for hyp in output.hypotheses:
        # Color confidence score
        conf = hyp.confidence_score
        if conf >= 70:
            conf_str = f"[bold green]{conf:.0f}[/bold green]"
        elif conf >= 40:
            conf_str = f"[yellow]{conf:.0f}[/yellow]"
        else:
            conf_str = f"[red]{conf:.0f}[/red]"

        level_str = f"[bold red]{hyp.evidence_level}[/bold red]" if hyp.evidence_level else "-"
        evidence_str = ", ".join(hyp.supporting_sources[:2]) if hyp.supporting_sources else "—"

        table.add_row(
            str(hyp.rank),
            hyp.biomarker,
            hyp.biomarker_type.value,
            hyp.biomarker_category.value,
            hyp.direction.value,
            conf_str,
            level_str,
            evidence_str[:30],
        )

    console.print(table)

    # Print hypotheses narratives
    if any(h.hypothesis and "unavailable" not in h.hypothesis for h in output.hypotheses):
        console.print()
        console.print("[bold cyan]Biomarker Narratives:[/bold cyan]")
        for hyp in output.hypotheses[:5]:
            if hyp.hypothesis:
                console.print(f"\n[bold yellow]#{hyp.rank} {hyp.biomarker}[/bold yellow]")
                console.print(f"  {hyp.hypothesis}")

    # Print KG summary
    kg = output.knowledge_graph_summary
    console.print()
    console.print(Panel(
        f"[bold]Knowledge Graph[/bold]: {kg.node_count} nodes, {kg.edge_count} edges\n"
        f"Top connected genes: {', '.join(kg.top_connected_genes[:8])}\n"
        f"Candidate biomarkers: {', '.join(kg.candidate_biomarkers[:8])}",
        title="KG Summary",
        expand=False,
    ))

    # Sources report
    console.print()
    if output.successful_sources:
        console.print(f"[green]Successful sources:[/green] {', '.join(set(output.successful_sources))}")
    if output.failed_sources:
        console.print(f"[red]Failed sources:[/red] {', '.join(set(output.failed_sources))}")

    metadata = output.run_metadata
    elapsed = metadata.get("elapsed_seconds", 0)
    console.print(f"\n[dim]Completed in {elapsed:.1f}s | Log: logs/oncomoa.log[/dim]")


def save_outputs(output: AgentOutput, output_dir: Path | None = None) -> None:
    """Save all output files: results JSON, evidence CSV."""
    from config import OUTPUT_DIR

    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(exist_ok=True)

    # results.json
    results_path = out_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output.model_dump(), f, indent=2, default=str)
    console.print(f"[green]Saved:[/green] {results_path}")

    # evidence_summary.csv
    if output.hypotheses:
        try:
            csv_path = out_dir / "evidence_summary.csv"
            rows = []
            for hyp in output.hypotheses:
                rows.append({
                    "rank": hyp.rank,
                    "biomarker": hyp.biomarker,
                    "category": hyp.biomarker_category.value,
                    "type": hyp.biomarker_type.value,
                    "direction": hyp.direction.value,
                    "confidence_score": round(hyp.confidence_score, 2),
                    "predictive_score": round(hyp.predictive_score, 2),
                    "prognostic_score": round(hyp.prognostic_score, 2),
                    "evidence_level": hyp.evidence_level or "",
                    "drug_relevance": hyp.drug_relevance,
                    "supporting_sources": "|".join(hyp.supporting_sources[:5]),
                    "civic_level": hyp.ranking_rationale.civic_level or "",
                    "pubmed_hits": hyp.ranking_rationale.pubmed_hits,
                    "clinical_trials": hyp.ranking_rationale.clinical_trials,
                    "hypothesis": hyp.hypothesis[:300] if hyp.hypothesis else "",
                })
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
            console.print(f"[green]Saved:[/green] {csv_path}")
        except Exception as exc:
            logging.getLogger(__name__).error("CSV save failed: %s", exc)


async def main() -> None:
    """Main async entrypoint."""
    setup_logging()
    args = parse_args()

    # Display header
    console.print(Panel(
        "[bold cyan]OncoMOA Biomarker Agent[/bold cyan]\n"
        "[dim]Agentic Oncology RAG + Knowledge Graph Platform[/dim]",
        expand=False,
    ))
    console.print(f"[bold]Drug:[/bold] [yellow]{args.drug}[/yellow]")
    console.print(f"[bold]MOA:[/bold] {args.moa}")
    console.print(f"[bold]Top N:[/bold] {args.top_n}")
    console.print(f"[bold]Backend:[/bold] {args.backend}")
    console.print()

    # Handle --no-llm by setting env override
    backend_override = args.backend
    if args.no_llm:
        backend_override = "none"
        console.print("[yellow]--no-llm: skipping LLM synthesis[/yellow]")

    from agents.orchestrator import OncologyOrchestrator

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        orchestrator = OncologyOrchestrator(backend_override=backend_override)
        output = await orchestrator.run(
            drug_name=args.drug,
            moa_description=args.moa,
            top_n=args.top_n,
            progress=progress,
        )

    # Display results
    display_results_table(output)

    # Save outputs
    save_outputs(output, output_dir=args.output)


if __name__ == "__main__":
    asyncio.run(main())
