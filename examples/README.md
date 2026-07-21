# OncoMOA examples

Run all examples from the repository root after installing dependencies:

```bash
cd oncomoa-agent
source .venv/bin/activate
```

## 1. Standard targeted-therapy run

This retrieves drug targets, expands pathway candidates, gathers evidence,
builds the graph, ranks candidates, and uses the configured LLM backend for
narratives.

```bash
python3.10 main.py \
  --drug "sotorasib" \
  --moa "Covalent KRAS G12C inhibitor" \
  --top-n 10 \
  --output output/sotorasib_run
```

Inspect `output/sotorasib_run/results.json` for hypotheses and
`knowledge_graph.graphml` in Cytoscape for the evidence graph.

## 2. Deterministic, no-LLM run

Use this mode for reproducible evidence scoring without any LLM request.

```bash
python3.10 main.py \
  --drug "olaparib" \
  --moa "PARP1/2 inhibitor" \
  --top-n 10 \
  --no-llm \
  --output output/olaparib_deterministic
```

The output records `"llm_backend_used": "none"`. Hypotheses, if present, are
ranked exclusively by the deterministic evidence engine.

## 3. Immune checkpoint inhibitor

```bash
python3.10 main.py \
  --drug "pembrolizumab" \
  --moa "PD-1 immune checkpoint inhibitor" \
  --top-n 10 \
  --no-llm \
  --output output/pembrolizumab_run
```

Remove `--no-llm` only when Gemini or Ollama has been configured. The
deterministic ranking and grounding gate always run before synthesis.

## 4. Sparse-evidence / unavailable-network behavior

This command demonstrates safe behavior when the public sources are unavailable
or do not return sufficient linked evidence:

```bash
python3.10 main.py \
  --drug "belzutifan" \
  --moa "HIF-2α inhibitor for VHL-associated and hypoxia-driven tumors" \
  --no-llm \
  --output output/belzutifan_safe
```

Expected safe output in that case:

```json
{
  "llm_backend_used": "none",
  "hypotheses": [],
  "total_evidence_items": 0,
  "run_metadata": {
    "insufficient_evidence": true
  }
}
```

An empty hypothesis list is intentional. It prevents the system from turning
mechanism-of-action text or missing retrieval data into unsupported biomarker
claims.

## 5. Tests

```bash
python3.10 -m pytest \
  tests/test_scoring.py \
  tests/test_schemas.py \
  tests/test_pipeline_safety.py -v
```

Integration tests query public services and require network access:

```bash
python3.10 -m pytest tests/test_integration.py -v
```
