# Fund Aggregation Pipeline

A transparent, plain-Python pipeline for answering auditable fund-aggregation
questions from fund documents.

## Current phase

Phase 1 provides the project skeleton and a deterministic synthetic dataset.
Run the following after installing the project dependencies:

```bash
python -m src.generate_synthetic_data
```

This writes 12 one-page fund PDFs into `data/raw_pdfs/`. The hand-verified
source data and expected result for the example query are in
`data/ground_truth.json`.

Later phases will implement extraction, structured filtering, verification,
calculation, orchestration, and the end-to-end test. They are deliberately not
implemented in this phase.
