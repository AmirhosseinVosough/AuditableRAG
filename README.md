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




#	Real-world problem	How industry mitigates it	Architecture	This project's answer
1	Hallucination — model states a wrong number confidently	Forced structured output, bounds/consistency checks, never let the model do final arithmetic	Hybrid	Forced tool-call extraction (have) + bounds/name checks (missing — build)
2	Non-reproducibility — same input, different answer on rerun	Temperature=0, deterministic downstream logic, versioned prompts	Linear for the math, Hybrid overall	Have temp=0/seed on extraction; math/filtering already deterministic
3	No audit trail — can't show a regulator where a number came from	Log every prompt, response, and decision; store source doc + page for every value	Linear	Spec requires it (Phase 7); not yet implemented
4	Messy/heterogeneous source docs — scanned PDFs, inconsistent layouts, OCR errors	OCR fallback, format-robust parsing, flag unreadable rather than force a parse	Hybrid	Phase 9c design (flag "could not be read") — not built yet
5	Silent data loss — a document quietly dropped, nobody notices	Hard completeness gates that block downstream calculation on any mismatch	Linear	Have this — verification.py's completeness check
6	Prompt injection via malicious/corrupted documents	Treat document content as untrusted data, sanitize/validate before it reaches a tool call, never let extracted text execute as instructions	Linear	Not addressed at all currently — real risk once Phase 9 ingests arbitrary real-world PDFs
7	Cost/latency at scale — LLM call per document doesn't scale	Batch/parallelize, cache, right-size the model, only call LLM where regex/rules can't do the job	Linear	Sequential today (no parallelism) — needs fixing before real-data mode
8	Model drift — provider updates the model, extraction quietly changes behavior	Pin model versions, regression-test against ground truth on every model bump	Linear	Have DEFAULT_MODEL pinned; no regression alerting on drift yet
9	Data privacy/compliance — sending sensitive fund docs to a third-party API	On-prem/VPC-hosted models, data processing agreements, redact PII before the call	Linear	Not addressed — real gap before this could touch real client data
10	Automation bias — humans trusting confident-looking output without checking	Confidence scoring + mandatory human review below a threshold, never fully autonomous for high-stakes numbers	Hybrid	Not built — Phase 9c is the seed of this, needs a real escalation path
11	No ground truth for real-world data — can't measure accuracy at scale	Small hand-verified sample sets, spot-check sampling, continuous eval pipelines	Linear	Spec's Phase 9d (real_ground_truth.json) — planned, not built
12	Black-box framework risk — can't debug what LangChain/LlamaIndex does internally	Build extraction/orchestration manually so every step is inspectable	Linear (your whole premise)	Already your core design choice — this is the one you've fully solved