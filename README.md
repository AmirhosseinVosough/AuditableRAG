# Fund Aggregation Pipeline

A transparent, hybrid deterministic/agentic pipeline for answering auditable
fund-aggregation questions — "what's the weighted average expense ratio
across our ESG funds, excluding any that closed this quarter?" — from real
or synthetic fund documents (fact sheets, prospectuses), with a full audit
trail from question to final number.

## The problem this solves

That kind of number normally means a human opening every PDF by hand,
deciding which funds qualify, and doing the math themselves — slow and
easy to get subtly wrong. The obvious fix, "just ask an AI to read
everything and give you the number," creates a worse problem: a black box.
If it's wrong, nobody can tell why, and there's no trail for a regulator or
auditor asking "where did this number come from."

This project's answer: split the work into two kinds and never let them
mix. The LLM is only ever used to *read* — pull a fact off a page, or say
"not found." It never decides which funds qualify and never does the
arithmetic. Every other decision is plain, deterministic code, and every
value, inclusion, and exclusion is logged with a reason back to its source
document and page.

## Architecture: hybrid deterministic/agentic

The hybrid split is structurally enforced, not just a design intention.
`src/agentic/agent_orchestrator.py` is the one file allowed to reason about
*what to do next* (retry, escalate, give up) — and it is only allowed to do
that by calling a fixed set of deterministic tools, never reimplementing
their logic:

| Deterministic tool | Lives in | Called as |
|---|---|---|
| Which funds qualify | `fund_filter.py` | `scope_documents(...)` |
| Is a value plausible | `verification.py` | `check_extraction_quality(...)` |
| Borderline vs. obvious garbage | `verification.py` | `is_borderline_quality_failure(...)` |
| The actual math | `calculator.py` | `weighted_average_expense_ratio(...)` |
| Real-document field extraction (tiered, `temperature=0`) | `extraction_cascade.py` | `extract_with_cascade(...)` |

No LangChain/LlamaIndex — every tool schema, retry loop, and piece of
orchestration state is hand-written, so every step is inspectable.

## Three ways to run it

**1. Synthetic mode** — the deterministic core alone, against a
hand-verified 12-document synthetic set with a known-correct answer:
```bash
python -m src.generate_synthetic_data   # writes data/raw_pdfs/
python -m src.pipeline                  # runs the deterministic chain, checks against data/ground_truth.json
```

**2. Real mode (no agent)** — the same deterministic chain, pointed at a
folder of arbitrary real PDFs (`data/user_uploads/`):
```python
from src.pipeline import run_pipeline
from src.fund_filter import FilterSpec
result = run_pipeline("weighted average expense ratio for ESG active funds",
                       FilterSpec(is_esg=True, status="active"), source="real")
```

**3. Real mode, agentic** — the full REASON → ACT → EVALUATE → RETRY →
ESCALATE loop: parses an arbitrary natural-language question, scopes
documents cheaply before paying for full extraction, retries genuine
infrastructure failures, and escalates (rather than silently drops)
anything it can't confidently resolve:
```python
from src.shared.env import load_dotenv
load_dotenv()
from src.agentic.agent_orchestrator import run_agentic_pipeline
result = run_agentic_pipeline(
    "weighted average expense ratio for ESG active funds", source="real"
)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY
```

`HF_TOKEN` in `.env` is optional — only affects download speed for the
local sentence-transformers model used in semantic page-narrowing.

## Verified result

A live run of mode 3 above, against real, publicly available fund fact
sheets (iShares, Vanguard, Xtrackers), produced:

```
Included: esgu_fact_sheet, esgv_fact_sheet, ussg_fact_sheet
Excluded: corrupted_fact_sheet (could not be read),
          ivv_fact_sheet, spy_fact_sheet (is_esg: not found in document)
Needs review: (none)
Final answer: 0.12380140330900345
```

Every included fund's extracted `expense_ratio`/`aum` and the final
weighted average match this project's own hand-verified ground truth
(`data/real_ground_truth.json`) exactly, to the decimal. Both exclusions
match that same ground truth's documented "legitimately ambiguous, correct
to flag rather than guess" cases — IVV and SPY are plain S&P 500 trackers
whose fact sheets never state an ESG position either way.

## Project layout

```
src/
  fund_filter.py, calculator.py          deterministic core: filtering, arithmetic
  generate_synthetic_data.py             synthetic fixture generator
  pipeline.py                            Phases 1-9: straight-line orchestration (synthetic + real)
  verification.py                        completeness/bounds/borderline-vs-garbage checks
  extraction/                            PDF text extraction, LLM field extraction, real-data loading
  cascade/                               real-data extraction cascade: regex -> BM25 -> semantic ->
                                          LLM -> table-data -> OCR -> flag, with citations
  agentic/                               Phases 10-16: query parsing, document scoping, the real
                                          agentic loop (agent_orchestrator.py)
  shared/                                cross-cutting: model fallback, .env loading, provenance types
tests/                                   pytest suite - fast/mocked control-flow tests plus
                                          live end-to-end ground-truth checks
```

## Testing

```bash
pytest tests/test_agentic_orchestrator.py tests/test_agentic_real_orchestrator.py \
       tests/test_calculator.py tests/test_fund_filter.py tests/test_retrieval_tiers.py
```
Fast, deterministic, no `GROQ_API_KEY` required. The full suite (including
`test_pipeline.py`/`test_real_data_pipeline.py`) additionally needs a real
key and makes live API calls.

## Documentation

- `AGENTIC_LAYER_BUILD_PROMPT.md` — the full phase-by-phase build spec and
  status, including design decisions and honestly-flagged open issues, not
  just what shipped.
- `INTERVIEW_PREP_QUESTIONS.md` — a self-quiz question bank covering the
  architecture end to end, no answers included on purpose.

## Known open items

Flagged rather than hidden, per this project's own "never silently
proceed" principle:
- One real-data test (`test_real_pipeline_flags_unreadable_document_without_halting_the_batch`)
  has failed identically on the same document twice — confirmed
  reproducible, not yet root-caused.
- The build spec calls for the Anthropic SDK; the implementation uses Groq
  throughout — never reconciled.
- No parallelism yet — extraction is sequential per document.
- Data privacy/compliance (on-prem hosting, PII redaction) not addressed —
  a real gap before this could touch actual client data.
