# Norns Timeline Agent — LLM Legal Intelligence & Reasoning Core

This package implements the **LLM reasoning and narrative synthesis layer** for **Phase 6: Norns Timeline Agent** of Project Odin.

---

## Architecture & Guiding Philosophy

In accordance with Odin's core architectural principle:
> *"Build the evidence and deterministic calculation layers first, then add AI agents. Otherwise, your agents may produce convincing but unsupported legal conclusions."*

The LLM layer **does not perform any date calculations or financial arithmetic**. Instead:
1. The **Non-LLM Core (`norns.non_llm`)** computes frozen, evidence-linked snapshots for each checkpoint date in `clock.json`.
2. The **LLM Agent (`norns.llm`)** consumes these snapshots and uses **Claude 3.5 Sonnet** (via Anthropic API) to synthesize:
   - High-level executive legal summaries.
   - Evidence-backed risk narratives explaining *why* obligations are upcoming, due, or overdue.
   - Projected worst-case legal outcomes (e.g., cure period expirations, automatic renewal locks, formal breaches).
   - Concrete next actions and feedback requests back to the **Thor extraction agent**.

---

## Directory Structure

```
ANOTICS/norns/llm/
├── __init__.py      # Public interface exports
├── schemas.py       # Pydantic v2 models for structured LLM analysis
├── prompts.py       # System prompt with strict numerical fidelity rules
├── client.py        # Direct HTTP Anthropic Claude client (urllib-based, zero extra dependencies)
├── mock_llm.py      # Offline deterministic legal synthesizer (for dev, testing, or offline runs)
├── agent.py         # Main NornsLLMAgent coordinating Claude API with offline fallback
├── run.py           # CLI runner for end-to-end multi-checkpoint analysis
├── tests.py         # Unittest test suite (4 passing tests)
└── README.md        # This documentation
```

---

## Output Schemas (`schemas.py`)

The LLM returns structured Pydantic models:

### 1. `ObligationNarrative`
- `obligation_id`: ID of the obligation (e.g., `OB-B1-RENEWAL-NOTICE`).
- `contract_id`: Contract reference (e.g., `B1`).
- `status`: Status from deterministic engine (`upcoming`, `due`, `overdue`, `completed`).
- `summary_explanation`: Human-readable legal status summary.
- `legal_basis`: Exact clauses and ledger rows cited from deterministic snapshot.
- `cited_evidence_ids`: Verifiable citation IDs (`['B1-Sec-2.2', 'ledger_row_6']`).
- `worst_case_scenario`: Contractual impact if notice is missed or payment defaults.

### 2. `TimelineCheckpointAnalysis`
- `simulated_date`: The checkpoint date (e.g., `2027-03-15`).
- `executive_summary`: Executive-level summary of company risk position.
- `total_confirmed_due`: Confirmed debt strictly from calculation engine.
- `total_potential_penalties`: Calculated late penalties.
- `critical_alerts`: List of `ObligationNarrative` items.
- `recommended_immediate_actions`: Concrete steps for legal / finance teams.
- `missing_information_requests`: Ambiguity flags for Thor re-extraction.

### 3. `MultiCheckpointTimelineReport`
- `checkpoints_analyzed`: Analysis for each checkpoint date.
- `overall_timeline_trend`: Multi-month risk evolution trajectory.
- `unresolved_legal_questions`: Open questions across all checkpoints.

---

## How to Run

### Option A: Using Anthropic Claude (When you are ready for final evaluation)
Set your API key in the environment or pass it as a flag:

```powershell
# Set via environment variable
$env:ANTHROPIC_API_KEY = "sk-ant-api..."
python ANOTICS/norns/llm/run.py

# Or pass directly via argument
python ANOTICS/norns/llm/run.py --api-key "sk-ant-api..." --model "claude-3-5-sonnet-20241022"
```

### Option B: Running Offline / Test Mode (Zero API Key needed)
If no API key is provided, or if you pass `--mock`, it automatically uses the offline synthesizer:

```powershell
python ANOTICS/norns/llm/run.py --mock
```

Output is printed formatted to the terminal and saved to:
`ANOTICS/norns/llm_timeline_report.json`

---

## Running the Unit Tests

Execute the test suite with:

```powershell
python ANOTICS/norns/llm/tests.py
```

Expected output:
```text
....
----------------------------------------------------------------------
Ran 4 tests in 0.013s

OK
```

All tests verify prompt construction, schema integrity, JSON extraction from markdown wrappers, and multi-checkpoint synthesis.
