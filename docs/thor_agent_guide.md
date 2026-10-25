# Thor: Contract Extraction Agent (Phase 4)

Thor is a contract extraction agent designed for the Odin Contract Intelligence System. Thor extracts structured, verifiable contractual facts from agreements with provenance citations, confidence scores, and ambiguity tracking.

---

## 1. Architecture

Thor is built as a compiled **LangGraph `StateGraph`** workflow with semantic BM25 clause indexing and deterministic entity extraction.

```
                  ┌─────────────────────┐
                  │        START        │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   load_and_chunk    │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   extract_entities  │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ validate_and_verify │
                  └──────────┬──────────┘
                             │
                [Verification Passed?]
                 /                  \
             (Yes)                  (No & targeted clauses)
               /                      \
              ▼                        ▼
       ┌─────────────┐       ┌──────────────────────┐
       │     END     │◄──────┤ re_extract_targeted  │
       └─────────────┘       └──────────────────────┘
```

### Graph Nodes
1. **`load_and_chunk`**:
   - Loads the contract clauses using Phase 1's ingestion engine or existing chunked records.
   - Preserves clause IDs, heading text, page numbers, and exact source text.

2. **`extract_entities`**:
   - Builds a `SemanticClauseIndex` over clause chunks using BM25 token scoring.
   - Performs semantic searches for specific contractual categories (Parties, Term, Payment, Obligations, Penalties, Compliance, Termination, Cross-References).
   - Validates data against typed Pydantic models.
   - Explicitly records missing or ambiguous fields in `ambiguous_or_missing` rather than hallucinating answers.

3. **`validate_and_verify`**:
   - Validates each piece of extracted `Evidence` against the actual verbatim clause source text.
   - Ensures no hallucinated quotes exist.
   - Computes `verification_passed` flag.

4. **`re_extract_targeted`**:
   - Conditional node triggered when another agent or workflow step targets specific clause IDs with evidence gaps.
   - Refreshes extractions specifically focused on the target clauses.

---

## 2. Structured Information Extracted

| Field | Description | Models |
|---|---|---|
| **Parties** | Contracting parties, legal names, roles, addresses, notice emails | `Party`, `Evidence` |
| **Duration & Term** | Start date, end date, initial period, renewal term, non-renewal notice | `DurationAndTerm` |
| **Payment Schedules** | Fee amount, billing frequency, payment terms in days, late payment interest | `PaymentSchedule` |
| **Obligations** | Specific duties, responsible party, due date, uncertainty/ambiguity note | `ExtractedObligation` |
| **Penalty Clauses** | Late fees, service credits, penalty calculations | `PenaltyClause` |
| **Compliance Requirements** | SOC 2, GDPR, HIPAA, security standards | `ComplianceRequirement` |
| **Termination Conditions** | Material breach triggers, cure periods, termination notice windows | `TerminationCondition` |
| **Cross-References** | Internal & external references (e.g. `Section 4.4` to `STD`), target document, resolution status | `CrossReferenceCitation` |
| **Ambiguity & Missing Information** | Explicit notes when clauses leave terms undefined | `ambiguity`, `ambiguous_or_missing` |

---

## 3. Evidence & Provenance Standard

Every extracted item attaches one or more `Evidence` objects:
```json
{
  "document_id": "A1",
  "clause_id": "C",
  "source_text": "Monthly Fee: $8,000.00 (Eight Thousand U.S. Dollars)",
  "page_number": 1,
  "confidence": 0.95
}
```

---

## 4. Usage Example

```python
import odin_thor

# Run Thor on contract A1
result = odin_thor.run_thor("A1")

print("Parties:", [f"{p.role}: {p.name}" for p in result.parties])
print("Term:", result.duration_and_term.initial_period)
print("Monthly Fee:", result.payment_schedule.fee_amount)
print("Late Payment Interest:", result.penalty_clauses[0].rate_or_amount)
print("Verification Passed:", result.verification_passed)
```

---

## 5. Test Suite

Run all Thor tests:
```powershell
pytest -v test_thor.py
```
Test coverage verifies:
1. BM25 semantic search across clause chunks.
2. Accurate extraction of all structured fields.
3. Explicit identification of missing information (no hallucination).
4. Verbatim evidence citation and verification.
5. Handling of inconsistent / ambiguous clauses.
6. Cross-reference detection and document linking (e.g., to `STD`).
7. Targeted clause re-extraction workflow.

