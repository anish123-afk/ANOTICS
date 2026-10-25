# Loki: Adversarial Contract-Risk Reviewer Agent (Phase 4)

Loki is an **adversarial contract-risk reviewer** designed for the ODIN Multi-Agent Contract Intelligence System. While Thor extracts contract facts, Loki challenges those facts, cross-examines evidence across the portfolio, stress-tests operational assumptions, uncovers hidden financial exposures, and flags timeline and compliance risks.

---

## 1. Operating Principles & Strict Offline Rule

- **Zero LLM Calls (Strictly Offline):** Operates 100% locally using deterministic legal logic, temporal math, business calendars, and verbatim evidence verification. No calls to Claude, OpenAI, or Gemini.
- **Evidence-Grounded Reasoning:** Every finding links to a verified citation in the canonical `.txt` files with document ID, clause reference, page number, and verbatim quote snippet.
- **Epistemic Classification:** Rigorously distinguishes:
  - `CONFIRMED_FACT`: Proven directly by contract text or verified incident records.
  - `DERIVED_CALCULATION`: Formally computed from verified inputs (e.g., uptime formulas, financial deficits).
  - `HYPOTHESIS_OR_CONDITIONAL`: Dependent on unconfirmed external events (e.g., whether timely upstream notice was delivered).
  - `UNRESOLVED_QUESTION`: Gaps where further discovery or executive action is required.
- **Adversarial Scrutiny:** Challenges superficial conclusions (e.g., refuting false territorial conflicts, debunking inaccurate ledger overdue flags, and clarifying non-consecutive SLA misses).

---

## 2. Architecture

Loki is implemented as a compiled **LangGraph `StateGraph`** workflow:

```
                  ┌───────────────────────────────────────────────┐
                  │                     START                     │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                  ┌───────────────────────────────────────────────┐
                  │                 load_context                  │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                  ┌───────────────────────────────────────────────┐
                  │     analyze_conflicts_and_supersessions       │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                  ┌───────────────────────────────────────────────┐
                  │            analyze_financial_gaps             │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                  ┌───────────────────────────────────────────────┐
                  │         analyze_timelines_and_events          │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                  ┌───────────────────────────────────────────────┐
                  │             adversarial_challenge             │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                  ┌───────────────────────────────────────────────┐
                  │                compile_report                 │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                  ┌───────────────────────────────────────────────┐
                  │                      END                      │
                  └───────────────────────────────────────────────┘
```

### Graph Nodes
1. **`load_context`**: Initializes state, sets clock checkpoint and event drop context.
2. **`analyze_conflicts_and_supersessions`**:
   - Compares multi-contract terms (e.g., B1 §6.1 vs C1 §4.1 exclusivity scopes; B1 §3.1 vs A2 §2.1 uptime targets).
   - Audits amendments (A3 modifying A1 Section B and A2 Section 3).
3. **`analyze_financial_gaps`**:
   - Calculates customer SLA credit exposure vs upstream host recovery for March 2027 and May 2027 incidents.
   - Evaluates notice window mismatches (Harbor's 45-day window vs StratoNode's 15-day window).
4. **`analyze_timelines_and_events`**:
   - Uses `bundle/data/holidays.csv` to calculate business days.
   - Evaluates D1 milestone deliverables and deemed acceptance (§4.3).
   - Challenges premature "overdue" flags in `bundle/data/payments_ledger.csv`.
   - Tracks insurance policy expiration (June 30, 2027) across clock checkpoints.
5. **`adversarial_challenge`**:
   - Cross-examines all findings, ensures no duplicate risk IDs, verifies all quote citations against canonical text files.
6. **`compile_report`**:
   - Generates the final structured `LokiReviewReport`.

---

## 3. Core Risk Modules & Findings

### A. Cross-Contract Conflicts
- **Coexisting Regional Exclusivities (`RISK-CONFLICT-001`):**
  - Harbor Retail (B1 §6.1) and Vantage Logistics (C1 §4.1) both cover the Pacific Northwest (WA and OR).
  - *Adversarial Finding:* They legitimately coexist because Harbor is restricted to the *retail grocery sector*, while Vantage is restricted to the *third-party logistics sector (3PL)*.
- **Operational SLA Gap (`RISK-CONFLICT-002`):**
  - Quillhub committed to 99.9% uptime for Harbor (B1 §3.1), but upstream provider StratoNode is only committed to 99.5% (A2 §2.1).

### B. Amendment & Supersession Risks
- **Notice Window Supersession (`RISK-SUPER-001`):**
  - A3 Section 1 cut StratoNode SLA credit request deadline from 30 days to **15 days**.
- **Non-Renewal Window Supersession (`RISK-SUPER-002`):**
  - A3 Section 3 shortened non-renewal notice from 90 days to **60 days**.
- **Bonus Credit (`RISK-SUPER-003`):**
  - A3 Section 2 added a 5% bonus credit for single outages exceeding 60 consecutive minutes.

### C. Upstream / Downstream Financial Recovery Gaps
- **March 2027 Downtime (`RISK-FIN-001`):**
  - 190 min downtime -> 99.57% uptime.
  - Harbor SLA (<99.9%): Quillhub owes 15% ($6,000.00).
  - StratoNode SLA (>=99.5%): StratoNode owes $0.00.
  - *Net Loss:* **$6,000.00 completely unrecoverable**.
- **May 2027 Downtime (`RISK-FIN-002`):**
  - 505 min downtime -> 98.87% uptime.
  - Harbor SLA (<99.9%): Quillhub owes 15% ($6,000.00).
  - StratoNode SLA (95-99%): 25% base + 5% bonus = 30% of $8k = $2,400.00.
  - *Net Loss:* **$3,600.00 structural deficit**.
- **Notice Window Asymmetry (`RISK-FIN-003`):**
  - Harbor has 45 days to claim; Quillhub has 15 days upstream. Upstream recovery is labeled **conditional** on whether Quillhub filed by June 15.

### D. Timeline & Obligation Risks
- **D1 Milestone M3 Deemed Acceptance (`RISK-TIME-001`):**
  - Submitted Feb 28, 2027. 10 business days review expired March 12, 2027 without rejection -> deemed accepted under D1 §4.3.
- **D1 Milestone M4 Informal Phone Delay (`RISK-TIME-002`):**
  - Elena's phone call (E-10) is not written notice (§10.2) and not a signed amendment (§10.6).
- **Accounting Ledger Discrepancy Challenge (`RISK-TIME-003`):**
  - Ledger marked April fee overdue on April 8. Challenged: Invoice E-09 specified Due Date May 17, 2027 (45 days).
- **Non-Consecutive SLA Miss Defense (`RISK-TIME-004`):**
  - March (99.57%) and May (98.87%) were separated by April (100%). Harbor has no immediate right to terminate under B1 §8.1.
- **Insurance Expiration Breach (`RISK-EVENT-002`):**
  - Policies expired June 30, 2027 without renewal, triggering material breach under B1 §7.2 and termination rights under §8.2.

---

## 4. API & Integration Usage

### Running Loki Independently
```python
from odin_loki import run_loki

# Run for Checkpoint 3 / Drop 2
report = run_loki(
    bundle_path="bundle",
    clock_checkpoint="2027-09-15",
    event_drop="Drop 2"
)

print(f"Total Findings: {report.total_findings}")
for f in report.findings:
    print(f"[{f.severity}] {f.title}: {f.summary}")
```

### Running Tests
```powershell
.\.venv\Scripts\python.exe -m pytest test_loki.py -v
```
