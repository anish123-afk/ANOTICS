# Norns Timeline Agent — Non-LLM Deterministic Intelligence Core

This package provides the **deterministic, non-LLM implementation** of **Phase 6: Norns Timeline Agent** for Project Odin.

In compliance with the system design principle:
> *"Build the evidence and deterministic calculation layers first, then add AI agents. Otherwise, your agents may produce convincing but unsupported legal conclusions."*

This module carries out all date calculations, payment reconciliations, penalty and exposure evaluations, and time-travel snapshots **without a single LLM call**.

---

## Architecture & Module Structure

```
odin/norns/
├── __init__.py      # Public exports and agent interfaces
├── schemas.py       # Pydantic v2 data models & enum types (Status, Exposure, Snapshots)
├── clock.py         # Simulated clock loader for clock.json checkpoints
├── calendar.py      # Holiday calendar math & business-day notice adjustments
├── payments.py      # Deterministic payment ledger reconciler
├── exposure.py      # Financial exposure and penalty calculation engine
├── timeline.py      # Norns Timeline Agent coordinating time-travel evaluations
├── run.py           # Standalone CLI execution runner with snapshot exporter
└── tests.py         # 100% passing test suite (unittest-based, zero external dependencies)
```

---

## What Each Non-LLM Component Does

### 1. `clock.py` (Clock Loading)
- **Role:** Loads `clock.json` from the Dev Pack.
- **Function:** Parses the starting date (`2027-01-15`) and extracts the three chronological simulation checkpoints:
  - Checkpoint 1: `2027-03-15`
  - Checkpoint 2: `2027-06-01`
  - Checkpoint 3: `2027-09-15`
- **Output:** Returns typed `datetime.date` checkpoints used to frozen-evaluate the company's contractual position.

### 2. `calendar.py` (Deterministic Date Engine)
- **Role:** Performs date arithmetic incorporating `holidays.csv`.
- **Function:**
  - `add_calendar_days()`: Pure calendar day offset.
  - `add_business_days()`: Adds/subtracts days while skipping weekends (Saturday/Sunday) and observed company/national holidays.
  - `get_effective_notice_deadline()`: Calculates when a notice must be delivered prior to renewal/expiry. If a calculated deadline falls on a holiday or weekend, it automatically rolls back to the prior business day.

### 3. `payments.py` (Ledger Reconciliation)
- **Role:** Analyzes `payments_ledger.csv`.
- **Function:**
  - Matches ledger payment entries against obligations and contract references (`QH-STRATO-2026-001`, `QH-HARBOR-2026-001`, `QH-PIXEL-2026-001`).
  - Filters strictly as-of the checkpoint date (`paid_date <= checkpoint_date`).
  - Computes exact `total_paid` and `outstanding_amount`.
  - Attaches verifiable citation IDs (`ledger_row_X`) for every payment considered.

### 4. `exposure.py` (Financial Exposure Math)
- **Role:** Classifies financial exposure deterministically.
- **Categories:**
  - `confirmed_amount_due`: Unpaid base fees due.
  - `potential_penalty`: Contractual late fees or percentage penalties (e.g., 5% late fee on overdue amounts).
  - `conditional_liability`: Contingent obligations without an overdue trigger.
  - `unknown_financial_exposure`: Unpriced obligations.
- **Rule:** Never treats potential exposure as a debt currently owed unless verified overdue.

### 5. `timeline.py` (Norns Timeline Agent Core)
- **Role:** Coordinates the deterministic components across checkpoints.
- **Function:**
  - Evaluates an obligation's lifecycle state: `upcoming` -> `due` -> `overdue` -> `completed`.
  - Calculates exact `days_remaining` (or days overdue).
  - Aggregates status counters, paid amounts, outstanding amounts, and penalties.
  - Generates immutable `CheckpointSnapshot` records.

---

## How to Run & CLI Usage

Run the standalone CLI runner directly from powershell:

```powershell
python odin/norns/run.py
```

Optional arguments:
- `--clock`: Path to `clock.json` (defaults to `ANOTICS/clock.json`)
- `--holidays`: Path to `holidays.csv` (defaults to `ANOTICS/bundle/data/holidays.csv`)
- `--ledger`: Path to `payments_ledger.csv` (defaults to `ANOTICS/bundle/data/payments_ledger.csv`)
- `--output`: Path to write the resulting snapshots JSON (defaults to `odin/norns_output.json`)

---

## Understanding the Output

When you run `odin/norns/run.py`, it executes across all three checkpoints and outputs:

```text
=================================================================
 NORNS - Deterministic Contract Intelligence & Timeline Agent   
=================================================================
Clock file:    D:\project\python\Ragnarok\ANOTICS\clock.json
Holidays file: D:\project\python\Ragnarok\ANOTICS\bundle\data\holidays.csv
Ledger file:   D:\project\python\Ragnarok\ANOTICS\bundle\data\payments_ledger.csv
Simulated start date: 2027-01-15
Loaded 3 checkpoints: ['2027-03-15', '2027-06-01', '2027-09-15']
Loaded 25 ledger payment records.
-----------------------------------------------------------------

[CHECKPOINT 1] Date: 2027-03-15
  Obligations Total: 4
  Status Summary:    {'upcoming': 1, 'due': 0, 'overdue': 0, 'completed': 3, 'no_due_date': 0}
  Total Paid:        $669,000.00
  Total Outstanding: $0.00
  Potential Penalty: $0.00
    [OK] OB-A1-HOSTING-Q1 (A1): COMPLETED | Due: 2027-03-05 | Rem: -10d | Outstanding: $0.00 | Penalty: $0.00
    [OK] OB-B1-PLATFORM-APR (B1): COMPLETED | Due: 2027-04-08 | Rem: 24d | Outstanding: $0.00 | Penalty: $0.00
    [OK] OB-D1-MILESTONE-3 (D1): COMPLETED | Due: 2027-02-18 | Rem: -25d | Outstanding: $0.00 | Penalty: $0.00
    [*] OB-B1-RENEWAL-NOTICE (B1): UPCOMING | Due: 2027-07-01 | Rem: 108d | Outstanding: $0.00 | Penalty: $0.00

[CHECKPOINT 2] Date: 2027-06-01
  Obligations Total: 4
  Status Summary:    {'upcoming': 1, 'due': 0, 'overdue': 0, 'completed': 3, 'no_due_date': 0}
  Total Paid:        $677,000.00
  Total Outstanding: $0.00
  Potential Penalty: $0.00
    ...
    [*] OB-B1-RENEWAL-NOTICE (B1): UPCOMING | Due: 2027-07-01 | Rem: 30d | Outstanding: $0.00 | Penalty: $0.00

[CHECKPOINT 3] Date: 2027-09-15
  Obligations Total: 4
  Status Summary:    {'upcoming': 0, 'due': 0, 'overdue': 1, 'completed': 3, 'no_due_date': 0}
  Total Paid:        $677,000.00
  Total Outstanding: $0.00
  Potential Penalty: $0.00
    ...
    [!] OB-B1-RENEWAL-NOTICE (B1): OVERDUE | Due: 2027-07-01 | Rem: -76d | Outstanding: $0.00 | Penalty: $0.00

[DONE] Saved timeline snapshots to: D:\project\python\Ragnarok\odin\norns_output.json
```

### Key Lifecycle Transitions Observed in Output:
1. **Notice Window Transition (`OB-B1-RENEWAL-NOTICE`):**
   - At **Checkpoint 1 (2027-03-15)**: 108 days remaining -> `UPCOMING`.
   - At **Checkpoint 2 (2027-06-01)**: 30 days remaining -> `UPCOMING`.
   - At **Checkpoint 3 (2027-09-15)**: Past 2027-07-01 (-76 days overdue) -> Automatically flips to `OVERDUE` alert.
2. **Deterministic Payment Matching:**
   - StratoNode hosting and Harbor Retail fees are matched against the CSV ledger. Total paid updates from `$669,000.00` to `$677,000.00` as payments clear.
3. **Traceability:**
   - Every evaluated item maintains `evidence_ids` tracing directly to clause references and ledger row indices.

---

## Running Unit & Integration Tests

Execute the test suite with:

```powershell
python odin/norns/tests.py
```

Expected output:
```text
........
----------------------------------------------------------------------
Ran 8 tests in 0.003s

OK
```
All 8 tests verify:
- Accurate reading and sorting of `clock.json`.
- Proper holiday filtering and rollover to preceding business days.
- Ledger reconciliation as of specific simulated dates.
- Overdue penalty arithmetic.
- Deterministic state transitions from `UPCOMING` to `OVERDUE`.
