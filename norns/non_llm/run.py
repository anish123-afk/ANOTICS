"""
Standalone CLI Runner for Norns Timeline Agent.
Reads real Dev Pack files, runs the deterministic timeline simulation across checkpoints,
and exports JSON snapshots and formatted summaries.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

# Add ANOTICS directory and project root to sys.path
ANOTICS_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = ANOTICS_DIR.parent
for p in (str(ANOTICS_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from norns.non_llm import NornsTimelineAgent, ObligationItem


def get_sample_obligations() -> list[ObligationItem]:
    """
    Sample obligations derived from contracts A1, B1, D1 for demonstration.
    In the full Odin orchestrator, these are passed directly from Thor's extracted database.
    """
    return [
        ObligationItem(
            id="OB-A1-HOSTING-Q1",
            contract_id="A1",
            description="Monthly StratoNode Cloud Hosting Fee (Q1)",
            responsible_party="Quillhub Inc.",
            due_date=date(2027, 3, 5),
            contract_amount=8000.0,
            penalty_rate_percent=2.0,
            evidence_ids=["A1-Sec-4.1", "A1-Sched-A"],
        ),
        ObligationItem(
            id="OB-B1-PLATFORM-APR",
            contract_id="B1",
            description="Harbor Retail Platform Fee (April 2027)",
            responsible_party="Harbor Retail Group LLC",
            due_date=date(2027, 4, 8),
            contract_amount=40000.0,
            penalty_rate_percent=5.0,
            evidence_ids=["B1-Sec-5.2"],
        ),
        ObligationItem(
            id="OB-D1-MILESTONE-3",
            contract_id="D1",
            description="Pixelwright Milestone M3 Front-End Delivery",
            responsible_party="Quillhub Inc.",
            due_date=date(2027, 2, 18),
            contract_amount=20000.0,
            penalty_rate_percent=0.0,
            evidence_ids=["D1-StatementOfWork-M3"],
        ),
        ObligationItem(
            id="OB-B1-RENEWAL-NOTICE",
            contract_id="B1",
            description="Harbor Retail Non-Renewal Notice Deadline (60 days prior to expiry)",
            responsible_party="Quillhub Inc.",
            due_date=date(2027, 7, 1),
            contract_amount=0.0,
            penalty_rate_percent=0.0,
            evidence_ids=["B1-Sec-2.2"],
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description="Run Norns Deterministic Timeline Simulation")
    parser.add_argument("--clock", default=str(ANOTICS_DIR / "clock.json"), help="Path to clock.json")
    parser.add_argument("--holidays", default=str(ANOTICS_DIR / "bundle" / "data" / "holidays.csv"), help="Path to holidays.csv")
    parser.add_argument("--ledger", default=str(ANOTICS_DIR / "bundle" / "data" / "payments_ledger.csv"), help="Path to payments_ledger.csv")
    parser.add_argument("--output", default=str(ANOTICS_DIR / "norns" / "norns_output.json"), help="Output path for snapshots JSON")

    args = parser.parse_args()

    print("=================================================================")
    print(" NORNS - Deterministic Contract Intelligence & Timeline Agent   ")
    print("=================================================================")
    print(f"Clock file:    {args.clock}")
    print(f"Holidays file: {args.holidays}")
    print(f"Ledger file:   {args.ledger}")

    agent = NornsTimelineAgent(
        clock_file=args.clock,
        holidays_file=args.holidays,
        payments_file=args.ledger,
    )

    print(f"Simulated start date: {agent.start_date}")
    print(f"Loaded {len(agent.checkpoints)} checkpoints: {[cp.isoformat() for cp in agent.checkpoints]}")
    print(f"Loaded {len(agent.reconciler.payments)} ledger payment records.")
    print("-----------------------------------------------------------------")

    obligations = get_sample_obligations()
    snapshots = agent.generate_all_snapshots(obligations)

    for snap in snapshots:
        print(f"\n[CHECKPOINT {snap.checkpoint_index + 1}] Date: {snap.simulated_date}")
        print(f"  Obligations Total: {snap.total_obligations}")
        print(f"  Status Summary:    {snap.status_counts}")
        print(f"  Total Paid:        ${snap.total_paid:,.2f}")
        print(f"  Total Outstanding: ${snap.total_outstanding:,.2f}")
        print(f"  Potential Penalty: ${snap.total_penalties:,.2f}")
        for ob in snap.obligations:
            status_symbol = "[OK]" if ob.status.value == "completed" else "[!]" if ob.status.value == "overdue" else "[*]"
            print(f"    {status_symbol} {ob.obligation_id} ({ob.contract_id}): {ob.status.value.upper()} | Due: {ob.due_date} | Rem: {ob.days_remaining}d | Outstanding: ${ob.outstanding_amount:,.2f} | Penalty: ${ob.potential_penalty:,.2f}")

    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([s.model_dump() for s in snapshots], f, indent=2)

    print(f"\n[DONE] Saved timeline snapshots to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
