"""
CLI Runner for Norns LLM Timeline Agent.
Loads deterministic snapshot records and synthesizes executive legal analyses using Claude (or mock mode).
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add ANOTICS to sys.path
ANOTICS_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = ANOTICS_DIR.parent
for p in (str(ANOTICS_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from norns.non_llm import NornsTimelineAgent, ObligationItem
from norns.non_llm.run import get_sample_obligations
from norns.llm import NornsLLMAgent


def main():
    parser = argparse.ArgumentParser(description="Run Norns LLM Legal Timeline Analysis")
    parser.add_argument("--api-key", default=os.environ.get("ANTHROPIC_API_KEY", ""), help="Anthropic API Key")
    parser.add_argument("--model", default="claude-3-5-sonnet-20241022", help="Anthropic model name")
    parser.add_argument("--mock", action="store_true", help="Force mock LLM mode (no API key needed)")
    parser.add_argument("--output", default=str(ANOTICS_DIR / "norns" / "llm_timeline_report.json"), help="Output path for final report JSON")

    args = parser.parse_args()

    print("==================================================================")
    print(" NORNS - LLM Legal Intelligence & Timeline Synthesis Agent      ")
    print("==================================================================")

    # 1. Run Non-LLM deterministic engine to get ground-truth snapshots
    clock_file = ANOTICS_DIR / "clock.json"
    holidays_file = ANOTICS_DIR / "bundle" / "data" / "holidays.csv"
    ledger_file = ANOTICS_DIR / "bundle" / "data" / "payments_ledger.csv"

    print(f"Loading deterministic engine from {ANOTICS_DIR}...")
    det_agent = NornsTimelineAgent(
        clock_file=clock_file,
        holidays_file=holidays_file,
        payments_file=ledger_file,
    )
    obligations = get_sample_obligations()
    snapshots = det_agent.generate_all_snapshots(obligations)
    print(f"Generated {len(snapshots)} deterministic snapshots.")

    # 2. Initialize LLM Agent
    use_mock = args.mock or not bool(args.api_key.strip())
    mode_label = "MOCK / OFFLINE (Deterministic Legal Synthesizer)" if use_mock else f"ANTHROPIC CLAUDE ({args.model})"
    print(f"LLM Reasoning Mode: {mode_label}")
    print("------------------------------------------------------------------")

    llm_agent = NornsLLMAgent(
        api_key=args.api_key,
        model=args.model,
        use_mock=use_mock,
    )

    print("\nSynthesizing multi-checkpoint legal analysis...")
    report = llm_agent.analyze_all_checkpoints(snapshots)

    print("\n=== OVERALL TIMELINE TREND ===")
    print(report.overall_timeline_trend)

    for analysis in report.checkpoints_analyzed:
        print(f"\n------------------------------------------------------------------")
        print(f"[CHECKPOINT {analysis.checkpoint_index + 1}] Simulated Date: {analysis.simulated_date}")
        print(f"Executive Summary: {analysis.executive_summary}")
        print(f"Total Due: ${analysis.total_confirmed_due:,.2f} | Penalties: ${analysis.total_potential_penalties:,.2f}")
        print("\n  Critical Alerts & Legal Narratives:")
        for alert in analysis.critical_alerts:
            print(f"    * [{alert.status.upper()}] {alert.obligation_id} ({alert.contract_id}):")
            print(f"        Explanation: {alert.summary_explanation}")
            print(f"        Legal Basis: {alert.legal_basis}")
            if alert.worst_case_scenario:
                print(f"        Worst-Case:  {alert.worst_case_scenario}")
        print("\n  Recommended Actions:")
        for action in analysis.recommended_immediate_actions:
            print(f"    - {action}")
        if analysis.missing_information_requests:
            print("\n  Thor Re-Extraction Requests:")
            for req in analysis.missing_information_requests:
                print(f"    ? {req}")

    # Save to output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)

    print(f"\n==================================================================")
    print(f"[SUCCESS] Complete LLM Timeline Report saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
