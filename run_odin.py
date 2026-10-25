"""
ODIN Master Orchestrator (run_odin.py).

Unified multi-agent pipeline executing Phase 1 through Phase 5 (plus Phase 9 Tyr)
in a deterministic, auditable, and repeatable sequence.

Pipeline Execution Sequence:
1. Initialize & Ingest: Phase 1 ingests manifest and populates SQLite database.
2. Fact Extraction: Thor extracts structured contract facts from all portfolio agreements.
3. Timeline & Exposure: Thor-to-Norns adapter feeds extracted obligations to Norns
   for deterministic checkpoint evaluation, payment reconciliation, and exposure math.
4. Adversarial Review: Loki challenges facts, audits supersessions, and calculates financial recovery gaps.
5. Event Monitoring: Heimdall ingests event feeds with strict checkpoint temporal filtering
   and malformed line resilience.
6. Arbitration & Action Drafting: Tyr arbitrates candidate risks and drafts traceable legal actions
   (strictly gated by human approval).

Zero LLM calls. Fully offline, testable, and deterministic.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

# Core Agents & Adapters
import odin_phase1
import odin_thor
import odin_loki
import odin_heimdall
import odin_tyr
from odin_thor_norns_adapter import adapt_thor_to_norns
from odin_tyr_adapter import run_tyr_on_live_outputs
from norns.non_llm.timeline import NornsTimelineAgent
from norns.non_llm.schemas import CheckpointSnapshot
from norns.llm import NornsLLMAgent, TimelineCheckpointAnalysis

logger = logging.getLogger("odin.orchestrator")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def run_odin_pipeline(
    checkpoint_date: str = "2027-03-15",
    event_drop: str = "all",  # "drop1", "drop2", or "all"
    bundle_path: Path | str = "bundle",
    db_path: Path | str = "odin.db",
    include_tyr: bool = True,
    contract_ids: Optional[List[str]] = None,
    include_norns_llm: bool = True,
    enable_live_llm: bool = False,
    llm_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute end-to-end ODIN pipeline across Phases 1–5 (and Tyr).

    Returns a structured dictionary with:
    - checkpoint_date
    - execution_status (overall and per-agent)
    - phase1_ingestion_summary
    - thor_extractions_summary
    - norns_timeline_summary
    - loki_review_summary
    - heimdall_event_summary
    - tyr_actions_summary
    - warnings_and_challenges
    - metadata
    """
    start_time = datetime.now()
    bpath = Path(bundle_path)
    db_file = Path(db_path)
    manifest_file = bpath / "manifest.json"
    clock_file = Path("clock.json")
    holidays_file = bpath / "data" / "holidays.csv"
    ledger_file = bpath / "data" / "payments_ledger.csv"

    pipeline_result: Dict[str, Any] = {
        "status": "success",
        "checkpoint_date": checkpoint_date,
        "event_drop": event_drop,
        "started_at": start_time.isoformat(),
        "agent_statuses": {},
        "warnings": [],
    }

    # -------------------------------------------------------------------------
    # Step 1: Phase 1 — Manifest Ingestion & Database Population
    # -------------------------------------------------------------------------
    logger.info("Executing Phase 1: Ingesting manifest into %s...", db_file)
    if not manifest_file.exists():
        msg = f"Phase 1 Error: Manifest not found at {manifest_file}"
        logger.error(msg)
        pipeline_result["status"] = "error"
        pipeline_result["agent_statuses"]["phase1"] = "failed"
        pipeline_result["warnings"].append(msg)
        return pipeline_result

    try:
        p1_report = odin_phase1.ingest_documents(manifest_file, db_path=db_file)
        pipeline_result["agent_statuses"]["phase1"] = "success"
        pipeline_result["phase1"] = {
            "processed_documents": p1_report["processed_documents"],
            "total_clauses": p1_report["total_clauses"],
            "total_cross_references": p1_report["total_cross_references"],
            "ocr_required": p1_report["ocr_required"],
            "failed_documents": p1_report["failed_documents"],
        }
    except Exception as e:
        msg = f"Phase 1 Ingestion failed: {e}"
        logger.error(msg)
        pipeline_result["status"] = "error"
        pipeline_result["agent_statuses"]["phase1"] = "failed"
        pipeline_result["warnings"].append(msg)
        return pipeline_result

    # -------------------------------------------------------------------------
    # Step 2: Phase 2 — Thor: Contract Extraction
    # -------------------------------------------------------------------------
    contracts_to_extract = contract_ids or ["A1", "A2", "A3", "B1", "B2", "C1", "D1", "STD"]
    logger.info("Executing Phase 2 Thor on %d contracts...", len(contracts_to_extract))
    thor_results = []
    thor_unverified_total = 0

    try:
        for cid in contracts_to_extract:
            res = odin_thor.run_thor(cid)
            thor_results.append(res)
            if not res.verification_passed:
                thor_unverified_total += len(res.unverified_citations)

        pipeline_result["agent_statuses"]["thor"] = "success"
        pipeline_result["thor"] = {
            "contracts_extracted": len(thor_results),
            "total_obligations_extracted": sum(len(r.obligations) for r in thor_results),
            "total_parties_extracted": sum(len(r.parties) for r in thor_results),
            "all_verifications_passed": thor_unverified_total == 0,
        }
    except Exception as e:
        msg = f"Phase 2 Thor failed: {e}"
        logger.error(msg)
        pipeline_result["agent_statuses"]["thor"] = "failed"
        pipeline_result["warnings"].append(msg)

    # -------------------------------------------------------------------------
    # Step 3: Phase 3 — Norns: Timeline, Reconciliation & Exposure
    # -------------------------------------------------------------------------
    logger.info("Executing Phase 3 Norns with Thor-adapted obligations...")
    norns_snapshot: Optional[CheckpointSnapshot] = None
    norns_llm_analysis: Optional[TimelineCheckpointAnalysis] = None
    try:
        adapted_obligations = adapt_thor_to_norns(thor_results)
        norns_agent = NornsTimelineAgent(
            clock_file=clock_file if clock_file.exists() else "clock.json",
            holidays_file=holidays_file if holidays_file.exists() else None,
            payments_file=ledger_file if ledger_file.exists() else None,
        )

        # Match checkpoint date to index
        target_d = datetime.strptime(checkpoint_date, "%Y-%m-%d").date()
        cp_idx = 0
        for idx, cp in enumerate(norns_agent.checkpoints):
            if cp <= target_d:
                cp_idx = idx

        norns_snapshot = norns_agent.generate_checkpoint_snapshot(cp_idx, adapted_obligations)
        pipeline_result["agent_statuses"]["norns"] = "success"
        pipeline_result["norns"] = {
            "simulated_date": norns_snapshot.simulated_date,
            "total_obligations_evaluated": norns_snapshot.total_obligations,
            "status_counts": norns_snapshot.status_counts,
            "total_paid_usd": norns_snapshot.total_paid,
            "total_outstanding_usd": norns_snapshot.total_outstanding,
            "total_penalties_usd": norns_snapshot.total_penalties,
        }

        # LLM Reasoning & Legal Narrative Synthesis Layer
        if include_norns_llm:
            try:
                llm_agent = NornsLLMAgent(
                    api_key=llm_api_key,
                    use_mock=not enable_live_llm,
                )
                norns_llm_analysis = llm_agent.analyze_checkpoint(norns_snapshot)
                pipeline_result["norns"]["llm_analysis"] = {
                    "executive_summary": norns_llm_analysis.executive_summary,
                    "critical_alerts_count": len(norns_llm_analysis.critical_alerts),
                    "recommended_immediate_actions": norns_llm_analysis.recommended_immediate_actions,
                    "missing_information_requests": norns_llm_analysis.missing_information_requests,
                }
            except Exception as e:
                logger.warning("Norns LLM synthesis note: %s", e)
                pipeline_result["warnings"].append(f"Norns LLM synthesis note: {e}")
    except Exception as e:
        msg = f"Phase 3 Norns failed: {e}"
        logger.error(msg)
        pipeline_result["agent_statuses"]["norns"] = "failed"
        pipeline_result["warnings"].append(msg)

    # -------------------------------------------------------------------------
    # Step 4: Phase 4 — Loki: Adversarial Risk Review
    # -------------------------------------------------------------------------
    logger.info("Executing Phase 4 Loki Adversarial Review at %s...", checkpoint_date)
    loki_report: Optional[odin_loki.LokiReviewReport] = None
    try:
        loki_report = odin_loki.run_loki(
            bundle_path=bpath,
            clock_checkpoint=checkpoint_date,
            event_drop=event_drop,
        )
        pipeline_result["agent_statuses"]["loki"] = "success"
        pipeline_result["loki"] = {
            "total_findings": loki_report.total_findings,
            "summary_by_severity": loki_report.summary_by_severity,
            "summary_by_category": loki_report.summary_by_category,
            "unverified_citations": loki_report.unverified_citations_count,
            "challenges_raised": loki_report.challenges_raised,
        }
    except Exception as e:
        msg = f"Phase 4 Loki failed: {e}"
        logger.error(msg)
        pipeline_result["agent_statuses"]["loki"] = "failed"
        pipeline_result["warnings"].append(msg)

    # -------------------------------------------------------------------------
    # Step 5: Phase 5 — Heimdall: Event Monitoring (Temporal Checkpoint Slicing)
    # -------------------------------------------------------------------------
    logger.info("Executing Phase 5 Heimdall Event Monitoring...")
    heimdall_alerts: List[odin_heimdall.ContractAlert] = []
    try:
        event_files: List[Path] = []
        drop1_path = bpath / "events" / "events_drop1.jsonl"
        drop2_path = bpath / "events" / "events_drop2.jsonl"

        if event_drop == "drop1" and drop1_path.exists():
            event_files.append(drop1_path)
        elif event_drop == "drop2" and drop2_path.exists():
            event_files.append(drop2_path)
        else:
            if drop1_path.exists():
                event_files.append(drop1_path)
            if drop2_path.exists():
                event_files.append(drop2_path)

        heim_res = odin_heimdall.process_events(
            event_files=event_files,
            db_path=db_file,
            severity_threshold="medium",
            checkpoint_date=checkpoint_date,
        )
        pipeline_result["agent_statuses"]["heimdall"] = "success"
        pipeline_result["heimdall"] = {
            "events_processed": heim_res["events_processed"],
            "malformed_lines_recovered": heim_res["malformed_lines_recovered"],
            "alerts_generated": heim_res["alerts_generated"],
            "alerts_summary": [
                {"alert_id": a["alert_id"], "severity": a["severity"], "trigger": a["potential_trigger"]}
                for a in heim_res["alerts"]
            ],
        }
        heimdall_alerts = [odin_heimdall.ContractAlert(**a) for a in heim_res["alerts"]]
    except Exception as e:
        msg = f"Phase 5 Heimdall failed: {e}"
        logger.error(msg)
        pipeline_result["agent_statuses"]["heimdall"] = "failed"
        pipeline_result["warnings"].append(msg)

    # -------------------------------------------------------------------------
    # Step 6: Phase 9 — Tyr: Arbitration & Action Drafting (Safety Guardrails)
    # -------------------------------------------------------------------------
    if include_tyr:
        logger.info("Executing Phase 9 Tyr Arbitration & Action Drafting...")
        try:
            tyr_state = run_tyr_on_live_outputs(
                loki_report=loki_report,
                heimdall_alerts=heimdall_alerts,
                norns_snapshot=norns_snapshot,
                checkpoint_date=checkpoint_date,
                norns_analysis=norns_llm_analysis,
            )
            pipeline_result["agent_statuses"]["tyr"] = "success"
            pipeline_result["tyr"] = {
                "arbitration_decisions_count": len(tyr_state.arbitration_decisions),
                "proposed_actions_count": len(tyr_state.proposed_actions),
                "all_actions_require_human_approval": all(a.requires_human_approval for a in tyr_state.proposed_actions),
                "proposed_actions": [
                    {
                        "action_id": a.action_id,
                        "type": a.action_type.value,
                        "title": a.title,
                        "recipient": a.intended_recipient,
                        "requires_human_approval": a.requires_human_approval,
                    }
                    for a in tyr_state.proposed_actions
                ],
            }
        except Exception as e:
            msg = f"Phase 9 Tyr failed: {e}"
            logger.error(msg)
            pipeline_result["agent_statuses"]["tyr"] = "failed"
            pipeline_result["warnings"].append(msg)
    else:
        pipeline_result["agent_statuses"]["tyr"] = "skipped"

    end_time = datetime.now()
    pipeline_result["completed_at"] = end_time.isoformat()
    pipeline_result["duration_seconds"] = round((end_time - start_time).total_seconds(), 2)

    return pipeline_result


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ODIN Multi-Agent Master Orchestrator")
    parser.add_argument("--checkpoint", default="2027-03-15", help="Simulation checkpoint date (e.g. 2027-03-15, 2027-06-01, 2027-09-15)")
    parser.add_argument("--drop", default="all", choices=["drop1", "drop2", "all"], help="Event drop to process")
    parser.add_argument("--bundle", default="bundle", help="Path to bundle directory")
    parser.add_argument("--db", default="odin.db", help="Path to shared SQLite database")
    parser.add_argument("--no-tyr", action="store_true", help="Skip Tyr arbitration step")
    parser.add_argument("--no-norns-llm", action="store_true", help="Skip Norns LLM synthesis step")
    parser.add_argument("--enable-live-llm", action="store_true", help="Enable live Anthropic Claude API calls (otherwise uses deterministic mock engine)")
    parser.add_argument("--api-key", default=None, help="Anthropic API Key for live Claude analysis")
    parser.add_argument("--json-out", default=None, help="Optional output JSON path for run results")

    args = parser.parse_args()

    print("=" * 78)
    print(" ODIN — Multi-Agent Contract Intelligence System (End-to-End Orchestrator)")
    print("=" * 78)
    print(f"Checkpoint Date: {args.checkpoint}")
    print(f"Event Drop:      {args.drop}")
    print(f"Bundle Path:     {args.bundle}")
    print(f"Database:        {args.db}")
    print(f"Include Tyr:     {not args.no_tyr}")
    print(f"Norns LLM Mode:  {'LIVE CLAUDE' if args.enable_live_llm else 'OFFLINE MOCK' if not args.no_norns_llm else 'SKIPPED'}")
    print("-" * 78)

    results = run_odin_pipeline(
        checkpoint_date=args.checkpoint,
        event_drop=args.drop,
        bundle_path=args.bundle,
        db_path=args.db,
        include_tyr=not args.no_tyr,
        include_norns_llm=not args.no_norns_llm,
        enable_live_llm=args.enable_live_llm,
        llm_api_key=args.api_key,
    )

    print("\n--- PIPELINE EXECUTION SUMMARY ---")
    print(f"Overall Status: {results['status'].upper()} (Duration: {results.get('duration_seconds', 0)}s)")
    for agent, st in results.get("agent_statuses", {}).items():
        print(f"  [{st.upper():7s}] {agent.upper()}")

    if "phase1" in results:
        p1 = results["phase1"]
        print(f"\n[Phase 1] Documents: {p1['processed_documents']} | Clauses: {p1['total_clauses']} | CrossRefs: {p1['total_cross_references']}")
    if "thor" in results:
        th = results["thor"]
        print(f"[Phase 2 Thor] Contracts Extracted: {th['contracts_extracted']} | Obligations: {th['total_obligations_extracted']} | Verifications: {'PASSED' if th['all_verifications_passed'] else 'FAIL'}")
    if "norns" in results:
        nr = results["norns"]
        print(f"[Phase 3 Norns] Obligations Evaluated: {nr['total_obligations_evaluated']} | Outstanding: ${nr['total_outstanding_usd']} | Penalties: ${nr['total_penalties_usd']}")
        if "llm_analysis" in nr:
            llm_a = nr["llm_analysis"]
            print(f"  [Norns LLM Narrative] {llm_a['executive_summary']}")
            if llm_a.get("recommended_immediate_actions"):
                print(f"  [Immediate Action] {llm_a['recommended_immediate_actions'][0]}")
            if llm_a.get("missing_information_requests"):
                print(f"  [Thor Feedback Loop] {llm_a['missing_information_requests'][0]}")
    if "loki" in results:
        lk = results["loki"]
        print(f"[Phase 4 Loki] Risk Findings: {lk['total_findings']} | Unverified Quotes: {lk['unverified_citations']}")
        for sev, count in lk.get("summary_by_severity", {}).items():
            print(f"    - {sev}: {count}")
    if "heimdall" in results:
        hm = results["heimdall"]
        print(f"[Phase 5 Heimdall] Events Processed: {hm['events_processed']} | Alerts: {hm['alerts_generated']} | Malformed Lines Recovered: {hm['malformed_lines_recovered']}")
    if "tyr" in results:
        ty = results["tyr"]
        print(f"[Phase 9 Tyr] Arbitrations: {ty['arbitration_decisions_count']} | Actions Drafted: {ty['proposed_actions_count']} (Human Approval Enforced: {ty['all_actions_require_human_approval']})")

    if results.get("warnings"):
        print("\nWarnings:")
        for w in results["warnings"]:
            print(f"  * {w}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nExported complete run results to {out_path}")

    print("=" * 78)


if __name__ == "__main__":
    main()
