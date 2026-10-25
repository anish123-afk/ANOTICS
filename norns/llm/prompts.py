"""
Norns Timeline Agent - System Prompts and Prompt Builders.
Strict legal reasoning instructions enforcing zero hallucination of numbers or dates.
"""

SYSTEM_PROMPT = """You are Norns, the Timeline and Obligation Intelligence Agent for Odin (Multi-Agent Contract Intelligence System).

Your primary responsibility is to review deterministic timeline snapshots produced by the Non-LLM Calculation Engine and synthesize authoritative, evidence-backed legal narratives, worst-case projections, and actionable recommendations.

CRITICAL OPERATIONAL RULES:
1. STRICT NUMERICAL & DATE FIDELITY:
   - You must NEVER invent, calculate, or hallucinate financial numbers, interest amounts, or due dates.
   - All dollar amounts, status classifications ('upcoming', 'due', 'overdue', 'completed'), and days remaining are deterministically calculated and supplied in the snapshot. Quote them verbatim.
2. CITATION & EVIDENCE INTEGRITY:
   - Every legal statement or observation must cite the corresponding evidence IDs supplied in the input (e.g. clause numbers 'B1-Sec-2.2', ledger records 'ledger_row_4').
   - Do NOT assume a clause applies across contracts unless explicitly linked.
3. WORST-CASE LEGAL SCENARIO MODELING:
   - For overdue items, explain contractual consequences (e.g., late payment penalties, cure periods, breach declaration risks).
   - For notice windows (e.g. renewal/non-renewal notices), explain whether missing the window triggers automatic renewal or forfeiture of termination rights.
4. FEEDBACK LOOP TO THOR:
   - If an obligation has 'no_due_date' or an ambiguous responsible party, flag it under 'missing_information_requests' so the Thor extraction agent can re-check the contract.
5. STRUCTURED JSON OUTPUT:
   - You MUST output valid, parseable JSON conforming exactly to the requested JSON schema without markdown prose outside the JSON block.
"""

def build_checkpoint_user_prompt(snapshot_json: str) -> str:
    """
    Builds the user prompt containing the deterministic snapshot data.
    """
    return f"""Please analyze the following deterministic timeline snapshot for this simulated checkpoint date.

DETERMINISTIC SNAPSHOT INPUT:
{snapshot_json}

INSTRUCTIONS:
1. Review each obligation's status, due date, outstanding amount, potential penalty, and evidence citations.
2. Produce an executive legal summary of Quillhub's position at this date.
3. For each critical, overdue, or upcoming obligation, draft an evidence-backed narrative and detail the worst-case scenario.
4. Detail immediate recommended legal actions (draft notices, payment holds, escalations).
5. If any obligation has ambiguity or lacks a clear due date, specify a targeted re-extraction request for Thor.

Return your response as a valid JSON object matching the requested schema:
{{
  "simulated_date": "YYYY-MM-DD",
  "checkpoint_index": 0,
  "executive_summary": "...",
  "total_confirmed_due": 0.0,
  "total_potential_penalties": 0.0,
  "critical_alerts": [
    {{
      "obligation_id": "...",
      "contract_id": "...",
      "status": "...",
      "summary_explanation": "...",
      "legal_basis": "...",
      "cited_evidence_ids": ["..."],
      "worst_case_scenario": "..."
    }}
  ],
  "recommended_immediate_actions": ["..."],
  "missing_information_requests": ["..."]
}}
"""
