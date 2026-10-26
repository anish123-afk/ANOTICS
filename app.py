"""
Project Odin — Streamlit Web Interface
======================================
Interactive interface for Project Odin's multi-agent contract & operations intelligence system.
Workflow:
  [Odin] ──> [Thor] ◄──> [Loki] ──> [Norns] ──> [Loki] ──> [Heimdall] ──> [Tyr] ──> [Final Ans]
"""

import os
import sys
import json
import streamlit as st

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from odin_state import OdinState
from thor_loki import ThorAgent, LokiAgent
from norns import NornsAgent
from heimdall import HeimdallAgent
from tyr import TyrAgent

# Page configuration
st.set_page_config(
    page_title="Project Odin — Multi-Agent Contract Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .workflow-pill {
        display: inline-block;
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #1D4ED8;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 8px;
    }
    .ans-box {
        background-color: #F0FDF4;
        border-left: 5px solid #16A34A;
        padding: 18px;
        border-radius: 6px;
        margin-top: 15px;
        margin-bottom: 15px;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .judgment-box {
        background-color: #F8FAFC;
        border-left: 5px solid #6366F1;
        padding: 16px;
        border-radius: 6px;
        margin-bottom: 12px;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Sidebar Controls & Configurations
# ----------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/scales.png", width=70)
    st.title("⚡ Project Odin")
    st.caption("Autonomous Legal & Operational Intelligence")
    st.markdown("---")

    st.subheader("⚙️ Configuration")
    checkpoint_choice = st.selectbox(
        "Clock Checkpoint Date",
        options=["0 (2027-03-15)", "1 (2028-06-01 - Urgent)", "2 (2028-09-01 - Overdue)"],
        index=0,
        help="Simulated point in time for calculating deadline compliance"
    )
    checkpoint_val = checkpoint_choice.split()[0]

    max_iterations = st.slider("Max Thor-Loki Loop Steps", min_value=1, max_value=5, value=3)

    st.markdown("---")
    st.subheader("📂 Ecosystem Agents")
    st.markdown("""
    - **🔨 Thor**: Vector Extractor (`bge-m3` + ChromaDB)
    - **🦊 Loki**: Sequence Verifier & Adversarial Review
    - **⏳ Norns**: Timeline & Non-Renewal Notice Engine
    - **🛡️ Heimdall**: Event Log Monitoring (Cosine Match)
    - **⚖️ Tyr**: Supreme Arbiter & Final Answer Synthesizer
    """)

    st.markdown("---")
    if st.button("Clear State & Reset"):
        st.session_state.clear()
        st.rerun()


# ----------------------------------------------------------------------
# Header & Workflow Visualization
# ----------------------------------------------------------------------
st.markdown('<div class="main-header">⚡ Project Odin — Main Orchestrator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Collaborative multi-agent execution pipeline synthesizing contract law, deadline calculations, and operational event logs.</div>', unsafe_allow_html=True)

# Workflow Diagram Indicator
st.markdown("""
<div style="margin-bottom: 1.2rem;">
    <span class="workflow-pill">⚡ Odin</span> ➔
    <span class="workflow-pill">🔨 Thor (Extract)</span> ⇄
    <span class="workflow-pill">🦊 Loki (Verify)</span> ➔
    <span class="workflow-pill">⏳ Norns (Timeline)</span> ➔
    <span class="workflow-pill">🦊 Loki (Legal)</span> ➔
    <span class="workflow-pill">🛡️ Heimdall (Events)</span> ➔
    <span class="workflow-pill">⚖️ Tyr (Arbiter)</span> ➔
    <span class="workflow-pill" style="background-color: #DCFCE7; border-color: #86EFAC; color: #15803D;">🎯 Final Ans</span>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Sample Query Quick-Select
# ----------------------------------------------------------------------
sample_queries = {
    "Select a pre-configured query...": "",
    "Harbor Retail SLA uptime commitments & service credits":
        "Harbor Retail SLA uptime commitments, service credits, and outage remedies",
    "March 3 CloudCore compute service disruption incident":
        "Service disruption March 3 infrastructure outage duration and resolution",
    "Commercial General Liability policy expiration and renewal notice":
        "Insurance policy renewal commercial general liability expiration date",
    "Pixelwright milestone M3 deliverable acceptance procedure":
        "Milestone M3 deliverable submission acceptance procedures and code coverage"
}

chosen_sample = st.selectbox("Quick Templates (Optional):", list(sample_queries.keys()))

default_text = sample_queries[chosen_sample] if chosen_sample != "Select a pre-configured query..." else ""
if not default_text and "current_query" in st.session_state:
    default_text = st.session_state["current_query"]
elif not default_text:
    default_text = "Harbor Retail SLA uptime commitments, service credits, and outage remedies"

user_query = st.text_area(
    "Enter Legal / Operational Query:",
    value=default_text,
    height=85,
    placeholder="e.g. Harbor Retail SLA uptime commitments, service credits, and outage remedies"
)


# ----------------------------------------------------------------------
# Orchestration Execution
# ----------------------------------------------------------------------
col1, col2 = st.columns([1, 4])
with col1:
    execute_button = st.button("🚀 Run Odin Pipeline", type="primary", use_container_width=True)

if execute_button and user_query.strip():
    st.session_state["current_query"] = user_query.strip()
    
    with st.status("Executing Multi-Agent Workflow...", expanded=True) as status:
        # Step 0: State Init
        st.write("⚡ Initializing `OdinState` with target query...")
        state = OdinState(query=user_query.strip())

        # Step 1: Thor <-> Loki Loop
        st.write("🔨 **Stage 1: Thor <──> Loki Extraction Loop**...")
        thor = ThorAgent(max_iteration=max_iterations)
        loki = LokiAgent()

        target_doc = None
        while thor.local_state.iteration <= thor.local_state.max_iteration:
            iter_num = thor.local_state.iteration
            thor.extract_clause(state, target_contract_id=target_doc)
            should_reextract, next_doc = loki.evaluate(state)

            if should_reextract:
                st.write(f"🦊 Loki: Subsequent document `{next_doc}` found. Re-extracting (Iteration {iter_num})...")
                if iter_num >= thor.local_state.max_iteration:
                    break
                target_doc = next_doc
                thor.local_state.iteration += 1
            else:
                doc_name = state.retrieved_document.get("document_name", "Unknown") if state.retrieved_document else "Unknown"
                st.write(f"🦊 Loki: Document sequence verified (`{doc_name}`). Extraction approved!")
                break

        # Step 2: Norns
        st.write("⏳ **Stage 2: Norns Timeline & Compliance Engine**...")
        norns = NornsAgent()
        norns.evaluate_portfolio(
            checkpoint_val=checkpoint_val,
            odin_state=state
        )
        st.write("⏳ Norns: Calculated notice deadlines and timeline status.")

        # Step 3: Loki Interpretation
        st.write("🦊 **Stage 3: Loki Adversarial Legal Review**...")
        loki.interpret_final_document(state)
        st.write("🦊 Loki: Synthesized adversarial legal analysis with Claude Haiku.")

        # Step 4: Heimdall Event Monitoring
        st.write("🛡️ **Stage 4: Heimdall Event & Incident Monitoring**...")
        heimdall = HeimdallAgent()
        heimdall_res = heimdall.evaluate(state)
        matched_ev = heimdall_res.get("matched_event")
        matched_id = matched_ev.get("event_id") if matched_ev else "None"
        st.write(f"🛡️ Heimdall: Cosine matched operational event log `{matched_id}`.")

        # Step 5: Tyr Arbiter & Final Answer
        st.write("⚖️ **Stage 5: Tyr Supreme Arbiter & Judgment**...")
        tyr = TyrAgent()
        tyr.synthesize_judgment(state, matched_event=matched_ev)
        tyr.derive_final_answer(state, matched_event=matched_ev)
        st.write("⚖️ Tyr: Synthesized final judgment and derived definitive final answer.")

        # Mark human approval pending
        state.update_human_approval(None, feedback="Awaiting user action in web interface")
        st.session_state["odin_state"] = state

        status.update(label="✅ Workflow Execution Complete!", state="complete", expanded=False)


# ----------------------------------------------------------------------
# Results Display
# ----------------------------------------------------------------------
if "odin_state" in st.session_state:
    state: OdinState = st.session_state["odin_state"]

    st.markdown("---")
    
    # 1. Prominent Display of Tyr's Final Judgment
    st.subheader("⚖️ Agent Tyr — Final Judgment (`final_judgment`)")
    st.caption("Supreme legal and operational synthesis balancing Loki's contract review against Heimdall's incident logs:")
    
    if state.final_judgment:
        st.info(state.final_judgment, icon="⚖️")
    else:
        st.warning("No final judgment generated yet.")

    # 2. Prominent Display of Tyr's Final Answer
    st.subheader("🎯 Definitive Final Answer (`final_ans`)")
    st.caption("Actionable resolution derived by Agent Tyr from the final judgment:")

    if state.final_ans:
        st.success(state.final_ans, icon="🎯")
    else:
        st.warning("No final answer generated yet.")

    # 3. Human-in-the-Loop Approval Section
    st.markdown("### 👤 Human Approval Required (Human-in-the-Loop)")
    
    col_app1, col_app2, col_app3 = st.columns([1.5, 1.5, 3])
    
    current_approval = state.human_approval

    with col_app1:
        if st.button("✅ Approve Final Answer", type="primary", use_container_width=True):
            state.update_human_approval(True, feedback="Approved by reviewer via web interface")
            st.session_state["odin_state"] = state
            st.rerun()

    with col_app2:
        if st.button("❌ Request Revision / Reject", type="secondary", use_container_width=True):
            state.update_human_approval(False, feedback="Rejected / revision requested by reviewer")
            st.session_state["odin_state"] = state
            st.rerun()

    with col_app3:
        if current_approval is True:
            st.success("Status: **APPROVED** by human reviewer")
        elif current_approval is False:
            st.error(f"Status: **REJECTED** ({state.human_feedback})")
        else:
            st.warning("Status: **PENDING APPROVAL** (Action required above)")

    # Optional feedback comment box
    feedback_input = st.text_input(
        "Reviewer Comments / Audit Notes:",
        value=state.human_feedback or "",
        placeholder="Add operational notes or compliance comments here..."
    )
    if feedback_input != (state.human_feedback or ""):
        state.human_feedback = feedback_input
        st.session_state["odin_state"] = state

    st.markdown("---")

    # Expandable Details Tabs for All Agents
    tab_summary, tab_loki, tab_heimdall, tab_norns, tab_thor, tab_state = st.tabs([
        "⚖️ Tyr Final Judgment",
        "🦊 Loki Legal Review",
        "🛡️ Heimdall Incident Log",
        "⏳ Norns Financial Review",
        "🔨 Thor Extraction",
        "📋 Full OdinState JSON"
    ])

    with tab_summary:
        st.markdown("#### Tyr's Synthesis & Final Judgment")
        if state.final_judgment:
            st.markdown(state.final_judgment)
        else:
            st.info("No final judgment available.")

    with tab_loki:
        st.markdown("#### Loki's Adversarial Legal Interpretation")
        st.info(state.loki_interpretation or "No Loki interpretation available.")

    with tab_heimdall:
        st.markdown("#### Heimdall Operational Event Review")
        st.info(state.heimdall_review or "No Heimdall review available.")

    with tab_norns:
        st.markdown("#### Norns Financial & Deadline Compliance Report")
        st.text_area("Portfolio Timeline & Non-Renewal Analysis", value=state.financial_review or "", height=250)

    with tab_thor:
        st.markdown("#### Thor Final Retrieved Chunk & Contract Mapping")
        if state.retrieved_document:
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                st.metric("Document Name", state.retrieved_document.get("document_name", "N/A"))
            with col_t2:
                st.metric("Contract ID", state.retrieved_document.get("contract_id", "N/A"))
            with col_t3:
                st.metric("Clause ID", state.retrieved_document.get("clause_id", "N/A"))

            st.markdown("##### 📄 Full Retrieved Clause Content (`retrieved_clause_content`):")
            clause_content = (
                state.retrieved_clause_content or 
                state.retrieved_document.get("content") or 
                state.retrieved_document.get("clause_text") or 
                "No clause content available."
            )
            st.text_area("Operative Contract Clause Text", value=clause_content, height=220)

            st.markdown("##### 🔍 Full Document Metadata:")
            st.json(state.retrieved_document)
        else:
            st.write("No document retrieved.")

    with tab_state:
        st.markdown("#### Shared Common State (`OdinState`)")
        st.json(state.to_dict())

