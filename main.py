import argparse
import asyncio
import time
from langgraph.graph import StateGraph, END
from langgraph.types import Send

from src import utils
from src.state import AgentState
from src.nodes import (
    input_router_node,
    extract_candidate_skills_from_source_node,
    canonicalize_skills_node,
    match_capabilities_node,
    triage_job_node,
    extract_job_requirements_node,
    skill_gap_analysis_node,
    analysis_path_node,
    build_rag_corpus_node,
    deep_analysis_node,
    retrieve_explanatory_context_node,
    light_report_node,
    professional_requirements_report_node
)

def map_sources(state: AgentState):
    return [
        Send("extract_source_skills", {"source_id": sid, "source_data": data})
        for sid, data in state["available_sources"].items()
    ]

def create_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("input_router_node", input_router_node)
    builder.add_node("extract_source_skills", extract_candidate_skills_from_source_node)

    builder.add_node("triage", triage_job_node)
    builder.add_node("extract_requirements", extract_job_requirements_node)
    builder.add_node("skill_gap", skill_gap_analysis_node)
    builder.add_node("canonical", canonicalize_skills_node)
    builder.add_node("final_match", match_capabilities_node)
    builder.add_node("analysis_path", analysis_path_node)

    builder.add_node("rag_init", build_rag_corpus_node)
    builder.add_node("retrieve_evidence", retrieve_explanatory_context_node)
    builder.add_node("deep_analysis", deep_analysis_node)

    builder.add_node("light_report", light_report_node)
    builder.add_node("soft_report", professional_requirements_report_node)


    builder.set_entry_point("input_router_node")

    builder.add_conditional_edges(
        "input_router_node",
        map_sources,
        ["extract_source_skills"]
    )

    builder.add_edge("extract_source_skills", "triage")

    builder.add_conditional_edges(
        "triage",
        lambda s: s["triage_decision"],
        {
            "REJECT": "light_report",
            "ACCEPT": "extract_requirements",
        }
    )

    builder.add_edge("extract_requirements", "skill_gap")
    builder.add_edge("skill_gap", "canonical")
    builder.add_edge("canonical", "final_match")
    builder.add_edge("final_match", "analysis_path")

    builder.add_conditional_edges(
        "analysis_path",
        lambda s: s["analysis_path"],
        {
            "LIGHT": "light_report",
            "DEEP": "rag_init",
        }
    )

    builder.add_edge("rag_init", "retrieve_evidence")
    builder.add_edge("retrieve_evidence", "deep_analysis")
    builder.add_edge("deep_analysis", "soft_report")

    builder.add_edge("light_report", "soft_report")
    builder.add_edge("soft_report", END)

    return builder.compile()


async def main():
    parser = argparse.ArgumentParser(description="AI Recruiter Agent")
    parser.add_argument("--cv", required=True, help="Path to CV PDF")
    parser.add_argument("--job", help="Path to Job Description PDF/TXT")
    parser.add_argument("--github", help="Path to GitHub directory")
    parser.add_argument("--publications", help="Path to Publications directory")
    args = parser.parse_args()

    cv_text = utils.extract_text_from_pdf(args.cv)

    if args.job:
        job_text = utils.extract_text_from_pdf(args.job)
    else:
        print("Incolla la Job Description (Ctrl+D per terminare):")
        job_text = utils.read_multiline_input()

    available_sources = {
        "cv": {"path": args.cv, "type": "cv"}
    }
    if args.github:
        available_sources["github"] = {"path": args.github, "type": "documentation"}
    if args.publications:
        available_sources["publications"] = {"path": args.publications, "type": "academic"}

    initial_state = utils.make_initial_state(cv_text, job_text, available_sources)

    print("\nAVVIO ANALISI AGENTE...\n")
    graph = create_graph()

    start_time = time.perf_counter()
    final_state = await graph.ainvoke(initial_state)
    end_time = time.perf_counter()

    print("\n" + "=" * 50)
    print("RISULTATO FINALE")
    print("=" * 50)
    print(f"Match Score: {final_state.get('match_score', 0.0):.2%}")
    print(f"\nReport breve:\n{final_state.get('report', 'Nessun report generato')}")

    utils.render_explanation(final_state)
    print(f"\nTempo totale: {end_time - start_time:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())