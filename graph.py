# graph.py
"""Defines and compiles the LangGraph for the underwriting workflow."""

import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver # Use memory saver for interruption

from state import GraphState
from nodes import (
    extract_text_from_documents,
    check_document_completeness,
    prepare_user_prompt,
    gather_all_information,
    perform_external_search,
    search_case_studies,
    generate_underwriting_decision,
)

logger = logging.getLogger(__name__)


def decide_completeness(state: GraphState) -> str:
    """Determines the next step based on document completeness."""
    logger.info("--- Checking Completeness Condition ---")
    missing_fields = state.get("missing_fields", "")
    user_response = state.get("user_input", "")
    if len(missing_fields) < 4 or user_response == "SKIP":
        logger.info("Condition: Complete -> Routing to gather_info_node")
        return "gather_info_node"
    else:
        logger.info(f"Condition: Incomplete -> Routing to user_prompt_node")
        return "user_prompt_node"

# --- Graph Definition ---

def build_graph() -> StateGraph:
    """Builds the underwriting workflow graph."""
    workflow = StateGraph(GraphState)

    # Add nodes
    logger.info("Adding nodes to the graph...")
    workflow.add_node("vision_node", extract_text_from_documents)
    workflow.add_node("completeness_check_node", check_document_completeness)
    workflow.add_node("user_prompt_node", prepare_user_prompt) # Prepares prompt for interruption
    workflow.add_node("gather_info_node", gather_all_information)
    workflow.add_node("tavily_node", perform_external_search)
    workflow.add_node("vertex_case_study_node", search_case_studies)
    workflow.add_node("gemini_node", generate_underwriting_decision)

    # Define edges
    logger.info("Defining graph edges...")
    workflow.set_entry_point("vision_node")

    workflow.add_edge("vision_node", "completeness_check_node")

    # Conditional edge based on completeness check
    workflow.add_conditional_edges(
        "completeness_check_node",
        decide_completeness,
        {
            "user_prompt_node": "user_prompt_node", # If incomplete, go to user prompt
            "gather_info_node": "gather_info_node", # If complete, skip to gather info
        },
    )

    # Edge after user prompt (interruption happens *before* gather_info_node in this path)
    workflow.add_edge("user_prompt_node", "gather_info_node")

    # Linear flow after gathering info (or skipping user prompt)
    workflow.add_edge("gather_info_node", "tavily_node")
    workflow.add_edge("tavily_node", "vertex_case_study_node")
    workflow.add_edge("vertex_case_study_node", "gemini_node")

    # End after Gemini node
    workflow.add_edge("gemini_node", END)

    logger.info("Graph definition complete.")
    return workflow

# --- Compilation ---

# Use MemorySaver for checkpointing, necessary for interruptions
memory_saver = MemorySaver()

# Compile the graph with interruption configured
logger.info("Compiling graph with interruption point before 'gather_info_node'...")
app = build_graph().compile(
    checkpointer=memory_saver,
    # Interrupt *before* the gather_info_node runs IF the path comes from user_prompt_node.
    # This ensures prepare_user_prompt runs, generates the message, then pauses.
    interrupt_before=["gather_info_node"],
    # interrupt_after=["user_prompt_node"] could also work, depending on desired state view at interrupt.
    # `interrupt_before` feels slightly more natural here - pause before the step needing user input.
)
logger.info("Graph compiled successfully.")
