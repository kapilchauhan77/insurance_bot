# nodes/gather_info_node.py
"""Node to consolidate all gathered information."""

import logging
from typing import Dict, Optional

from state import GraphState
from utils import config

logger = logging.getLogger(__name__)


def gather_all_information(state: GraphState) -> Dict[str, Optional[str]]:
    """
    Consolidates extracted text and user input into a single context string.

    Args:
        state: The current graph state.

    Returns:
        A dictionary with the updated 'gathered_context'.
    """
    logger.info("--- Starting Information Gathering Node ---")
    extracted_text = state.get("extracted_information", "N/A") # This comes from outside the graph flow (main.py)
    guidelines = open(config.GUIDELINE_PATH, 'r').read()

    context_parts = [
        "--- Extracted Document Text ---",
        extracted_text,
        "\n--- Underwriting Guidelines Context ---",
        guidelines 
    ]

    gathered_context = "\n".join(context_parts)

    logger.info("--- Information Gathering Complete ---")
    return {"gathered_context": gathered_context}
