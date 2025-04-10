# nodes/tavily_node.py
"""Node for external information verification using Tavily Search."""

import logging
from typing import Dict, Optional

from tavily import TavilyClient

from state import GraphState
from utils import config

logger = logging.getLogger(__name__)


def perform_external_search(state: GraphState) -> Dict[str, Optional[str]]:
    """
    Performs a web search using Tavily based on gathered context.

    Args:
        state: The current graph state containing gathered_context.

    Returns:
        A dictionary with 'tavily_results' or 'error'.
    """
    logger.info("--- Starting External Search (Tavily) Node ---")
    api_key = config.TAVILY_API_KEY
    context = state.get("gathered_context")

    if "PLACEHOLDER" in api_key:
        logger.warning("Tavily API Key is a placeholder. Skipping search.")
        return {
            "tavily_results": "Placeholder: Tavily search skipped due to placeholder API key.",
            "error": None
        }

    if not context:
        logger.warning("No gathered context available for Tavily search.")
        return {"tavily_results": "No context provided for search.", "error": None}

    query = f"Verify professional indemnity information for: {context[:200]}"
    logger.info(f"Tavily search query (truncated): {query}")

    try:
        tavily = TavilyClient(api_key=api_key)
        response = tavily.search(query=query, search_depth="advanced", max_results=5)

        # --- MODIFIED SECTION ---
        # Use .get() for safer access to potentially missing keys
        results_list = []
        search_results = response.get('results', []) # Safely get the results list
        print(search_results, file=open("raw_tavily.txt", 'w'))

        for res in search_results:
            # Provide default values ('N/A' or similar) if a key is missing
            title = res.get('title', 'No Title Available')
            snippet = res.get('content', 'No Snippet Available') # <-- Fixes the error
            url = res.get('url', '#') # Also make url access safer
            results_list.append(f"- {title}: {snippet} ({url})")

        results_str = "\n".join(results_list)
        # --- END MODIFIED SECTION ---

        logger.info(f"Tavily search successful. Found {len(search_results)} results.")
        logger.info("--- External Search (Tavily) Complete ---")
        print(results_str, file=open('tavify_api.txt', 'w'))
        return {"tavily_results": results_str or "No results found.", "error": None}

    except Exception as e:
        logger.error(f"Tavily API client error: {e}") # Log the specific TavilyError
        return {"tavily_results": None, "error": f"Tavily API client error: {e}"}
