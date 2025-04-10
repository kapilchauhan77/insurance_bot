# nodes/vertex_case_study_node.py
"""Node to retrieve relevant case studies from Vertex AI Search."""

import logging
from typing import Dict, Optional

from google.cloud import discoveryengine_v1alpha as discoveryengine
from google.api_core.exceptions import GoogleAPICallError

from state import GraphState
from utils import config

logger = logging.getLogger(__name__)

def search_case_studies(state: GraphState) -> Dict[str, Optional[str]]:
    """
    Queries a *different* Vertex AI Search datastore for relevant case studies.

    Args:
        state: The current graph state, using gathered_context for the query.

    Returns:
        A dictionary with 'case_study_results' or 'error'.
    """
    logger.info("--- Starting Case Study Search Node ---")
    query_text = state.get("gathered_context") # Use the consolidated context
    project_id = config.GCP_PROJECT_ID
    location = config.GCP_LOCATION # Use global location for Vertex AI Search API calls
    datastore_id = config.CASE_STUDY_DATASTORE_ID # Use the specific case study datastore ID

    if not query_text:
         logger.warning("No gathered context for case study search.")
         return {"case_study_results": "No context for search.", "error": None}

    if "PLACEHOLDER" in project_id or "PLACEHOLDER" in datastore_id:
         logger.warning("Placeholder GCP Project ID or Case Study Datastore ID detected.")
         return {
             "case_study_results": "Placeholder: Case study search skipped due to placeholder config.",
              "error": None
         }

    client_options = (
        {"api_endpoint": f"{location}-discoveryengine.googleapis.com"}
        if location != "global" else {}
    )
    client = discoveryengine.SearchServiceClient(client_options=client_options)

    serving_config = client.serving_config_path(
        project=project_id,
        location=location,
        data_store=datastore_id,
        serving_config="default_config",
    )

    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=query_text[:1000], # Limit query length for safety
        page_size=5, # Limit number of case studies
    )

    try:
        logger.info(f"Querying Case Study Datastore: {datastore_id}")
        response = client.search(request)
        print(' '.join([str(res) for res in response.results]), file=open('vector_search.txt', 'w'))

        case_studies_content = []
        for result in response.results:
             # Adjust based on your datastore structure
             data = result.document.derived_struct_data
             title = data.get('title', 'N/A') if data else 'N/A'
             link = data.get('link', 'N/A') if data else 'N/A'
             snippet = " ".join([s.get('content', '') for s in data.get('extractive_answers', [])]) if data else 'N/A'
             case_studies_content.append(f"Case Study: {title}\nLink: {link}\nSnippet: {snippet}")

        results_str = "\n---\n".join(case_studies_content) if case_studies_content else "No relevant case studies found."
        logger.info(f"Retrieved {len(case_studies_content)} case studies.")
        logger.info("--- Case Study Search Complete ---")
        print(results_str, file=open('case_study_results.txt', 'w'))
        return {"case_study_results": results_str, "error": None}

    except GoogleAPICallError as e:
        logger.error(f"Vertex AI Search API error (Case Studies): {e}")
        return {"case_study_results": None, "error": f"Vertex AI Search API error (Case Studies): {e}"}
    except Exception as e:
        logger.exception(f"Unexpected error searching case studies: {e}")
        return {"case_study_results": None, "error": f"Unexpected error searching case studies: {e}"}
