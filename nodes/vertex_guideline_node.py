# nodes/vertex_guideline_node.py
"""Node to retrieve underwriting guidelines from Vertex AI Search."""

import logging
from typing import Dict, Optional

from google.cloud import discoveryengine_v1alpha as discoveryengine
from google.api_core.exceptions import GoogleAPICallError

from state import GraphState
from utils import config

logger = logging.getLogger(__name__)

def retrieve_guidelines(state: GraphState) -> Dict[str, Optional[str]]:
    """
    Queries Vertex AI Search for relevant underwriting guidelines.

    Args:
        state: The current graph state. Uses extracted_text for context if available.

    Returns:
        A dictionary with the updated 'guidelines' or 'error'.
    """
    logger.info("--- Starting Guideline Retrieval Node ---")
    # Use extracted text or a default query if text extraction failed
    query_text = state.get("extracted_text") or "professional indemnity underwriting guidelines"
    project_id = config.GCP_PROJECT_ID
    location = config.GCP_LOCATION # Use global location for Vertex AI Search API calls
    datastore_id = config.GUIDELINE_DATASTORE_ID

    if "PLACEHOLDER" in project_id or "PLACEHOLDER" in datastore_id:
         logger.warning("Placeholder GCP Project ID or Guideline Datastore ID detected.")
         # Return dummy data or skip if placeholders are present
         return {
             "guidelines": "Placeholder: Guideline retrieval skipped due to placeholder config.",
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
        serving_config="default_config", # Usually 'default_config'
    )

    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=query_text,
        page_size=5, # Adjust as needed
        # Add other parameters like query_expansion_spec if desired
    )

    try:
        logger.info(f"Querying Guideline Datastore: {datastore_id}")
        response = client.search(request)

        guidelines_content = []
        for result in response.results:
             # Adjust based on your datastore structure (e.g., unstructured, structured)
             # This example assumes unstructured data stored in 'content' field
             data = result.document.derived_struct_data
             if data and 'link' in data:
                 guidelines_content.append(f"Source: {data['link']}")
             if data and 'snippets' in data:
                snippet_text = " ".join([s.get('snippet', '') for s in data['snippets']])
                if snippet_text:
                    guidelines_content.append(f"Snippet: {snippet_text}")
             # Add content extraction from result.document.struct_data if needed

        retrieved_guidelines = "\n---\n".join(guidelines_content) if guidelines_content else "No relevant guidelines found."
        logger.info(f"Retrieved {len(guidelines_content)} guideline snippets.")
        logger.info("--- Guideline Retrieval Complete ---")
        return {"guidelines": retrieved_guidelines, "error": None}

    except GoogleAPICallError as e:
        logger.error(f"Vertex AI Search API error: {e}")
        return {"guidelines": None, "error": f"Vertex AI Search API error: {e}"}
    except Exception as e:
        logger.exception(f"Unexpected error retrieving guidelines: {e}")
        return {"guidelines": None, "error": f"Unexpected error retrieving guidelines: {e}"}
