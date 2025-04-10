# nodes/completeness_check_node.py
"""
Node to check if extracted text contains required fields using Gemini.
"""

import logging
import json
from typing import Dict, List, Optional

import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.api_core.exceptions import GoogleAPICallError

from state import GraphState
from utils import config

logger = logging.getLogger(__name__)

extract_data = """
You are an expert business analyst tasked with extracting and organizing key information from a detailed business description. Your goal is to provide a structured summary of the business based on specific categories.

Here is the business description you need to analyze:

<business_description>
{TEXT}
</business_description>

Your task is to carefully read through this description and extract relevant information for the following categories:

1. Firm/Business Details
2. Services & Operations
3. Personnel
4. Financials
5. Risk Management & Quality Control
6. Insurance & Claims History
7. Coverage Requirements

Instructions:
1. Analyze the text thoroughly, identifying information relevant to each category.
2. Extract and organize the information into bullet points under each category heading.
3. If specific information for a category or subcategory is not present in the text, write "Information not provided" for that item.
4. After completing the extraction, create a list of any fields or subcategories for which information was not available in the text.

Before providing your final output, break down your thought process for each category inside <information_breakdown> tags. This will ensure a comprehensive interpretation of the data. In your breakdown:
1. List all the categories and potential subcategories you need to look for.
2. For each category:
   - Quote relevant text from the business description.
   - Identify any ambiguities or unclear information.
   - Note any subcategories or fields that might be relevant but are not mentioned.
It's OK for this section to be quite long.

After your breakdown, provide your extracted information in the following format:

<extracted_information>
1. Firm/Business Details:
   • [Extracted information]
   • [Extracted information]
   ...

2. Services & Operations:
   • [Extracted information]
   • [Extracted information]
   ...

[Continue with remaining categories]
</extracted_information>


<missing_fields>
[List any fields or subcategories for which information was not provided]
</missing_fields>

Be thorough in your extraction, ensuring you capture all relevant details provided in the text. If you encounter any ambiguous information, include it and note the ambiguity. Do not add any information that is not explicitly stated or directly implied in the given business description.

Please proceed with your information breakdown and extraction.
"""


def check_document_completeness(state: GraphState):
    """
    Checks document completeness using the Gemini model.

    Args:
        state: The current graph state containing extracted_text.

    Returns:
        A dictionary with the updated 'missing_fields' list or 'error'.
    """
    logger.info("--- Starting Completeness Check Node (Gemini) ---")
    extracted_text = state.get("extracted_text")
    required_fields = config.REQUIRED_FIELDS
    project_id = config.GCP_PROJECT_ID
    location = config.GCP_LOCATION
    model_name = config.GEMINI_MODEL_NAME

    if not extracted_text:
        logger.warning("Extracted text is empty. Cannot perform Gemini completeness check.")
        # If no text, assume all required fields are missing
        return {
            "missing_fields": '\n'.join(list(required_fields)), 
            'extracted_information': '',
            "error": "Extracted text was empty."
        }

    if not required_fields:
        logger.warning("No required fields defined in config. Skipping completeness check.")
        return {"missing_fields": '','extracted_information': '', "error": None} # Assume complete if no fields are required

    # Check for placeholder configurations
    if "PLACEHOLDER" in project_id or "PLACEHOLDER" in model_name:
        logger.warning("Placeholder GCP Project ID or Gemini Model Name detected. Skipping Gemini check.")
        # Fallback: Return empty list (assume complete) or potentially all fields as missing?
        # Returning empty is less disruptive if placeholders are accidentally left.
        return {
            "missing_fields": '', # Or: list(required_fields) if you prefer to halt on placeholders
            'extracted_information': '',
            "error": "Skipped Gemini check due to placeholder config."
        }

    logger.info(f"Initializing Vertex AI for completeness check (Project: {project_id}, Location: {location})")
    vertexai.init(project=project_id, location=location)
    model = GenerativeModel(model_name)
    logger.info(f"Using model: {model_name}")


    # Create the prompt
    prompt = extract_data.format(
        TEXT=extracted_text
    )

    logger.info(f"Calling Gemini to check document completeness with prompt: {prompt}")
    # Use lower temperature for more deterministic checking
    # Enforce JSON output
    response = model.generate_content(
        [Part.from_text(prompt)],
        generation_config={
            "temperature": 0.1
         }
    )

    logger.info(f"Gemini response received for completeness check: {response.text}")
    resp = response.text
    extracted_information = resp.split('<extracted_information>')[-1].split('</extracted_information>')[0]
    missing_fields = resp.split('<missing_fields>')[-1].split('</missing_fields>')[0]
    return {'missing_fields': missing_fields, 'extracted_information': extracted_information, 'error': None}
