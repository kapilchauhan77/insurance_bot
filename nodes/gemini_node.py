# nodes/gemini_node.py
"""Node to generate underwriting decision using Vertex AI Gemini."""

import logging
import json
from typing import Dict, Optional, Any

from langchain_core.runnables.utils import Output

import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.api_core.exceptions import GoogleAPICallError

from state import GraphState
from utils import config

logger = logging.getLogger(__name__)

# --- Helper Function to Load Prompt ---
def load_prompt_template(file_path: str) -> Optional[str]:
    """Loads the prompt template from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Prompt template file not found: {file_path}")
        return None
    except Exception as e:
        logger.exception(f"Error reading prompt template file {file_path}: {e}")
        return None

# --- Main Node Function ---
def generate_underwriting_decision(state: GraphState) -> Dict[str, Any]:
    """
    Generates underwriting decision, rate card, and reasoning using Gemini.

    Args:
        state: The current graph state containing all gathered information.

    Returns:
        A dictionary with 'underwriting_decision', 'rate_card', 'reasoning', or 'error'.
    """
    logger.info("--- Starting Underwriting Generation (Gemini) Node ---")
    project_id = config.GCP_PROJECT_ID
    location = config.GCP_LOCATION
    model_name = config.GEMINI_MODEL_NAME
    prompt_template_path = config.PROMPT_TEMPLATE_PATH

    # Retrieve necessary data from state
    context = state.get("gathered_context", "No context available.")
    guidelines = open(config.GUIDELINE_PATH, 'r').read()
    tavily_results = state.get("tavily_results", "No external search results.")
    case_studies = state.get("case_study_results", "No case studies found.")
    if tavily_results == None or len(tavily_results) == 0: tavily_results = ''
    if guidelines == None or len(guidelines) == 0: guidelines = ''
    if case_studies == None or len(case_studies) == 0: case_studies = ''

    # Load the prompt template
    prompt_template = load_prompt_template(prompt_template_path)
    if not prompt_template:
        return {"error": f"Failed to load prompt template from {prompt_template_path}"}

    # Format the prompt using an f-string
    formatted_prompt = prompt_template.format(
        gathered_context=context.replace('{', '[').replace('}', ']'),
        guidelines=guidelines.replace('{', '[').replace('}', ']'),
        tavily_results=tavily_results.replace('{', '[').replace('}', ']'),
        case_study_results=case_studies.replace('{', '[').replace('}', ']')
    )
    logger.info(f"Formatted prompt: {formatted_prompt}")

    if "PLACEHOLDER" in project_id or "PLACEHOLDER" in model_name:
        logger.warning("Placeholder GCP Project ID or Gemini Model Name detected.")
        # Return dummy data
        return {
             "underwriting_decision": "Placeholder Decision",
             "rate_card": {"placeholder_rate": 0.0},
             "reasoning": "Placeholder: Gemini generation skipped due to placeholder config.",
             "error": None
         }

    try:
        logger.info(f"Initializing Vertex AI for project {project_id} in {location}")
        vertexai.init(project=project_id, location=location)

        logger.info(f"Loading Gemini model: {model_name}")
        model = GenerativeModel('gemini-2.5-pro-preview-03-25')

        logger.info("Generating content with Gemini...")
        # Consider adding safety_settings if needed
        response = model.generate_content(
            [Part.from_text(formatted_prompt)],
            generation_config={ 
                "temperature": 0.4,
            }
        )

        logger.info("Gemini response received.")
        # Assuming the prompt asks Gemini to return JSON like:
        # { "decision": "...", "rate_card": {...}, "reasoning": "..." }
        output = (response.text)

        logger.info("--- Underwriting Generation Complete ---")
        return {
            "output": output,
            "error": None # Clear previous errors
        }

    except GoogleAPICallError as e:
        logger.error(f"Vertex AI Gemini API error: {e}")
        return {"error": f"Vertex AI Gemini API error: {e}"}
    except json.JSONDecodeError as e:
         logger.error(f"Failed to parse Gemini JSON response: {e}\nResponse text: {response.text[:500]}...") # Log part of the raw response
         return {"error": f"Failed to parse Gemini JSON response: {e}"}
    except Exception as e:
        logger.exception(f"Unexpected error during Gemini generation: {e}")
        return {"error": f"Unexpected error during Gemini generation: {e}"}
