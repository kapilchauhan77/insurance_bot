# state.py
"""Defines the state dictionary for the underwriting graph."""

from typing import TypedDict, List, Optional, Dict, Any

class GraphState(TypedDict):
    """
    Represents the state shared across nodes in the underwriting workflow.

    Attributes:
        document_paths: List of paths to the input image documents.
        extracted_text: Text extracted from the documents via OCR.
        guidelines: Relevant underwriting guidelines retrieved from Vertex AI Search.
        missing_fields: List of fields identified as missing during completeness check.
        user_prompt_message: Message generated to prompt the user for missing info.
        user_input: Information provided by the user in response to the prompt.
        gathered_context: Consolidated information for downstream tasks.
        tavily_results: Results from the Tavily web search.
        case_study_results: Relevant case studies found via Vertex AI Search.
        underwriting_decision: The final decision (e.g., Approve, Decline, Refer).
        rate_card: Generated pricing or rate information.
        reasoning: Explanation for the decision, citing sources.
        error: Optional field to capture errors during node execution.
    """
    document_paths: List[str]
    extracted_text: Optional[str] = None
    guidelines: Optional[str] = None
    missing_fields: str = ''
    extracted_information: Optional[str] = None
    user_prompt_message: Optional[str] = None
    user_input: Optional[str] = None
    gathered_context: Optional[str] = None
    tavily_results: Optional[str] = None
    case_study_results: Optional[str] = None
    underwriting_decision: Optional[str] = None
    rate_card: Optional[Dict[str, Any]] = None # Or adjust type as needed
    reasoning: Optional[str] = None
    output: Optional[str] = None
    error: Optional[str] = None
