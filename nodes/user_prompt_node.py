import logging

from state import GraphState

logger = logging.getLogger(__name__)

def prepare_user_prompt(state: GraphState):
    """
    Formats a message listing missing fields for the user.

    Args:
        state: The current graph state containing missing_fields.

    Returns:
        A dictionary with the 'user_prompt_message'.
    """
    logger.info("--- Preparing User Prompt Node ---")
    missing_fields = state.get("missing_fields")

    if len(missing_fields) == 0:
        logger.info("No missing fields, user prompt not needed (though node was called).")
        return {"user_prompt_message": None}

    prompt_message = "The following required information is missing or couldn't be identified:\n"
    prompt_message += missing_fields
    prompt_message += "\nPlease provide the missing information, or type 'SKIP' if unavailable:"

    logger.info(f"Generated user prompt: {prompt_message}")
    logger.info("--- User Prompt Preparation Complete ---")
    # The actual pause/input happens via LangGraph's interrupt mechanism
    return {"user_prompt_message": prompt_message}
