# nodes/vision_node.py
"""Node to extract text from documents using Google Cloud Vision."""

import logging
from typing import Dict, List, Optional

from utils.gemini_ocr import extract_text_with_gemini_flash

from state import GraphState

logger = logging.getLogger(__name__)

def extract_text_from_documents(state: GraphState) -> Dict[str, Optional[str]]:
    """
    Extracts text from image documents using Google Cloud Vision OCR.

    Args:
        state: The current graph state containing document_paths.

    Returns:
        A dictionary with the updated 'extracted_text' or 'error'.
    """
    logger.info("--- Starting Text Extraction Node ---")
    document_paths = state.get("document_paths")
    if not document_paths:
        logger.warning("No document paths found in state.")
        return {"error": "No document paths provided."}

    all_text = []

    for doc_path in document_paths:
        try:
            logger.info(f"Processing document: {doc_path}")
            response = extract_text_with_gemini_flash(doc_path)
            all_text.append(response)
            logger.info(f"Successfully extracted text from: {doc_path}")
        except FileNotFoundError:
            logger.error(f"Document not found: {doc_path}")
            return {"error": f"Document not found: {doc_path}"}
        except Exception as e:
            logger.exception(f"Unexpected error processing {doc_path}: {e}")
            return {"error": f"Unexpected error processing {doc_path}: {e}"}

    extracted_text = " ".join(all_text)
    print(extracted_text, file=open("ocr.txt", 'w'))
    logger.info(f"--- Text Extraction Complete ({len(all_text)} docs) ---")
    return {"extracted_text": extracted_text, "error": None} # Clear previous errors
