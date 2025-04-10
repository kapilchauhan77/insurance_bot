# utils/config.py
"""Placeholder for configuration variables."""

import os

# --- API Keys & Secrets (Load from environment variables or secure storage) ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "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")
# Add other necessary API keys if needed (e.g., for specific GCP service accounts)

# --- Google Cloud Configuration ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "mythic-lattice-455715-q1")
# GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1") # e.g., "us-central1"
GCP_LOCATION = os.getenv("GCP_LOCATION", "global") # e.g., "us-central1"

# --- Vertex AI Search Datastore IDs ---
GUIDELINE_DATASTORE_ID = os.getenv(
    "GUIDELINE_DATASTORE_ID",
    "pi-insurance-docs_1744003036846"
)
CASE_STUDY_DATASTORE_ID = os.getenv(
    "CASE_STUDY_DATASTORE_ID",
    "pi-insurance-docs_1744003036846"
)

# --- Vertex AI Gemini Model ---
GEMINI_MODEL_NAME = os.getenv(
    "GEMINI_MODEL_NAME",
    "gemini-2.0-flash-001" # Or your preferred Gemini model
)

# --- File Paths ---
PROMPT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), # Go up one level from utils
    "prompts",
    "gemini_underwriting_prompt.txt"
)

GUIDELINE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), # Go up one level from utils
    "prompts",
    "guidelines.txt"
)

# --- Completeness Check ---
REQUIRED_FIELDS = [
    "Firm/Business Details",
    "Services & Operations",
    "Personnel",
    "Financials",
    "Risk Management & Quality Control",
    "Insurance & Claims History",
    "Coverage Requirements"
]

# --- Other Settings ---
# Add any other configuration variables needed by the nodes
