# streamlit_app.py
import streamlit as st
import os
import uuid
import json
import shutil
import tempfile
from pathlib import Path
import logging
from pprint import pprint # For potentially printing state in debug/error

# --- PDF/Image Processing Libs ---
try:
    from pdf2image import convert_from_path
except ImportError:
    st.error("`pdf2image` not installed. PDF processing disabled. Please run `pip install pdf2image` and ensure `poppler` is installed.")
    convert_from_path = None
except Exception as e: # Catch potential poppler issues early if possible
     st.error(f"Error importing `pdf2image`. Is `poppler` installed and in PATH? Error: {e}")
     convert_from_path = None


try:
    from PyPDF2 import PdfReader, PdfWriter
except ImportError:
    st.error("`PyPDF2` not installed. PDF processing disabled. Please run `pip install PyPDF2`")
    PdfReader = None
    PdfWriter = None

# Import necessary components from your LangGraph package
from graph import app, memory_saver
from state import GraphState
from utils import config

# Configure basic logging for the app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}
SUPPORTED_PDF_EXTENSIONS = {'.pdf'}

# --- Helper Functions from main.py (Adapted for Streamlit Context) ---

def split_pdf(pdf_path, output_folder):
    """Splits a multi-page PDF into multiple single-page PDFs."""
    split_files = []
    if not PdfReader: return split_files # Return empty if PyPDF2 not loaded

    try:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True) # Use Pathlib

        pdf_reader = PdfReader(pdf_path)
        total_pages = len(pdf_reader.pages)

        for page_num in range(total_pages):
            pdf_writer = PdfWriter()
            pdf_writer.add_page(pdf_reader.pages[page_num])

            split_pdf_path = output_path / f"page_{page_num + 1}.pdf"
            with open(split_pdf_path, "wb") as output_pdf:
                pdf_writer.write(output_pdf)
            split_files.append(str(split_pdf_path)) # Return string paths

    except Exception as e:
        st.warning(f"⚠️ Failed to split PDF '{Path(pdf_path).name}': {e}")
        logger.error(f"Failed to split PDF {pdf_path}: {e}", exc_info=True)
    return split_files


def pdf_to_images(pdf_path, temp_folder):
    """Converts a single-page PDF to a JPG image."""
    image_paths = []
    if not convert_from_path: return [], image_paths # Return empty if pdf2image not loaded

    try:
        output_path = Path(temp_folder)
        # Convert PDF page (should be single page from split_pdf) to image
        images = convert_from_path(pdf_path, first_page=1, last_page=1) # Process only the first (only) page
        if images:
            img = images[0] # Get the first (only) image
            # Use a name derived from the single-page PDF name
            img_path = output_path / f"{Path(pdf_path).stem}.jpg"
            img.convert('RGB').save(img_path, "JPEG")
            image_paths.append(str(img_path)) # Return string path
    except Exception as e:
        st.warning(f"⚠️ Failed to convert PDF page '{Path(pdf_path).name}' to image: {e}. Is Poppler installed correctly?")
        logger.error(f"pdf_to_images failed for {pdf_path}: {e}", exc_info=True)
    # Return the PIL image object (or empty list) and the list of paths
    return images, image_paths


# --- User-Friendly Status Messages (Same as before) ---
NODE_STATUS_MESSAGES = {
    "vision_node": "🔬 Extracting text from documents (OCR)...",
    "vertex_guideline_node": "📚 Retrieving underwriting guidelines...",
    "completeness_check_node": "✅ Checking document completeness (AI)...",
    "user_prompt_node": "⚠️ Preparing request for missing information...",
    "gather_info_node": "✍️ Consolidating information...",
    "tavily_node": "🌐 Performing external web search (Tavily)...",
    "vertex_case_study_node": "🔎 Searching for relevant case studies...",
    "gemini_node": "🧠 Generating underwriting decision with AI...",
}

def get_user_friendly_status(node_name: str) -> str:
    return NODE_STATUS_MESSAGES.get(node_name, f"⚙️ Completed step: {node_name}")

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("🤖 Professional Indemnity Underwriting Assistant")
st.markdown("Provide a folder path containing application documents (images and PDFs).")
st.warning("Ensure `poppler` is installed on your system for PDF processing (`brew install poppler` or `sudo apt install poppler-utils`).")


# --- Initialize Session State (Same as before) ---
if "workflow_thread_id" not in st.session_state: st.session_state.workflow_thread_id = None
if "latest_graph_state" not in st.session_state: st.session_state.latest_graph_state = None
if "needs_user_input" not in st.session_state: st.session_state.needs_user_input = False
if "prompt_message" not in st.session_state: st.session_state.prompt_message = ""
if "workflow_finished" not in st.session_state: st.session_state.workflow_finished = False
if "workflow_error" not in st.session_state: st.session_state.workflow_error = None
if "temp_dirs_to_cleanup" not in st.session_state: st.session_state.temp_dirs_to_cleanup = []
if "analysis_running" not in st.session_state: st.session_state.analysis_running = False

# --- Input Section (Same as before) ---
folder_path_input = st.text_input(
    "Enter the full path to the folder containing documents:",
    key="folder_path_input",
)
analyze_button = st.button("Analyze Documents", type="primary", disabled=st.session_state.analysis_running)

# Placeholders (Same as before)
status_placeholder = st.empty()
results_placeholder = st.empty()
hitl_placeholder = st.empty()

# --- Workflow Execution Logic ---
if analyze_button and folder_path_input:
    st.session_state.analysis_running = True # Disable button
    # Reset state (Same as before)
    st.session_state.workflow_thread_id = None
    st.session_state.latest_graph_state = None
    st.session_state.needs_user_input = False
    st.session_state.prompt_message = ""
    st.session_state.workflow_finished = False
    st.session_state.workflow_error = None
    st.session_state.temp_dirs_to_cleanup = []

    input_folder_path = Path(folder_path_input.strip())

    if not input_folder_path.is_dir():
        st.error(f"❌ Error: '{folder_path_input}' is not a valid directory.")
        st.session_state.analysis_running = False # Re-enable button
        st.stop()

    # --- Scan Folder and Process PDFs (Using new main.py logic) ---
    image_document_paths = []
    pdf_document_paths = []
    extracted_image_paths = []
    temp_dirs = [] # List to store Path objects of temp dirs

    # Wrap scanning and processing in try/except
    try:
        with st.spinner("Scanning folder..."):
            logger.info(f"Scanning folder: {input_folder_path}")
            for item in input_folder_path.iterdir():
                if item.is_file():
                    ext_lower = item.suffix.lower()
                    if ext_lower in SUPPORTED_IMAGE_EXTENSIONS:
                        image_document_paths.append(str(item))
                    elif PdfReader and convert_from_path and ext_lower in SUPPORTED_PDF_EXTENSIONS:
                        pdf_document_paths.append(str(item)) # Store original PDF path

        # --- Process PDFs using split_pdf and pdf_to_images ---
        if pdf_document_paths:
            st.info(f"Found {len(pdf_document_paths)} PDF(s). Processing pages...")
            logger.info(f"Starting PDF page conversion for {len(pdf_document_paths)} files.")
            pdf_progress = st.progress(0.0) # Add progress bar for PDFs

            for i, pdf_path_str in enumerate(pdf_document_paths):
                pdf_path = Path(pdf_path_str)
                pdf_temp_dir = None # Initialize
                try:
                    # Create a unique temporary directory for this PDF's processing
                    pdf_temp_dir = Path(tempfile.mkdtemp(prefix=f"pdf_{pdf_path.stem}_"))
                    temp_dirs.append(pdf_temp_dir) # Add Path object for cleanup
                    logger.info(f"Created temp dir for {pdf_path.name}: {pdf_temp_dir}")

                    status_placeholder.info(f"Processing PDF: {pdf_path.name}...")

                    # 1. Split PDF into single pages
                    split_pdf_paths = split_pdf(pdf_path_str, str(pdf_temp_dir)) # Pass string path

                    # 2. Convert each single-page PDF to an image
                    if split_pdf_paths:
                        for single_page_pdf_path in split_pdf_paths:
                             _, image_paths = pdf_to_images(single_page_pdf_path, str(pdf_temp_dir)) # Pass string path
                             extracted_image_paths.extend(image_paths) # Add paths of created JPGs

                    # Update progress bar
                    pdf_progress.progress((i + 1) / len(pdf_document_paths))

                except Exception as pdf_err:
                    st.warning(f"⚠️ Failed during processing of PDF '{pdf_path.name}': {pdf_err}")
                    logger.error(f"Failed to process PDF {pdf_path_str}: {pdf_err}", exc_info=True)
                    # Cleanup specific temp dir if error occurs mid-way? Optional.

            pdf_progress.empty() # Remove progress bar after loop

        st.session_state.temp_dirs_to_cleanup = temp_dirs # Store Path objects

        # --- Combine Paths and Check ---
        all_document_paths = image_document_paths + extracted_image_paths
        if not all_document_paths:
            st.warning("⚠️ No processable image files (original or extracted from PDFs) found.")
            st.session_state.analysis_running = False # Re-enable button
            st.stop()

        st.info(f"Found {len(all_document_paths)} images to analyze (original + extracted). Starting workflow...")

        # --- Initialize and Run Graph (Same as before) ---
        st.session_state.workflow_thread_id = str(uuid.uuid4())
        config_ = {"configurable": {"thread_id": st.session_state.workflow_thread_id}}
        initial_state: GraphState = {"document_paths": all_document_paths} # Pass combined list

        with status_placeholder.status("Running Underwriting Workflow...", expanded=True) as status:
            try:
                events = app.stream(input=initial_state, config=config_, stream_mode="values")
                last_node_executed = None
                for event in events:
                    st.session_state.latest_graph_state = event # Store latest full state
                    # Handle potential variations in event dictionary keys more safely
                    keys = list(event.keys())
                    node_keys = [k for k in keys if k != 'error' and k!= 'messages'] # Example filter
                    if node_keys:
                        # Attempt to get a meaningful node name, avoid just getting 'messages' if present
                        last_node_executed = node_keys[0] if node_keys else keys[0]
                        status.write(f"{get_user_friendly_status(last_node_executed)}")
                    else:
                        status.write("Processing...") # Fallback

                # --- Check for Interruption (Same as before) ---
                final_snapshot = app.get_state(config_)
                st.session_state.latest_graph_state = final_snapshot.values

                if final_snapshot.next == ("gather_info_node",):
                    st.session_state.needs_user_input = True
                    st.session_state.prompt_message = final_snapshot.values.get("user_prompt_message", "Error: Prompt message missing.")
                    status.update(label="⏳ Action Required: Waiting for user input...", state="running", expanded=True)
                else:
                    st.session_state.workflow_finished = True
                    st.session_state.needs_user_input = False
                    status.update(label="✅ Workflow Complete!", state="complete", expanded=False)

            except Exception as e:
                # --- Graph Error Handling (Same as before) ---
                st.session_state.workflow_error = f"An error occurred during workflow execution: {e}"
                logger.error(f"Workflow error for thread {st.session_state.workflow_thread_id}: {e}", exc_info=True)
                status.update(label="❌ Workflow Error!", state="error", expanded=True)
                # Try to get latest state even on error
                try:
                    error_snapshot = app.get_state(config_)
                    st.session_state.latest_graph_state = error_snapshot.values
                    status.warning("Attempting to show partial state before error.")
                except Exception:
                    st.session_state.latest_graph_state = None # Clear state if retrieval fails too
                status.error(st.session_state.workflow_error)
                st.session_state.workflow_finished = True # Mark as finished to show error below


    except Exception as setup_err:
         # --- Setup/Scanning Error Handling (Same as before) ---
         st.error(f"An error occurred during setup or file scanning: {setup_err}")
         logger.error(f"Setup/Scanning Error: {setup_err}", exc_info=True)
    finally:
        # --- Re-enable button (Same as before) ---
        st.session_state.analysis_running = False

# --- Human-in-the-Loop (HITL) Input Section ---
if st.session_state.get('needs_user_input'):
    with hitl_placeholder.container():
        st.warning("⚠️ Action Required: Please provide missing information.")
        st.markdown(st.session_state.prompt_message)
        user_response = st.text_area("Your input (or type 'SKIP'):", key="hitl_input")
        submit_hitl_button = st.button("Submit Missing Info", key="submit_hitl")

        if submit_hitl_button and user_response:
            st.session_state.analysis_running = True # Disable buttons
            config_ = {"configurable": {"thread_id": st.session_state.workflow_thread_id}}

            if user_response.strip().upper() == "SKIP":
                st.info("➡️ Skipping missing information as requested. Resuming workflow...")
                logger.info(f"User provided SKIP for thread {st.session_state.workflow_thread_id}")
            else:
                st.info("✅ Thank you! Adding your information. Resuming workflow...")
                logger.info(f"User provided input for thread {st.session_state.workflow_thread_id}")

            # Clear the HITL display
            hitl_placeholder.empty()
            # Display status again
            with status_placeholder.status("Resuming Underwriting Workflow...", expanded=True) as status:
                try:
                    # --- IMPORTANT: Update only user_input ---
                    # Let the graph's 'gather_info_node' handle consolidation
                    app.update_state(config_, {"user_input": user_response})
                    # DO NOT manually update extracted_text here:
                    # app.update_state(config_, {"extracted_text": ... }) # <--- REMOVED

                    # Resume the graph execution stream
                    events = app.stream(input=None, config=config_, stream_mode="values")
                    last_node_executed = None
                    for event in events:
                        st.session_state.latest_graph_state = event
                        # Safer key access again
                        keys = list(event.keys())
                        if keys:
                            node_keys = [k for k in keys if k != 'messages']
                            last_node_executed = node_keys[-1] if node_keys else keys[-1]
                            status.write(f"{get_user_friendly_status(last_node_executed)}")
                        else:
                             status.write("Processing...")

                    # Check final state after resuming
                    final_snapshot = app.get_state(config_)
                    st.session_state.latest_graph_state = final_snapshot.values
                    st.session_state.workflow_finished = True
                    st.session_state.needs_user_input = False # Turn off HITL flag
                    status.update(label="✅ Workflow Complete!", state="complete", expanded=False)

                except Exception as e:
                    # --- Resume Error Handling (Same as before) ---
                    st.session_state.workflow_error = f"An error occurred while resuming workflow: {e}"
                    logger.error(f"Resume workflow error for thread {st.session_state.workflow_thread_id}: {e}", exc_info=True)
                    try: # Try get state on resume error
                         error_snapshot = app.get_state(config_)
                         st.session_state.latest_graph_state = error_snapshot.values
                         status.warning("Attempting to show partial state before error during resume.")
                    except Exception:
                         st.session_state.latest_graph_state = None
                    status.update(label="❌ Workflow Error During Resume!", state="error", expanded=True)
                    status.error(st.session_state.workflow_error)
                    st.session_state.workflow_finished = True
                    st.session_state.needs_user_input = False

            st.session_state.analysis_running = False # Re-enable buttons

# --- Results / Error Display Section ---
# This section remains largely the same, ensuring it reads the correct keys
# (underwriting_decision, rate_card, reasoning) from the state.
if st.session_state.get('workflow_finished') and not st.session_state.get('needs_user_input'):
    with results_placeholder.container():
        final_state_values = st.session_state.latest_graph_state

        st.subheader("📊 Final Underwriting Result", divider="rainbow")

        if st.session_state.workflow_error:
             st.error(f"Workflow ended with an error: {st.session_state.workflow_error}")

        if final_state_values:
            # Use the keys expected from your Gemini node's output definition
            decision = final_state_values.get('output', 'N/A')

            st.info(f"**Output:** {decision}")


        else:
             st.warning("Could not retrieve final workflow state details (possibly due to an earlier error).")

    # --- Cleanup ---
    # Logic remains the same, using st.session_state.temp_dirs_to_cleanup
    temp_dirs = st.session_state.get("temp_dirs_to_cleanup", [])
    if temp_dirs:
        st.info("Cleaning up temporary files...")
        logger.info(f"Cleaning up {len(temp_dirs)} temporary directories.")
        cleaned_count = 0
        error_count = 0
        for temp_dir_path in temp_dirs: # Iterate through Path objects
            try:
                temp_dir_path_obj = Path(temp_dir_path) # Ensure it's a Path object
                if temp_dir_path_obj.exists():
                     shutil.rmtree(temp_dir_path_obj)
                     cleaned_count += 1
                     logger.info(f"Removed temp dir: {temp_dir_path_obj}")
            except Exception as e:
                error_count += 1
                logger.warning(f"Failed to remove temporary directory {temp_dir_path_obj}: {e}")
        st.info(f"Cleanup finished. Removed {cleaned_count} directories" + (f", {error_count} errors." if error_count else "."))
        st.session_state.temp_dirs_to_cleanup = [] # Clear list
