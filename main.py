# main.py
"""Example invocation and execution logic for the underwriting graph."""

import logging
import uuid
import os
import json
from pathlib import Path
from pprint import pprint
import shutil # Added for cleaning up temporary directories
import tempfile # Added for creating temporary directories
import fitz # PyMuPDF: Added for PDF processing
from pdf2image import convert_from_path
from PyPDF2 import PdfReader, PdfWriter


# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the compiled graph and state definition using relative imports
from graph import app
from state import GraphState
from utils import config

def split_pdf(pdf_path, output_folder):
    """
    Splits a multi-page PDF into multiple single-page PDFs.

    :param pdf_path: Path to the multi-page PDF file.
    :param output_folder: Folder to save the split PDFs.
    :return: List of file paths for the split PDFs.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Read the original PDF
    pdf_reader = PdfReader(pdf_path)
    total_pages = len(pdf_reader.pages)
    split_files = []

    for page_num in range(total_pages):
        pdf_writer = PdfWriter()
        pdf_writer.add_page(pdf_reader.pages[page_num])

        # Generate a new filename
        split_pdf_path = os.path.join(output_folder, f"page_{page_num + 1}.pdf")
        with open(split_pdf_path, "wb") as output_pdf:
            pdf_writer.write(output_pdf)

        split_files.append(split_pdf_path)

    return split_files


def pdf_to_images(pdf_path, temp_folder):
    """Converts a PDF to images"""
    images = convert_from_path(pdf_path)
    image_paths = []

    for i, img in enumerate(images):
        img_path = os.path.join(temp_folder, f"page_{i + 1}.jpg")
        img.convert('RGB').save(img_path, "JPEG")
        image_paths.append(img_path)

    return images, image_paths

# --- Constants ---
SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}
SUPPORTED_PDF_EXTENSIONS = {'.pdf'} # Define PDF extension

# --- Helper function for user-friendly status messages ---
NODE_STATUS_MESSAGES = {
    "vision_node": "🔬 Extracting text from documents (OCR)...",
    "vertex_guideline_node": "📚 Retrieving underwriting guidelines...",
    "completeness_check_node": "✅ Checking document completeness...",
    "user_prompt_node": "⚠️ Preparing request for missing information...",
    "gather_info_node": "✍️ Consolidating information...",
    "tavily_node": "🌐 Performing external web search (Tavily)...",
    "vertex_case_study_node": "🔎 Searching for relevant case studies...",
    "gemini_node": "🧠 Generating underwriting decision with AI...",
}

def get_user_friendly_status(node_name: str) -> str:
    """Returns a user-friendly message for a given node name."""
    return NODE_STATUS_MESSAGES.get(node_name, f"⚙️ Completed step: {node_name}")

# --- Main Workflow Function ---
# run_underwriting_workflow function remains the same as before

# --- Main Workflow Function ---
# ... (run_underwriting_workflow remains the same) ...
def run_underwriting_workflow(document_paths: list[str]):
    """
    Initializes and runs the underwriting workflow, handling interruptions.

    Args:
        document_paths: List of paths to the input documents (images or extracted images).
    """
    if not document_paths:
        logger.error("No document paths provided to start the workflow.")
        print("❌ Error: No document paths were provided.")
        return

    thread_id = str(uuid.uuid4())
    config_ = {"configurable": {"thread_id": thread_id}}

    print(f"\n🚀 Starting Underwriting Workflow for Thread ID: {thread_id}")
    print(f"📄 Processing {len(document_paths)} document images (original & extracted from PDFs).")
    # Limit printing too many filenames if list is long
    if len(document_paths) < 10:
         print(f"   Files: {', '.join([Path(p).name for p in document_paths])}")
    print("-" * 30)

    initial_state: GraphState = {"document_paths": document_paths}

    # --- Graph Invocation Loop (Handles Interruptions) ---
    # --- This section remains unchanged ---
    try:
        current_state_values = None
        while True:
            logger.info(f"\n{'='*10} Invoking Graph Stream {'='*10}")
            print("⏳ Processing workflow step...")

            invocation_input = initial_state if current_state_values is None else None
            events = app.stream(input=invocation_input, config=config_, stream_mode="values")


            last_node_executed = None
            for event in events:
                current_state_values = event
                last_node_executed = list(event.keys())[0]
                print(f"  {get_user_friendly_status(last_node_executed)}")

            final_snapshot = app.get_state(config_)
            logger.info(f"Graph execution paused or finished. Last node executed: {last_node_executed}")
            logger.debug(f"Final snapshot next step: {final_snapshot.next}")

            if final_snapshot.next == ("gather_info_node",):
                # --- Interruption handling remains unchanged ---
                print("\n" + "="*15 + " ACTION REQUIRED " + "="*15)
                prompt_message = final_snapshot.values.get("user_prompt_message", "Error: Prompt message not found in state.")
                print("\nMissing Information:")
                print("-" * 20)
                print(prompt_message.split("Please provide")[0])
                print("-" * 20)
                print("Please provide the missing details below.")
                print("You can type 'SKIP' (all caps) to continue without this information.")
                print("-" * 20)
                user_response = input("Your input: ")

                if user_response.strip().upper() == "SKIP":
                    print("\n➡️ Skipping missing information as requested.")
                    logger.info("User chose to SKIP providing missing information.")
                else:
                    print("\n✅ Thank you! Adding your information to the workflow.")
                    logger.info(f"User provided input: {user_response}")

                app.update_state(config_, {"user_input": user_response})
                app.update_state(config_, {"extracted_text": final_snapshot.values.get('extracted_text', '') + '\n' +user_response})
                logger.info("State updated with user input. Resuming graph...")
                print("-" * 30)
            else:
                 logger.info("Workflow processing complete (no further interruptions).")
                 print("\n✨ Workflow processing complete.")
                 break # Exit the loop if no interruption is pending

    except Exception as e:
        # --- Error handling remains unchanged ---
        logger.exception(f"An error occurred during workflow execution for thread {thread_id}: {e}")
        print(f"\n❌ An unexpected error occurred: {e}")
        print("Attempting to retrieve the last known state...")
        try:
            last_state = app.get_state(config_)
            print("\n--- Last known state before error ---")
            pprint(last_state.values)
        except Exception as state_err:
            logger.error(f"Could not retrieve state after error: {state_err}")
            print("❌ Error: Could not retrieve the final state after the error.")
        # Important: Return here so cleanup below still happens in the main block
        return
    finally:
        # --- Final Output section moved to the main block after cleanup ---
        pass # Cleanup is handled outside the function call now

    # Get and print final state *after* successful completion
    final_state_values = app.get_state(config_).values
    print("\n" + "="*30)
    print("      FINAL UNDERWRITING RESULT")
    print("="*30)
    # ...(Printing logic remains the same)...
    decision = final_state_values.get('output', 'N/A')

    print(decision)

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Professional Indemnity Underwriting Assistant ---")

    input_folder_path = None
    while True:
        # --- Folder input logic remains the same ---
        folder_str = input("Enter the full path to the folder containing documents:\n> ")
        potential_path = Path(folder_str.strip())
        print(potential_path)

        if not potential_path.is_dir():
            print(f"❌ Error: '{folder_str}' is not a valid directory. Please try again.")
            continue
        else:
            print(f"✅ Valid folder found: {potential_path}")
            input_folder_path = potential_path
            break

    # Scan the folder for supported files
    image_document_paths = []
    pdf_document_paths = []
    extracted_image_paths = []
    temp_dirs_to_cleanup = [] # Keep track of temp dirs for cleanup

    print(f"\n🔎 Scanning folder for supported document types...")
    for item in input_folder_path.iterdir():
        if item.is_file():
            ext_lower = item.suffix.lower()
            if ext_lower in SUPPORTED_IMAGE_EXTENSIONS:
                image_document_paths.append(str(item))
            elif ext_lower in SUPPORTED_PDF_EXTENSIONS:
                pdf_document_paths.append(str(item)) # Store PDF paths

    if image_document_paths:
        print(f"  Found {len(image_document_paths)} supported image file(s).")
    if pdf_document_paths:
        print(f"  Found {len(pdf_document_paths)} PDF file(s).")

    # --- Process PDFs to Extract Images ---
    if pdf_document_paths:
        print(f"\n📄 Extracting images from {len(pdf_document_paths)} PDF file(s)...")
        for pdf_path_str in pdf_document_paths:
            pdf_path = Path(pdf_path_str)
            pdf_temp_dir = None # Initialize for error handling scope
            # Create a unique temporary directory for this PDF's images
            pdf_temp_dir = Path(tempfile.mkdtemp(prefix=f"pdf_{pdf_path.stem}_"))
            temp_dirs_to_cleanup.append(pdf_temp_dir) # Add to cleanup list *before* potential errors
            logger.info(f"Created temp dir for {pdf_path.name}: {pdf_temp_dir}")

            print(f"  Processing PDF: {pdf_path.name}")
            doc = fitz.open(pdf_path_str)
            split_pdfs = split_pdf(pdf_path_str, pdf_temp_dir)
            all_images = []

            for single_pdf in split_pdfs:
                images, image_paths = pdf_to_images(single_pdf, pdf_temp_dir)
                all_images.extend(images)
                extracted_image_paths.extend(image_paths)

            img_count_pdf = len(all_images)


    # --- Combine and Run Workflow ---
    all_document_paths = image_document_paths + extracted_image_paths

    if all_document_paths:
        print(f"\n✅ Ready to process {len(all_document_paths)} total images (original + extracted).")
        print("IMPORTANT: Ensure API keys and configuration are set correctly.")
        try:
            # Call the main workflow function with the combined list
            run_underwriting_workflow(all_document_paths)
        finally:
            # --- Cleanup Temporary Directories ---
            if temp_dirs_to_cleanup:
                print("\n🧹 Cleaning up temporary extracted image directories...")
                for temp_dir in temp_dirs_to_cleanup:
                    try:
                        shutil.rmtree(temp_dir)
                        logger.info(f"Successfully removed temp dir: {temp_dir}")
                        # print(f"  Removed: {temp_dir}") # Optional: uncomment for verbose cleanup
                    except Exception as e:
                        print(f"  ⚠️ Error removing temporary directory {temp_dir}: {e}")
                        logger.warning(f"Failed to remove temporary directory {temp_dir}: {e}")
    else:
        print("\nWorkflow cannot start as no supported image or PDF documents were found or processed in the folder.")

    print("\n--- End of Session ---")
