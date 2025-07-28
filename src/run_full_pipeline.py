# run_pipeline_from_input.py
import os
import json
from pathlib import Path
from auto_processor import run_extractor  # Remove process_pdf from import
from extractor import process_pdf  # Import process_pdf from extractor instead
from main_classifier import PDFSectionClassifier
from outline_builder import OutlineBuilder
from task_focused_summary_builder import TaskFocusedSummaryBuilder

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
PDF_FOLDER = os.path.join(PROJECT_ROOT, "PDFs")
JSON_FOLDER = os.path.join(PROJECT_ROOT, "JSONs")


def ensure_folders():
    os.makedirs(PDF_FOLDER, exist_ok=True)
    os.makedirs(JSON_FOLDER, exist_ok=True)


def extract_pdfs_from_input(input_path):
    """Extract all PDFs listed in input.json to JSONs"""
    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    documents = input_data.get("documents", [])
    for doc in documents:
        filename = doc.get("filename")
        if not filename:
            continue

        pdf_path = os.path.join(PDF_FOLDER, filename)
        output_json_path = os.path.join(JSON_FOLDER, filename.replace(".pdf", ".json"))

        if os.path.exists(pdf_path):
            print(f"\n→ Extracting: {filename}")
            process_pdf(pdf_path, output_json_path)
        else:
            print(f"[!] Missing file: {pdf_path}")


def run_full_pipeline():
    ensure_folders()
    
    # Get input and output paths
    input_json_path = os.path.join(PROJECT_ROOT, "input", "challenge1b_input.json")
    output_json_path = os.path.join(PROJECT_ROOT, "output", "challenge1b_output.json")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    extract_pdfs_from_input(input_json_path)
    
    # Step 1: Extract text from PDFs
    if not run_extractor():
        print("❌ Pipeline failed at extraction step")
        return False

    # Step 2: Classify sections
    classifier = PDFSectionClassifier()
    try:
        classifier.batch_classify(
            input_folder=os.path.join(PROJECT_ROOT, "JSONs"),
            output_folder=os.path.join(PROJECT_ROOT, "classified_jsons")
        )
        print("✓ Section classification completed")
    except Exception as e:
        print(f"❌ Pipeline failed at classification step: {e}")
        return False

    # Step 3: Generate task-focused summary
    try:
        # Load input data
        with open(input_json_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        # Generate summary
        summary_builder = TaskFocusedSummaryBuilder()
        summary = summary_builder.process_documents(input_data)
        
        # Save output
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        
        print(f"✓ Task-focused summary generated successfully and saved to: {output_json_path}")
        return True
    
    except Exception as e:
        print(f"❌ Pipeline failed at summary generation step: {e}")
        return False


if __name__ == "__main__":
    run_full_pipeline()
