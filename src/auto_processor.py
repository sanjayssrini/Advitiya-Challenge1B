# auto_processor.py
"""
Complete automated workflow for PDF processing:
1. Extract text from PDFs (using your extractor.py)
2. Classify sections using trained model
3. Generate outlines
"""

import os
from pathlib import Path
import subprocess

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
import sys
from main_classifier import PDFSectionClassifier
from outline_builder import OutlineBuilder
import json
import re

def run_extractor():
    """Run the PDF extractor to generate JSON files"""
    print("Step 1: Extracting text from PDFs...")
    
    if not os.path.exists(os.path.join(PROJECT_ROOT, "src", "extractor.py")):
        print("Error: extractor.py not found!")
        return False
    
    if not os.path.exists(os.path.join(PROJECT_ROOT, "PDFs")):
        print("Error: PDFs folder not found!")
        return False
    
    try:
        # Import and run your extractor
        import extractor
        extractor.process_all_pdfs()
        print("✓ PDF extraction completed")
        return True
    except Exception as e:
        print(f"✗ Error running extractor: {e}")
        return False

def validate_heading_hierarchy(json_path):
    """Validate and fix potential heading level issues"""
    print(f"\nValidating heading hierarchy in {os.path.basename(json_path)}...")
    
    with open(json_path, 'r') as f:
        blocks = json.load(f)
    
    if not isinstance(blocks, list):
        print("Error: Expected a list of blocks")
        return False
    
    # Check if document has uniform font sizes
    uniform_fonts = is_uniform_font_document(blocks)
    if uniform_fonts:
        print("Detected uniform font size document - using bold/italic for heading detection")
        
    modified = False
    prev_level = 0
    
    # Map heading labels to numeric levels
    heading_levels = {
        'title': 0,
        'H1': 1,
        'H2': 2,
        'H3': 3,
        'H4': 4,
        'Other': 99
    }
    
    # Sort blocks by page and position
    blocks.sort(key=lambda x: (x.get('page', 0), x.get('y_center', 0)))
    
    # First pass: identify potential headings in uniform font documents
    if uniform_fonts:
        for block in blocks:
            text = block.get('text', '').strip()
            if block.get('label', 'Other') == 'Other':
                # Check for bold text
                if block.get('is_bold', False):
                    if text.isupper():
                        block['label'] = 'H1'
                    else:
                        block['label'] = 'H2'
                    modified = True
                # Check for italic text
                elif block.get('is_italic', False):
                    block['label'] = 'H3'
                    modified = True
                # Check for numbered sections
                elif re.match(r'^\d+\.(\d+\.?)?\s', text):
                    block['label'] = 'H2'
                    modified = True
                # Check for common heading patterns
                elif any(text.lower().startswith(word) for word in ['chapter', 'section', 'part']):
                    block['label'] = 'H1'
                    modified = True
    
    # Second pass: fix heading hierarchy
    prev_level = 0
    for block in blocks:
        label = block.get('label', 'Other')
        curr_level = heading_levels.get(label, 99)
        
        if curr_level == 99:  # Skip non-heading content
            continue
        
        # Fix skipped heading levels
        if curr_level > prev_level + 1:
            new_level = prev_level + 1
            for level_name, level_num in heading_levels.items():
                if level_num == new_level:
                    block['label'] = level_name
                    modified = True
                    curr_level = new_level
                    break
        
        # Update for next iteration
        if curr_level != 99:
            prev_level = curr_level
    
    if modified:
        with open(json_path, 'w') as f:
            json.dump(blocks, f, indent=2)
        print("✓ Fixed heading hierarchy issues")
    
    return modified

def run_classifier():
    """Run the section classifier"""
    print("\nStep 2: Classifying sections...")
    
    # Check if model files exist in the models directory
    models_dir = os.path.join(PROJECT_ROOT, "models")
    required_files = [
        'ensemble_model.pkl',
        'feature_scaler.pkl', 
        'label_encoder.pkl',
        'feature_columns.pkl'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(models_dir, f))]
    if missing_files:
        print("Error: Missing model files. Please train the model first:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    try:
        classifier = PDFSectionClassifier()
        classifier.batch_classify(
            input_folder=os.path.join(PROJECT_ROOT, "JSONs"), 
            output_folder=os.path.join(PROJECT_ROOT, "classified_jsons")
        )
        
        # Validate heading hierarchy in all classified files
        classified_json_path = os.path.join(PROJECT_ROOT, "classified_jsons")
        classified_files = [f for f in os.listdir(classified_json_path) 
                          if f.endswith('_classified.json')]
        
        validation_errors = []
        for file in classified_files:
            try:
                validate_heading_hierarchy(os.path.join(classified_json_path, file))
            except Exception as e:
                validation_errors.append(f"Error validating {file}: {str(e)}")
        
        if validation_errors:
            print("\nValidation Errors:")
            for error in validation_errors:
                print(f"  - {error}")
            return False
            
        print("✓ Section classification completed")
        return True
    except Exception as e:
        print(f"✗ Error running classifier: {e}")
        return False

def run_outline_builder():
    """Generate outlines from classified JSONs"""
    print("\nStep 3: Generating outlines...")
    
    try:
        builder = OutlineBuilder()
        
        json_folder = "classified_jsons"
        output_folder = "outlines"
        
        if not os.path.exists(json_folder):
            print("Error: classified_jsons folder not found!")
            return False
        
        os.makedirs(output_folder, exist_ok=True)
        
        # Only process the classified files, not the outline files
        json_files = [f for f in os.listdir(json_folder) 
                    if f.endswith('.json') and not f.endswith('_outline.json')]
        
        if not json_files:
            print("No classified JSON files found!")
            return False
        
        for file in json_files:
            input_path = os.path.join(json_folder, file)
            output_path = os.path.join(output_folder, file.replace('.json', '_outline.json'))
            
            try:
                builder.save_outline(input_path, output_path)
                print(f"✓ Generated outline for {file}")
            except Exception as e:
                print(f"✗ Error processing {file}: {str(e)}")
                continue
        
        print(f"✓ Generated {len(json_files)} outlines")
        return True
        
    except Exception as e:
        print(f"✗ Error generating outlines: {e}")
        return False

def is_uniform_font_document(blocks):
    """Check if document has uniform font sizes"""
    font_sizes = set()
    for block in blocks:
        if 'font_size' in block:
            font_sizes.add(round(block['font_size'], 1))  # Round to 1 decimal to handle minor variations
    return len(font_sizes) <= 2  # Allow for up to 2 different sizes to account for minor variations

def main():
    """Main automated workflow"""
    print("=" * 60)
    print("AUTOMATED PDF PROCESSING WORKFLOW")
    print("=" * 60)
    
    # Step 1: Extract text from PDFs
    if not run_extractor():
        print("\n❌ Workflow failed at extraction step")
        return
    
    # Step 2: Classify sections
    if not run_classifier():
        print("\n❌ Workflow failed at classification step")
        return
    
    # Step 3: Generate outlines
    if not run_outline_builder():
        print("\n❌ Workflow failed at outline generation step")
        return
    
    print("\n" + "=" * 60)
    print("🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nResults:")
    print("📁 Raw JSONs: JSONs/")
    print("📁 Classified JSONs: classified_jsons/")
    print("📁 Outlines: outlines/")
    
    # Show summary
    try:
        json_count = len([f for f in os.listdir("JSONs") if f.endswith('.json')])
        classified_count = len([f for f in os.listdir("classified_jsons") if f.endswith('.json')])
        outline_count = len([f for f in os.listdir("outlines") if f.endswith('.json')])
        
        print(f"\n📊 Summary:")
        print(f"   • {json_count} PDFs extracted")
        print(f"   • {classified_count} files classified")
        print(f"   • {outline_count} outlines generated")
        
    except Exception as e:
        print(f"Could not generate summary: {e}")

if __name__ == "__main__":
    main()