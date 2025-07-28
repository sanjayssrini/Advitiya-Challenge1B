# outline_builder.py
import os
from pathlib import Path
import json
import re

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

class OutlineBuilder:
    def __init__(self):
        self.level_hierarchy = {
            'title': 0,
            'H1': 1,
            'H2': 2, 
            'H3': 3,
            'H4': 4,
            'Body': 5,
            'Other': 6
        }
    
    def clean_text(self, text):
        """Clean text by removing unwanted characters or garbled content."""
        if not text:
            return ""
        # Remove extra spaces, line breaks, and non-printable characters
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = re.sub(r'[^\x20-\x7E]', '', text)  # Remove non-printable characters
        return text
    
    def merge_similar_blocks(self, blocks, level):
        """Merge consecutive blocks of same level with similar features on same page"""
        if not blocks:
            return blocks
        
        # Filter blocks for this specific level
        level_blocks = [block for block in blocks if block['label'] == level]
        if not level_blocks:
            return blocks
        
        # Sort by page, then by y_center
        level_blocks.sort(key=lambda x: (x['page'], x['y_center']))
        
        merged_blocks = []
        i = 0
        
        while i < len(level_blocks):
            current = level_blocks[i].copy()
            j = i + 1
            
            # Look for consecutive similar blocks to merge
            while j < len(level_blocks):
                next_block = level_blocks[j]
                
                # Check if blocks should be merged
                same_page = current['page'] == next_block['page']
                similar_font_size = abs(current['font_size'] - next_block['font_size']) <= 1
                similar_font_name = current.get('font_name', '') == next_block.get('font_name', '')
                
                # Check vertical proximity (reasonable gap between lines)
                y_gap = abs(next_block['y_center'] - current['y_center'])
                reasonable_gap = y_gap <= 50  # Adjust this threshold as needed
                
                # Get font weight safely
                current_weight = current.get('Font_weight', current.get('font_weight', 400))
                next_weight = next_block.get('Font_weight', next_block.get('font_weight', 400))
                similar_weight = abs(current_weight - next_weight) <= 100
                
                if (same_page and similar_font_size and similar_font_name and 
                    reasonable_gap and similar_weight):
                    
                    # Merge the text
                    current['text'] = current['text'] + ' ' + next_block['text']
                    
                    # Update bounding box if available
                    if 'bbox' in current and 'bbox' in next_block:
                        current['bbox'] = [
                            min(current['bbox'][0], next_block['bbox'][0]),
                            min(current['bbox'][1], next_block['bbox'][1]),
                            max(current['bbox'][2], next_block['bbox'][2]),
                            max(current['bbox'][3], next_block['bbox'][3])
                        ]
                        # Update center coordinates
                        current['x_center'] = (current['bbox'][0] + current['bbox'][2]) / 2
                        current['y_center'] = (current['bbox'][1] + current['bbox'][3]) / 2
                    
                    j += 1
                else:
                    break
            
            merged_blocks.append(current)
            i = j
        
        # Replace original blocks with merged ones
        other_blocks = [block for block in blocks if block['label'] != level]
        return other_blocks + merged_blocks

    def extract_title(self, blocks):
        """Extract document title from blocks"""
        # First merge similar title blocks
        merged_blocks = self.merge_similar_blocks(blocks, 'title')
        
        title_candidates = []
        
        for block in merged_blocks:
            if block['label'] == 'title':
                title_candidates.append({
                    'text': self.clean_text(block['text']),
                    'page': block['page'],
                    'font_size': block['font_size']
                })
        
        if title_candidates:
            # Get the largest font size title from first page
            first_page_titles = [t for t in title_candidates if t['page'] == 1]
            if first_page_titles:
                title = max(first_page_titles, key=lambda x: x['font_size'])
                return title['text']
            else:
                # Fallback to any title
                title = max(title_candidates, key=lambda x: x['font_size'])
                return title['text']
        
        return "Untitled Document"
        
    def build_outline(self, blocks):
        """Build document outline from classified sections"""
        # Extract title
        title = self.extract_title(blocks)

        # Check if document has uniform font sizes with more precise detection
        font_sizes = {}
        for block in blocks:
            if 'font_size' in block:
                size = round(block['font_size'], 1)
                font_sizes[size] = font_sizes.get(size, 0) + 1
        
        # Calculate if uniform by checking if one size dominates (>80% of blocks)
        total_blocks = sum(font_sizes.values())
        is_uniform_fonts = any(count/total_blocks > 0.8 for count in font_sizes.values())

        if is_uniform_fonts:
            print("Detected uniform font document - using styling for heading detection")
            for block in blocks:
                if block['label'] not in ['title']:
                    text = block.get('text', '').strip()
                    
                    # Strong indicators for headings
                    if block.get('is_bold', False):
                        if any(text.lower().startswith(word) for word in [
                            'chapter', 'section', 'part', 'introduction', 'conclusion',
                            'overview', 'summary'
                        ]):
                            block['label'] = 'H1'
                        elif text.isupper() and len(text.split()) <= 6:
                            block['label'] = 'H1'
                        elif re.match(r'^\d+\.\s', text):  # Numbered sections like "1. "
                            block['label'] = 'H2'
                        else:
                            block['label'] = 'H2'
                    
                    # Check for italic text with specific patterns
                    elif block.get('is_italic', False):
                        if len(text.split()) <= 8:  # Short italic phrases
                            block['label'] = 'H3'
                    
                    # Additional heading indicators
                    elif re.match(r'^[A-Z][^.!?]*(?:[.!?]|$)', text):  # Sentence case
                        if len(text.split()) <= 6:  # Short complete sentences
                            block['label'] = 'H3'
                    
                    # Check for semantic indicators
                    elif any(pattern in text.lower() for pattern in [
                        'what you need to know', 'key points', 'quick guide',
                        'tips for', 'guide to', 'how to', 'best places',
                        'things to do', 'where to'
                    ]):
                        block['label'] = 'H2'

        # Filter and sort heading blocks
        heading_blocks = [block for block in blocks if block['label'] in ['H1', 'H2', 'H3', 'H4']]
        heading_blocks.sort(key=lambda x: (x['page'], x['y_center']))

        # Build outline with duplicate prevention
        outline = []
        seen_texts = set()
        prev_level = None
        
        for block in heading_blocks:
            cleaned_text = self.clean_text(block['text'])
            if not cleaned_text or cleaned_text in seen_texts:
                continue
                
            # Ensure proper heading hierarchy
            if prev_level == 'H1' and block['label'] == 'H3':
                block['label'] = 'H2'  # Fix skipped levels
            
            outline.append({
                "level": block['label'],
                "text": cleaned_text,
                "page": block['page']
            })
            seen_texts.add(cleaned_text)
            prev_level = block['label']

        return {
            "title": title,
            "outline": outline
        }


    
    def create_document_structure(self, json_file_path):
        """Create complete document structure with title and outline"""
        with open(json_file_path, 'r', encoding='utf-8') as f:
            try:
                blocks = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                content = f.read()
                blocks = json.loads(content)
    
        # Ensure blocks is a list
        if not isinstance(blocks, list):
            raise ValueError("Input JSON should contain a list of blocks")
        
        # Get outline structure which includes both title and outline
        document_structure = self.build_outline(blocks)
        
        return document_structure
    
    def save_outline(self, json_file_path, output_file_path=None):
        """Process JSON file and save outline"""
        if output_file_path is None:
            output_file_path = json_file_path.replace('.json', '_outline.json')
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                # First try to load as regular JSON
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    # If that fails, try to load as string representation
                    f.seek(0)
                    content = f.read()
                    data = json.loads(content)
            
            # Ensure we have a list of dictionaries
            if not isinstance(data, list):
                raise ValueError("Expected a list of blocks in JSON file")
                
            document_structure = self.create_document_structure(json_file_path)
            
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(document_structure, f, indent=4, ensure_ascii=False)
            
            return output_file_path
        except Exception as e:
            print(f"Error processing {json_file_path}: {str(e)}")
            raise
    
    def print_outline(self, json_file_path):
        """Print outline in a readable format"""
        document_structure = self.create_document_structure(json_file_path)
        
        print(f"Title: {document_structure['title']}")
        print("\nOutline:")
        print("-" * 50)
        
        for item in document_structure['outline']:
            level = item['level']
            text = item['text']
            page = item['page']
            
            # Create indentation based on heading level
            indent = "  " * (int(level[1]) - 1) if level.startswith('H') else ""
            print(f"{indent}{level}: {text} (Page {page})")

# Example usage
if __name__ == "__main__":
    builder = OutlineBuilder()
    
    # Auto process all files in classified_jsons folder
    # Use absolute paths for folders
    json_folder = os.path.join(PROJECT_ROOT, "classified_jsons")
    output_folder = os.path.join(PROJECT_ROOT, "outlines")
    
    if os.path.exists(json_folder):
        os.makedirs(output_folder, exist_ok=True)
        
        json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
        
        if json_files:
            print(f"Processing {len(json_files)} classified JSON files...")
            
            for file in json_files:
                input_path = os.path.join(json_folder, file)
                output_path = os.path.join(output_folder, file.replace('.json', '_outline.json'))
                
                try:
                    builder.save_outline(input_path, output_path)
                    print(f"✓ Processed: {file}")
                except Exception as e:
                    print(f"✗ Error processing {file}: {e}")
            
            print(f"All outlines saved to: {output_folder}")
        else:
            print("No JSON files found in classified_jsons folder")
    else:
        print("Please ensure 'classified_jsons' folder exists with your JSON files")