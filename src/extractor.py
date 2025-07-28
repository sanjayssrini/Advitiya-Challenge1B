import fitz  # PyMuPDF
import os
from pathlib import Path
import json
import re
from statistics import mean

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

def parse_font_name(font_name):
    """Parse font name to extract font properties."""
    is_bold = bool(re.search(r"(bold|black)", font_name.lower()))
    is_italic = "italic" in font_name.lower()
    weight = 700 if is_bold else 400
    return {
        "alt_family_name": font_name.split("-")[0],
        "embedded": False,
        "encoding": "WinAnsiEncoding",
        "family_name": font_name.split("-")[0],
        "font_type": "TrueType",
        "italic": is_italic,
        "monospaced": False,
        "name": font_name,
        "subset": False,
        "weight": weight
    }

def estimate_better_label(block, largest_font, avg_font_size):
    """Estimate label using logic based on font size, weight, and position."""
    size = block["font_size"]
    weight = block["Font"]["weight"]
    centered = abs(block["x_center"] - 300) < 100
    is_bold = weight >= 700

    if size >= largest_font - 0.5 and centered:
        return "title"
    elif size >= avg_font_size + 4 and is_bold:
        return "H1"
    elif size >= avg_font_size + 3:
        return "H2"
    elif size >= avg_font_size + 2:
        return "H3"
    elif size >= avg_font_size + 1:
        return "H4"
    elif size >= avg_font_size - 1:
        return "Body"
    else:
        return "Other"

def extract_text_blocks(pdf_path):
    """Extract text blocks from PDF with font information and positioning."""
    doc = fitz.open(pdf_path)
    raw_blocks = []

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block['type'] != 0:
                continue
            for line in block["lines"]:
                line_text = " ".join([span["text"].strip() for span in line["spans"]]).strip()
                if not line_text:
                    continue

                span = line["spans"][0]
                text_size = span["size"]
                font_info = parse_font_name(span["font"])
                bbox = span["bbox"]

                block_data = {
                    "page": page_num,
                    "text": line_text,
                    "font_size": round(text_size, 2),
                    "font_name": span["font"],
                    "bbox": bbox,
                    "x_center": (bbox[0] + bbox[2]) / 2,
                    "y_center": (bbox[1] + bbox[3]) / 2,
                    "Lang": "en",
                    "Font": font_info,
                    "attributes": {
                        "LineHeight": round(text_size * 1.2, 1),
                        "SpaceAfter": 0.0,
                        "TextAlign": "Center" if abs(((bbox[0] + bbox[2]) / 2) - 300) < 100 else "Left"
                    }
                }
                raw_blocks.append(block_data)

    # Analyze font sizes to define threshold
    font_sizes = [b["font_size"] for b in raw_blocks]
    if not font_sizes:
        return []

    avg_font_size = mean(font_sizes)

    # Sort by page then y_center
    raw_blocks.sort(key=lambda b: (b["page"], b["y_center"]))

    # Merge only title-like blocks
    merged_blocks = []
    i = 0
    while i < len(raw_blocks):
        curr = raw_blocks[i]
        j = i + 1

        # Only merge if current block is "title-like"
        is_title_like = (
            curr["font_size"] >= avg_font_size + 4 and
            curr["Font"]["weight"] >= 700 and
            abs(curr["x_center"] - 300) < 100
        )

        while is_title_like and j < len(raw_blocks):
            nxt = raw_blocks[j]

            # Next block should also be title-like
            nxt_title_like = (
                nxt["font_size"] >= avg_font_size + 4 and
                nxt["Font"]["weight"] >= 700 and
                abs(nxt["x_center"] - 300) < 100
            )

            vertical_gap = nxt["bbox"][1] - curr["bbox"][3]
            same_font = curr["font_name"] == nxt["font_name"]
            size_diff = abs(curr["font_size"] - nxt["font_size"]) <= 1

            if (
                nxt["page"] == curr["page"] and
                0 <= vertical_gap <= 40 and
                nxt_title_like and
                same_font and
                size_diff
            ):
                # Merge
                curr["text"] += " " + nxt["text"]
                curr["bbox"] = [
                    min(curr["bbox"][0], nxt["bbox"][0]),
                    min(curr["bbox"][1], nxt["bbox"][1]),
                    max(curr["bbox"][2], nxt["bbox"][2]),
                    max(curr["bbox"][3], nxt["bbox"][3]),
                ]
                curr["x_center"] = (curr["bbox"][0] + curr["bbox"][2]) / 2
                curr["y_center"] = (curr["bbox"][1] + curr["bbox"][3]) / 2
                j += 1
            else:
                break

        merged_blocks.append(curr)
        i = j

    doc.close()
    return merged_blocks

def process_pdf(pdf_path, output_path):
    """Process a single PDF and save extracted blocks to JSON."""
    print(f"Processing: {pdf_path}")
    blocks = extract_text_blocks(pdf_path)
    if not blocks:
        print("[!] No blocks extracted.")
        return

    font_sizes = [b["font_size"] for b in blocks if b["text"]]
    largest_font = max(font_sizes)
    avg_font_size = mean(font_sizes)

    for b in blocks:
        b["level"] = estimate_better_label(b, largest_font, avg_font_size)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(blocks, f, indent=2)
    print(f"[✔] Saved: {os.path.basename(output_path)}")

def process_all_pdfs(pdf_folder="PDFs", json_folder="JSONs"):
    """Process all PDFs in a folder and save JSON files."""
    # Convert to absolute paths using PROJECT_ROOT
    pdf_folder = os.path.join(PROJECT_ROOT, pdf_folder)
    json_folder = os.path.join(PROJECT_ROOT, json_folder)
    
    os.makedirs(json_folder, exist_ok=True)
    
    # Check if PDF folder exists
    if not os.path.exists(pdf_folder):
        print(f"Error: PDF folder not found at {pdf_folder}")
        return
        
    for file in os.listdir(pdf_folder):
        print("[checking]", file)
        if file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, file)
            json_path = os.path.join(json_folder, file.replace(".pdf", ".json"))
            process_pdf(pdf_path, json_path)

def flatten_dict(d, parent_key='', sep='_'):
    """Flatten nested dictionary for CSV export."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep))
        else:
            items[new_key] = v
    return items

def load_all_blocks(json_folder):
    """Load all blocks from JSON files in a folder."""
    json_folder = os.path.join(PROJECT_ROOT, json_folder)
    blocks = []
    if not os.path.exists(json_folder):
        print(f"Error: JSON folder not found at {json_folder}")
        return blocks
        
    for file in os.listdir(json_folder):
        if file.endswith(".json"):
            with open(os.path.join(json_folder, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                blocks.extend(data)
    return blocks

def get_all_fieldnames(blocks):
    """Get all unique field names from blocks for CSV header."""
    fieldnames = set()
    for b in blocks:
        flat = flatten_dict(b)
        fieldnames.update(flat.keys())
    fieldnames.add("label")
    return sorted(fieldnames)


