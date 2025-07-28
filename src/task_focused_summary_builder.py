# task_focused_summary_builder.py
import os
from pathlib import Path
import json
import datetime
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from main_classifier import PDFSectionClassifier
from outline_builder import OutlineBuilder

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Define folder paths
PDF_FOLDER = os.path.join(PROJECT_ROOT, "PDFs")
JSON_FOLDER = os.path.join(PROJECT_ROOT, "JSONs")
CLASSIFIED_FOLDER = os.path.join(PROJECT_ROOT, "classified_jsons")

class TaskFocusedSummaryBuilder:
    def __init__(self):
        # Initialize summarization model (lightweight)
        model_path = os.path.join(PROJECT_ROOT, "models", "falconsai_text_summarization")
        if not os.path.exists(model_path):
            print("⏬ Downloading Falconsai model...")
            model_id = "Falconsai/text_summarization"
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)

        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

        # Load sentence embedding model
        self.embedder = SentenceTransformer(os.path.join(PROJECT_ROOT, "models", "paraphrase-albert-small-v2"), local_files_only=True)

        self.outline_builder = OutlineBuilder()

        self.relevant_categories = {
            'cities': 1,
            'activities': 2,
            'cuisine': 3,
            'tips': 4,
            'entertainment': 5
        }

    def process_documents(self, input_data):
        job_text = input_data["job_to_be_done"]["task"]
        job_embedding = self.embedder.encode(job_text, convert_to_tensor=True)
        all_sections = []
        section_contents = {}

        for doc in input_data["documents"]:
            filename = doc["filename"]
            classified_path = os.path.join(CLASSIFIED_FOLDER, filename.replace(".pdf", "_classified.json"))
            outline = self.outline_builder.create_document_structure(classified_path)

            with open(classified_path, 'r', encoding='utf-8') as f:
                contents = json.load(f)

            for section in outline['outline']:
                section_text = section['text']
                section_embedding = self.embedder.encode(section_text, convert_to_tensor=True)
                relevance_score = float(util.cos_sim(job_embedding, section_embedding))

                category = self._determine_category(section_text)
                importance = self.relevant_categories.get(category, 99)

                section_data = {
                    "document": filename,
                    "section_title": section_text,
                    "importance_rank": importance,
                    "page_number": section['page'],
                    "relevance_score": relevance_score,
                    "category": category
                }

                all_sections.append(section_data)
                section_contents[f"{filename}_{section_text}"] = self._get_section_content(contents, section)

        all_sections.sort(key=lambda x: (x['importance_rank'], -x['relevance_score']))
        top_sections = all_sections[:5]
        subsection_analysis = self._generate_subsection_analysis(top_sections, section_contents)

        return {
            "metadata": {
                "input_documents": [doc["filename"] for doc in input_data["documents"]],
                "persona": input_data["persona"]["role"],
                "job_to_be_done": job_text,
                "processing_timestamp": datetime.datetime.now().isoformat()
            },
            "extracted_sections": [{
                "document": section["document"],
                "section_title": section["section_title"],
                "importance_rank": idx + 1,
                "page_number": section["page_number"]
            } for idx, section in enumerate(top_sections)],
            "subsection_analysis": subsection_analysis
        }

    def _determine_category(self, text):
        text_lower = text.lower()
        if any(word in text_lower for word in ['city', 'cities', 'town', 'region']):
            return 'cities'
        elif any(word in text_lower for word in ['beach', 'sport', 'adventure', 'activity']):
            return 'activities'
        elif any(word in text_lower for word in ['food', 'cuisine', 'restaurant', 'wine']):
            return 'cuisine'
        elif any(word in text_lower for word in ['tip', 'guide', 'advice', 'pack']):
            return 'tips'
        elif any(word in text_lower for word in ['night', 'entertainment', 'bar', 'club']):
            return 'entertainment'
        return 'other'

    def _get_section_content(self, contents, section):
        relevant_blocks = []
        page = section['page']
        for block in contents:
            if (block['page'] == page and 
                block.get('label') == 'Body' and 
                len(block.get('text', '').strip()) > 0):
                relevant_blocks.append(block['text'])
        return ' '.join(relevant_blocks)

    def _generate_subsection_analysis(self, sections, section_contents):
        analysis = []
        for section in sections:
            content = section_contents.get(f"{section['document']}_{section['section_title']}", '')
            if content:
                summary = self.summarizer(
                    content, 
                    max_length=120,
                    min_length=50,
                    do_sample=False
                )[0]['summary_text']

                analysis.append({
                    "document": section["document"],
                    "refined_text": summary,
                    "page_number": section["page_number"]
                })
        return analysis

# Remaining global helper functions (unchanged)

def load_input_json(path="input.json"):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_local_embedder(model_path="models/paraphrase-albert-small-v2"):
    if not os.path.exists(model_path):
        model = SentenceTransformer('paraphrase-albert-small-v2')
        model.save(model_path)
    else:
        model = SentenceTransformer(model_path)
    return model

embedder = get_local_embedder()

def extract_job_embedding(job_text):
    return embedder.encode(job_text, convert_to_tensor=True)

def get_relevant_sections(blocks, job_embedding):
    headings = [b for b in blocks if b['label'] in ['title', 'H1', 'H2', 'H3', 'H4'] and len(b['text'].strip()) > 3]
    heading_embeddings = embedder.encode([b['text'] for b in headings], convert_to_tensor=True)
    similarities = util.cos_sim(heading_embeddings, job_embedding).squeeze().tolist()
    for i, sim in enumerate(similarities):
        headings[i]['similarity'] = sim
    headings.sort(key=lambda b: b['similarity'], reverse=True)
    return headings[:5]

def get_refined_explanation(blocks, selected_blocks):
    refined = []
    for b in selected_blocks:
        doc = b['document']
        page = b['page']
        candidates = [x for x in blocks if x['page'] == page and x['label'] == 'Body']
        candidates.sort(key=lambda x: abs(x['y_center'] - b['y_center']))
        nearby = candidates[:3]
        for n in nearby:
            refined.append({
                'document': doc,
                'refined_text': n['text'],
                'page_number': page
            })
    return refined

def build_output(input_data):
    output = {
        "metadata": {
            "input_documents": [doc["filename"] for doc in input_data["documents"]],
            "persona": input_data["persona"]["role"],
            "job_to_be_done": input_data["job_to_be_done"]["task"],
            "processing_timestamp": datetime.datetime.now().isoformat()
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }

    job_embedding = extract_job_embedding(input_data["job_to_be_done"]["task"])
    importance_counter = 1
    classifier = PDFSectionClassifier()
    for doc in input_data["documents"]:
        filename = doc['filename']
        title = doc['title']
        json_path = os.path.join(JSON_FOLDER, filename.replace(".pdf", ".json"))
        classified_path = os.path.join(CLASSIFIED_FOLDER, filename.replace(".pdf", "_classified.json"))

        if not os.path.exists(classified_path):
            print(f"Classifying {filename}...")
            classifier.classify_pdf(json_path, classified_path)

        with open(classified_path, 'r', encoding='utf-8') as f:
            blocks = json.load(f)
            for b in blocks:
                b['document'] = filename

        top_sections = get_relevant_sections(blocks, job_embedding)

        for sec in top_sections:
            output["extracted_sections"].append({
                "document": filename,
                "section_title": sec['text'],
                "importance_rank": importance_counter,
                "page_number": sec['page']
            })
            importance_counter += 1

        refined = get_refined_explanation(blocks, top_sections)
        output["subsection_analysis"].extend(refined)

    return output

def main():
    input_data = load_input_json("challenge1b_input.json")
    output_data = build_output(input_data)
    with open("output.json", "w", encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print("✓ Task-focused summary written to output.json")

if __name__ == "__main__":
    main()
