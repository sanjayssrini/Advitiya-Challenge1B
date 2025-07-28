# ✅ main_classifier.py (Merged: Original + Ensemble + Outline + Stats + Merging)
import os
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
from sklearn.preprocessing import LabelEncoder, StandardScaler
from extractor import estimate_better_label
from outline_builder import OutlineBuilder
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

class PDFSectionClassifier:
    def __init__(self):
        models_dir = os.path.join(PROJECT_ROOT, "models")
        self.label_encoder = joblib.load(os.path.join(models_dir, "label_encoder.pkl"))
        self.feature_columns = joblib.load(os.path.join(models_dir, "feature_columns.pkl"))
        self.scaler = joblib.load(os.path.join(models_dir, "feature_scaler.pkl"))
        self.outline_builder = OutlineBuilder()

        self.model = joblib.load(os.path.join(models_dir, "ensemble_model.pkl"))

        print("✓ Ensemble model initialized")

    def create_features(self, df):
        font_cols = ['Font_alt_family_name', 'Font_embedded', 'Font_encoding', 
                     'Font_family_name', 'Font_font_type', 'Font_italic', 
                     'Font_monospaced', 'Font_name', 'Font_subset', 'Font_weight']

        attr_cols = ['attributes_LineHeight', 'attributes_SpaceAfter', 'attributes_TextAlign']

        for col in font_cols:
            if col not in df.columns and 'Font' in df.columns:
                col_name = col.replace('Font_', '')
                df[col] = df['Font'].apply(lambda x: x.get(col_name, 0) if isinstance(x, dict) else 0)

        for col in attr_cols:
            if col not in df.columns and 'attributes' in df.columns:
                col_name = col.replace('attributes_', '')
                df[col] = df['attributes'].apply(lambda x: x.get(col_name, 0) if isinstance(x, dict) else 0)

        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()

        if 'bbox' in df.columns:
            df['bbox_width'] = df['bbox'].apply(lambda x: x[2] - x[0])
            df['bbox_height'] = df['bbox'].apply(lambda x: x[3] - x[1])
        else:
            df['bbox_width'] = 0
            df['bbox_height'] = 0

        df['bbox_area'] = df['bbox_width'] * df['bbox_height']

        for col in ['Font_alt_family_name', 'Font_family_name', 'Font_name']:
            if col in df.columns:
                df[col] = df[col].astype('category').cat.codes
        # Derived features
        df['avg_word_length'] = df['text_length'] / df['word_count']
        df['avg_word_length'] = df['avg_word_length'].replace([float('inf'), -float('inf')], 0).fillna(0)
        df['starts_with_number'] = df['text'].str.match(r'^\\d').astype(int)
        df['has_punctuation'] = df['text'].str.contains(r'[.,!?]').astype(int)

        # Font z-score
        if 'font_size' in df.columns:
            mean_font = df['font_size'].mean()
            std_font = df['font_size'].std() or 1  # avoid division by zero
            df['font_size_zscore'] = (df['font_size'] - mean_font) / std_font
        else:
            df['font_size_zscore'] = 0

        # Position features
        if 'y_center' in df.columns:
            df['normalized_y_pos'] = df['y_center'] / df['y_center'].max()
            df['is_top_third'] = (df['normalized_y_pos'] <= 0.33).astype(int)
        else:
            df['normalized_y_pos'] = 0
            df['is_top_third'] = 0

        df.fillna(0, inplace=True)
        return df

    def prepare_input(self, df):
        """Prepare input features for classification."""
        # Ensure all categorical columns are encoded
        for col in self.feature_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))

        # Scale features
        X_scaled = self.scaler.transform(df[self.feature_columns])
        return X_scaled

    def predict_labels(self, json_file_path):
        """Predict labels for blocks in a JSON file."""
        with open(json_file_path, 'r', encoding='utf-8') as f:
            blocks = json.load(f)

        if not blocks:
            return []

        df = pd.DataFrame(blocks)
        df = self.create_features(df)
        X_scaled = self.prepare_input(df)

        predicted = ["Other"] * len(blocks)
        confidences = [0.0] * len(blocks)

        try:
            # Ensure the model is trained
            if not hasattr(self.model, 'estimators_'):
                raise RuntimeError("Ensemble model not trained. Please train the model first.")

            probs = self.model.predict_proba(X_scaled)
            predicted_classes = self.model.predict(X_scaled)
            predicted = self.label_encoder.inverse_transform(predicted_classes)
            confidences = np.max(probs, axis=1)
        except Exception as e:
            print(f"Warning: {e}. Using fallback heuristics only.")

        font_sizes = [b.get("font_size", 0) for b in blocks]
        largest_font = max(font_sizes)
        avg_font_size = np.mean(font_sizes)

        for i, block in enumerate(blocks):
            if confidences[i] < 0.6:
                block['label'] = estimate_better_label(block, largest_font, avg_font_size)
            else:
                block['label'] = predicted[i]
            block['confidence'] = float(confidences[i])

        return blocks

    def classify_pdf(self, json_file_path, output_file_path=None, create_outline=False):
        if output_file_path is None:
            output_file_path = json_file_path.replace(".json", "_classified.json")

        print(f"Classifying: {json_file_path}")
        blocks = self.predict_labels(json_file_path)

        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(blocks, f, indent=2, ensure_ascii=False)

        if create_outline:
            outline_path = output_file_path.replace(".json", "_outline.json")
            self.outline_builder.save_outline(output_file_path, outline_path)
            self.outline_builder.print_outline(output_file_path)

        return output_file_path

    def merge_similar_sections(self, blocks):
        model_path = "models/paraphrase-albert-small-v2"
        if not os.path.exists(model_path):
            model = SentenceTransformer('paraphrase-albert-small-v2')
            model.save(path=model_path)
        else:
            model = SentenceTransformer(model_path)
        merged_blocks = []
        heading_levels = ['title', 'H1', 'H2', 'H3', 'H4']

        for level in heading_levels:
            level_blocks = [b for b in blocks if b['label'] == level]
            level_blocks.sort(key=lambda x: (x['page'], x['y_center']))

            i = 0
            while i < len(level_blocks):
                current = level_blocks[i].copy()
                j = i + 1
                while j < len(level_blocks):
                    next_block = level_blocks[j]
                    if current['page'] == next_block['page']:
                        sim = cosine_similarity(
                            model.encode([current['text']]),
                            model.encode([next_block['text']])
                        )[0][0]
                        if sim > 0.8:
                            current['text'] += ' ' + next_block['text']
                            j += 1
                        else:
                            break
                    else:
                        break
                merged_blocks.append(current)
                i = j

        others = [b for b in blocks if b['label'] not in heading_levels]
        return merged_blocks + others

    def batch_classify(self, input_folder="JSONs", output_folder="classified_jsons"):
        """Batch classify all JSON files in the input folder."""
        input_path = os.path.join(PROJECT_ROOT, input_folder)
        output_path = os.path.join(PROJECT_ROOT, output_folder)
        os.makedirs(output_path, exist_ok=True)
        json_files = [f for f in os.listdir(input_path) if f.endswith(".json")]

        if not json_files:
            print("No JSON files found.")
            return

        print(f"Found {len(json_files)} JSON files to classify")

        for i, file in enumerate(json_files):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file.replace(".json", "_classified.json"))
            try:
                self.classify_pdf(input_path, output_path, create_outline=False)
                print(f"[{i+1}/{len(json_files)}] ✓ {file}")
            except Exception as e:
                print(f"[{i+1}/{len(json_files)}] ✗ {file}: {e}")

        print(f"\nBatch classification complete! Results saved in: {output_folder}")

    def extract_features(self, text_block):
        """Enhanced feature extraction to handle uniform font sizes"""
        features = {
            # Existing features
            'font_size': text_block.get('font_size', 0),
            'is_bold': text_block.get('is_bold', False),
            
            # New features for uniform font size cases
            'is_all_caps': text_block.get('text', '').isupper(),
            'starts_with_number': bool(re.match(r'^\d+\.?\s', text_block.get('text', ''))),
            'indentation': text_block.get('x_coord', 0),
            'text_length': len(text_block.get('text', '')),
            'line_spacing_before': text_block.get('space_before', 0),
            'has_following_content': bool(text_block.get('next_block_distance', 0) < 20),
            'position_on_page': text_block.get('y_coord', 0) / text_block.get('page_height', 1),
        }
        
        # Language pattern features
        text = text_block.get('text', '').lower()
        features.update({
            'looks_like_title': any(word in text for word in ['chapter', 'section', 'part']),
            'ends_with_colon': text.strip().endswith(':'),
            'contains_heading_keywords': any(word in text for word in [
                'introduction', 'conclusion', 'summary', 'methodology', 
                'results', 'discussion', 'references'
            ])
        })
        
        return features

if __name__ == "__main__":
    print("PDF Section Classifier - Merged Version")
    clf = PDFSectionClassifier()
    clf.batch_classify()