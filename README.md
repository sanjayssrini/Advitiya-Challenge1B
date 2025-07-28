# PDF Section Classifier and Outline Generator

This project is a machine learning-based system that processes PDF documents to classify sections, generate outlines, and create task-focused summaries. It uses TensorFlow for classification and includes various processing components for PDF extraction and text analysis.

## Project Structure

```
├── requirements.txt        # Project dependencies
├── Dockerfile             # Docker configuration for containerization
├── src/                   # Source code directory
│   ├── run_full_pipeline.py           # Main entry point for the pipeline
│   ├── auto_processor.py              # Automated processing utilities
│   ├── extractor.py                   # PDF extraction functionality
│   ├── main_classifier.py             # TensorFlow-based section classifier
│   ├── outline_builder.py             # Document outline generation
│   ├── task_focused_summary_builder.py # Task-focused summary generation
│   └── train_ensemble.py              # Model training script
├── models/                # Trained model files
│   ├── ensemble_model.pkl         # Ensemble classifier model
│   ├── feature_columns.pkl       # Feature column configurations
│   ├── feature_scaler.pkl       # Feature scaling parameters
│   ├── label_encoder.pkl        # Label encoding information
│   └── section_classifier_model.h5    # TensorFlow model
├── data/                  # Training and validation data
│   ├── cleaned_dataset_final.csv
│   ├── cleaned_dataset.csv
│   └── dataset.csv
├── input/                 # Input directory for documents to process
│   └── challenge1b_input.json
├── output/                # Output directory for processed results
│   └── challenge1b_output.json
├── PDFs/                  # Directory for PDF documents
├── JSONs/                 # Extracted JSON data from PDFs
├── classified_jsons/      # Classification results
└── outlines/              # Generated document outlines
```

## Prerequisites

- Docker installed on your system
- Input JSON file following the required format
- PDF documents referenced in the input JSON

## Installation

### Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd tensorflow-classifier
   ```

2. Build the Docker image:
   ```bash
   docker build -t tensorflow-classifier .
   ```

3. Run the container:
   ```bash
   docker run -v ${PWD}/input:/app/input -v ${PWD}/output:/app/output tensorflow-classifier
   ```

### Manual Installation

1. Create a Python virtual environment (Python 3.9 recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your input JSON file in the following format and place it in the `input` directory:
   ```json
   {
       "documents": [
           {
               "filename": "document1.pdf",
               "content": "base64_encoded_pdf_content"
           }
       ]
   }
   ```

2. Run the pipeline:
   - Using Docker:
     ```bash
     docker run -v ${PWD}/input:/app/input -v ${PWD}/output:/app/output tensorflow-classifier
     ```
   - Without Docker:
     ```bash
     python src/run_full_pipeline.py
     ```

3. Check the `output` directory for results.

## Component Description

- **run_full_pipeline.py**: Main entry point that orchestrates the entire processing pipeline
- **auto_processor.py**: Handles automated processing of documents
- **extractor.py**: Extracts text and structure from PDF documents
- **main_classifier.py**: Implements the TensorFlow-based section classifier
- **outline_builder.py**: Generates document outlines based on classified sections
- **task_focused_summary_builder.py**: Creates focused summaries for specific tasks

## Model Information

The system uses several trained models:
- A TensorFlow-based section classifier (section_classifier_model.h5)
- An ensemble model for additional classification tasks (ensemble_model.pkl)
- Feature processing components (feature_scaler.pkl, label_encoder.pkl)
- FalconSAI text summarization model for generating concise summaries
- ALBERT-based paraphrase model (paraphrase-albert-small-v2) for text similarity and paraphrasing

### Pre-trained Models

#### FalconSAI Text Summarization
Located in `models/falconsai_text_summarization/`, this model is used for generating concise and coherent summaries of document sections. It includes:
- Model weights (model.safetensors)
- Configuration files (config.json, generation_config.json)
- Tokenizer files for text processing
- Special tokens mapping for summarization tasks

#### ALBERT Paraphrase Model
Located in `models/paraphrase-albert-small-v2/`, this is a fine-tuned ALBERT model specialized for paraphrasing and semantic similarity tasks. It includes:
- Model weights and architecture (model.safetensors)
- BERT configuration (config.json)
- Sentence transformers configuration
- Modules configuration for the model's components

These models work together to provide:
- High-quality text summarization
- Semantic understanding of document sections
- Improved paraphrasing capabilities
- Better text similarity matching

## Directory Purposes

- **input/**: Place input JSON files here
- **output/**: Contains processed results
- **PDFs/**: Stores extracted PDF documents
- **JSONs/**: Contains JSON representations of processed PDFs
- **classified_jsons/**: Stores classification results
- **outlines/**: Contains generated document outlines
- **models/**: Stores all trained models and related files
- **data/**: Contains training and validation datasets

## Error Handling

The pipeline includes error handling for common issues:
- Missing PDF files
- Invalid input JSON format
- PDF processing errors
- Classification errors

Check the console output for error messages and troubleshooting information.

## Development

To modify or extend the pipeline:
1. Make changes to relevant Python files in the `src` directory
2. Test changes using a small sample document
3. Rebuild Docker image if using containerized deployment

## License

[Add your license information here]
