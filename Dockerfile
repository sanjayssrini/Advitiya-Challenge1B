# Use Python 3.9 as base image (compatible with the project's .pyc files)
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for PyMuPDF
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p PDFs JSONs classified_jsons outlines output

# Set environment variables
ENV PYTHONPATH=/app

# Command to run the pipeline
CMD ["python", "src/run_full_pipeline.py"]
