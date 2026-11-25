
# DocuSort AI

## Intelligent Document Classification and Organization System

DocuSort AI is a Python-based document management application
that uses Optical Character Recognition (OCR) and Machine Learning
to extract, classify, organize and search documents.

## Features

- Upload PDF and image documents
- Extract text from PDFs
- OCR text extraction from images
- Machine Learning document classification
- Automatic category-based organization
- Exact duplicate file detection
- Search documents by filename and extracted content
- Dashboard with document statistics

## Technologies

- Python
- Streamlit
- Scikit-learn
- TF-IDF
- Multinomial Naive Bayes
- PyMuPDF
- Tesseract OCR
- SQLite

## Installation

Clone the repository:

    git clone YOUR_GITHUB_REPOSITORY_URL

Open the project folder:

    cd DocuSort_AI

Create a virtual environment:

    python -m venv .venv

Activate it on Windows:

    .venv\Scripts\activate

Install dependencies:

    pip install -r requirements.txt

Train the model:

    python train_model.py

Run the application:

    streamlit run app.py

## Document Categories

- Education
- Finance
- Career
- Medical
- Utility
- Others

## Project Status

College Mini Project - Prototype

## Note

Use sample documents for testing. The prototype is not designed
for storing sensitive personal documents.
