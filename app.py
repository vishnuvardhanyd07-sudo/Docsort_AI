
import os
import re
import hashlib
import sqlite3
from datetime import datetime

import streamlit as st
import pymupdf as fitz
import joblib
import pytesseract

from PIL import Image


# -----------------------------
# APPLICATION CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="DocuSort AI",
    page_icon="📁",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STORAGE_DIR = os.path.join(BASE_DIR, "storage")
DATABASE_PATH = os.path.join(BASE_DIR, "database", "documents.db")
MODEL_PATH = os.path.join(BASE_DIR, "machine_learning", "classifier.pkl")

os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

CATEGORIES = [
    "Education",
    "Finance",
    "Career",
    "Medical",
    "Utility",
    "Others"
]

# If Tesseract is not in Windows PATH, configure it here.
WINDOWS_TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if os.path.exists(WINDOWS_TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = (
        WINDOWS_TESSERACT_PATH
    )


# -----------------------------
# DATABASE
# -----------------------------

def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def create_database():
    connection = get_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            stored_path TEXT,
            category TEXT,
            extracted_text TEXT,
            file_hash TEXT UNIQUE,
            upload_date TEXT
        )
    """)

    connection.commit()
    connection.close()


create_database()


# -----------------------------
# LOAD MACHINE LEARNING MODEL
# -----------------------------

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)


model = load_model()


# -----------------------------
# TEXT EXTRACTION
# -----------------------------

def extract_pdf_text(file_path):
    text = ""

    pdf = fitz.open(file_path)

    for page in pdf:
        page_text = page.get_text()
        text += page_text + "\n"

        # OCR fallback for scanned pages
        if len(page_text.strip()) < 20:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            image = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )
            text += pytesseract.image_to_string(image) + "\n"

    pdf.close()
    return text


def extract_image_text(file_path):
    image = Image.open(file_path)
    return pytesseract.image_to_string(image)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# -----------------------------
# MACHINE LEARNING PREDICTION
# -----------------------------

def classify_document(text):
    if model is None:
        return "Others", 0.0

    cleaned = clean_text(text)

    if not cleaned:
        return "Others", 0.0

    prediction = model.predict([cleaned])[0]

    probabilities = model.predict_proba([cleaned])[0]
    confidence = float(max(probabilities))

    return prediction, confidence


# -----------------------------
# DUPLICATE CHECKING
# -----------------------------

def calculate_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def is_duplicate(file_hash):
    connection = get_connection()

    result = connection.execute(
        "SELECT id FROM documents WHERE file_hash = ?",
        (file_hash,)
    ).fetchone()

    connection.close()

    return result is not None


# -----------------------------
# SAVE DOCUMENT
# -----------------------------

def save_document(
    original_filename,
    temporary_path,
    category,
    extracted_text,
    file_hash
):
    safe_name = re.sub(
        r"[^a-zA-Z0-9._-]",
        "_",
        original_filename
    )

    category_folder = os.path.join(
        STORAGE_DIR,
        category
    )

    os.makedirs(category_folder, exist_ok=True)

    stored_filename = (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + "_"
        + safe_name
    )

    stored_path = os.path.join(
        category_folder,
        stored_filename
    )

    with open(temporary_path, "rb") as source:
        with open(stored_path, "wb") as destination:
            destination.write(source.read())

    connection = get_connection()

    connection.execute("""
        INSERT INTO documents
        (filename, stored_path, category,
         extracted_text, file_hash, upload_date)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        original_filename,
        stored_path,
        category,
        extracted_text,
        file_hash,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    connection.commit()
    connection.close()


# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------

st.sidebar.title("📁 DocuSort AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Upload Document",
        "My Documents",
        "Search Documents"
    ]
)


# -----------------------------
# DASHBOARD
# -----------------------------

if page == "Dashboard":

    st.title("📊 DocuSort AI Dashboard")

    connection = get_connection()

    total = connection.execute(
        "SELECT COUNT(*) FROM documents"
    ).fetchone()[0]

    rows = connection.execute("""
        SELECT category, COUNT(*)
        FROM documents
        GROUP BY category
    """).fetchall()

    connection.close()

    col1, col2 = st.columns(2)

    col1.metric("Total Documents", total)
    col2.metric("Categories Used", len(rows))

    st.subheader("Documents by Category")

    if rows:
        chart_data = {
            category: count
            for category, count in rows
        }

        st.bar_chart(chart_data)
    else:
        st.info("Upload your first document to see statistics.")

    st.write(
        "DocuSort AI uses OCR and Machine Learning "
        "to classify and organize documents."
    )


# -----------------------------
# UPLOAD DOCUMENT
# -----------------------------

elif page == "Upload Document":

    st.title("📤 Upload Your Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF, PNG or JPG document",
        type=["pdf", "png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        st.write("Selected file:", uploaded_file.name)

        if st.button("Process Document"):

            temporary_path = os.path.join(
                STORAGE_DIR,
                "temp_" + re.sub(
                    r"[^a-zA-Z0-9._-]",
                    "_",
                    uploaded_file.name
                )
            )

            with open(temporary_path, "wb") as file:
                file.write(uploaded_file.getbuffer())

            try:
                file_hash = calculate_hash(temporary_path)

                if is_duplicate(file_hash):
                    st.warning(
                        "This exact file has already been uploaded."
                    )

                else:
                    extension = uploaded_file.name.lower()

                    if extension.endswith(".pdf"):
                        extracted_text = extract_pdf_text(
                            temporary_path
                        )
                    else:
                        extracted_text = extract_image_text(
                            temporary_path
                        )

                    if not extracted_text.strip():
                        st.error(
                            "No text could be extracted. "
                            "Check the document or OCR installation."
                        )

                    else:
                        category, confidence = classify_document(
                            extracted_text
                        )

                        st.success("Document processed!")

                        st.write("Predicted category:", category)

                        st.write(
                            "Model confidence:",
                            f"{confidence * 100:.2f}%"
                        )

                        st.subheader("Extracted Text")
                        st.text_area(
                            "Document content",
                            extracted_text,
                            height=180
                        )

                        selected_category = st.selectbox(
                            "Confirm or change the category",
                            CATEGORIES,
                            index=CATEGORIES.index(category)
                            if category in CATEGORIES else 5
                        )

                        if st.button("Save Document"):
                            save_document(
                                uploaded_file.name,
                                temporary_path,
                                selected_category,
                                extracted_text,
                                file_hash
                            )

                            st.success(
                                "Document saved successfully!"
                            )

            except Exception as error:
                st.error(f"Processing failed: {error}")

            finally:
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)


# -----------------------------
# MY DOCUMENTS
# -----------------------------

elif page == "My Documents":

    st.title("📚 My Documents")

    connection = get_connection()

    documents = connection.execute("""
        SELECT id, filename, category, upload_date
        FROM documents
        ORDER BY id DESC
    """).fetchall()

    connection.close()

    if documents:

        for document in documents:

            doc_id, filename, category, date = document

            with st.expander(filename):

                st.write("Document ID:", doc_id)
                st.write("Category:", category)
                st.write("Uploaded:", date)

    else:
        st.info("No documents uploaded yet.")


# -----------------------------
# SEARCH DOCUMENTS
# -----------------------------

elif page == "Search Documents":

    st.title("🔍 Search Documents")

    keyword = st.text_input(
        "Enter filename or a word from the document"
    )

    category_filter = st.selectbox(
        "Filter by category",
        ["All"] + CATEGORIES
    )

    if st.button("Search"):

        connection = get_connection()

        query = """
            SELECT id, filename, category, upload_date
            FROM documents
            WHERE (
                filename LIKE ?
                OR extracted_text LIKE ?
            )
        """

        params = [
            f"%{keyword}%",
            f"%{keyword}%"
        ]

        if category_filter != "All":
            query += " AND category = ?"
            params.append(category_filter)

        results = connection.execute(
            query,
            params
        ).fetchall()

        connection.close()

        if results:

            st.success(f"Found {len(results)} document(s).")

            for result in results:
                st.write(
                    f"**{result[1]}** | "
                    f"{result[2]} | {result[3]}"
                )

        else:
            st.warning("No matching documents found.")


# -----------------------------
# FOOTER
# -----------------------------

st.sidebar.divider()
st.sidebar.caption("DocuSort AI | Mini Project")
