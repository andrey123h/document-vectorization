import os
import re
import sys
import argparse
from pathlib import Path
from typing import List
from dotenv import load_dotenv
import psycopg2
from pypdf import PdfReader
from docx import Document
from google import genai

MAX_CHARS = 6000  # safely below Gemini's 2048 token limit. 1 token ≈ 4 chars, 6,000 characters ≈ 1,500 tokens

def setup_env():
    """Load and validate required environment variables."""
    load_dotenv()
    required_vars = [
        "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB",
        "POSTGRES_USER", "POSTGRES_PASSWORD", "GEMINI_API_KEY"
    ]

    for var in required_vars:
        if not os.getenv(var):
            raise ValueError(f"Missing required environment variable: {var}")

def extract_text_from_pdf(file_path: str) -> str:
    """Extract raw text from PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_text_from_docx(file_path: str) -> str:
    """Extract raw text from DOCX file."""
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text(file_path: str) -> str:
    """Extract text based on file extension."""
    file_ext = Path(file_path).suffix.lower()

    if file_ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif file_ext == ".docx":
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

def clean_chunk(text: str) -> str:
    """Normalize whitespace inside a single chunk."""
    text = re.sub(r'[ \t]+', ' ', text) # Replace multiple spaces/tabs with single space
    text = re.sub(r'\n+', '\n', text) # Replace multiple newlines with single newline
    return text.strip() # Trim leading or trailing whitespace

def enforce_max_length(text: str, max_chars: int = MAX_CHARS) -> List[str]:
    """
    Ensure text does not exceed embedding input limits.
    if text length exceeds max_chars, it will be split into smaller chunks.
    """
    return [
        text[i:i + max_chars]
        for i in range(0, len(text), max_chars)
    ]

def split_text_fixed(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into fixed-size chunks with overlap."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

def split_text_sentence(text: str) -> List[str]:
    """Split text by sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s for s in sentences if s.strip()]


def split_text_paragraph(text: str) -> List[str]:
    """Split text by paragraphs."""
    paragraphs = text.split('\n')
    return [p for p in paragraphs if p.strip()]


def chunk_text(text: str, strategy: str) -> List[str]:
    """Split text into chunks using the specified strategy."""
    if strategy == "fixed":
        return split_text_fixed(text)
    elif strategy == "sentence":
        return split_text_sentence(text)
    elif strategy == "paragraph":
        return split_text_paragraph(text)
    else:
        raise ValueError(f"Unsupported chunking strategy: '{strategy}'")


def generate_embedding(text: str) -> List[float]:
    """Generate embedding using Google Gemini API."""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values

def get_db_connection():
    """Establish PostgreSQL database connection."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD")
    )

def insert_to_db(conn, chunk_text: str, embedding: List[float], filename: str, strategy: str):
    """Insert data into PostgreSQL database."""
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO document_embeddings (chunk_text, embedding, filename, strategy_split)
        VALUES (%s, %s, %s, %s)
        """,
        (chunk_text, embedding, filename, strategy)
    )
    cursor.close()

def process_document(file_path: str, strategy: str):
    """document vectorization pipeline."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    filename = Path(file_path).name
    print(f"Processing: {filename}")

    # Extract raw text
    text = extract_text(file_path)

    # Chunk text
    chunks = chunk_text(text, strategy)

    # Clean each chunk
    chunks = [clean_chunk(c) for c in chunks]

    print(f"Created {len(chunks)} chunks using {strategy} strategy")

    # Connect to DB
    conn = get_db_connection()

    try:
        chunk_counter = 0 # counter for processed chunks. used only to track progress

        for chunk in chunks:

            safe_chunks = enforce_max_length(chunk) # ensure chunk is within embedding limits
            for safe_chunk in safe_chunks: # process each chunk
                embedding = generate_embedding(safe_chunk)
                insert_to_db(conn, safe_chunk, embedding, filename, strategy)

                chunk_counter += 1
                print(f"Processed chunk {chunk_counter}")

        conn.commit()
        print(f"✅ Successfully stored {chunk_counter} embeddings in database")

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def main():
    """Script entry point."""
    parser = argparse.ArgumentParser(description="Document vectorization pipeline")
    parser.add_argument("file_path", help="Path to PDF or DOCX file")
    parser.add_argument(
        "--strategy",
        choices=["fixed", "sentence", "paragraph"],
        default="fixed",
        help="Chunking strategy (default: fixed)"
    )

    args = parser.parse_args()

    try:
        setup_env()
        process_document(args.file_path, args.strategy)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
