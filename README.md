# document-vectorization

A single-script document vectorization pipeline that extracts text from PDF/DOCX files, chunks content using configurable strategies, applies post-chunk normalization and token safety guards, generates Gemini embeddings, and persists results in PostgreSQL DB.

## Table of Contents

- [Running the Project Locally](#running-the-project-locally)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#1-clone-the-repository)
  - [Install Dependencies](#2-install-dependencies)
  - [Configure Environment Variables](#3-configure-environment-variables)
  - [Create the Database Table](#4-create-the-database-table)
  - [Run the Script](#run-the-script)
- [Engineering Decisions](#engineering-decisions)
  - [Embedding Storage and Database Selection](#embedding-storage-and-database-selection)
  - [Text Normalization](#text-normalization)
  - [Token Safety and Input Size Limits](#token-safety-and-input-size-limits)
- [Example Output](#example-output)

## Running the Project Locally

### Prerequisites:

- Python 3.9+ (used Python 3.14)
- PostgreSQL 13+ (used PostgreSQL 16.10)
- Git

### 1. Clone the Repository

`git clone https://github.com/andrey123h/document-vectorization.git`

### 2. Install Dependencies
`pip install -r requirements.txt`

### 3. Configure Environment Variables 

Create a .env file in the project root.

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=document_vectors
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password_here

GEMINI_API_KEY=your_gemini_api_key_here

### 4. Create the Database Table
execute the following SQL:
```sql
CREATE TABLE document_embeddings (
    id SERIAL PRIMARY KEY,
    chunk_text TEXT NOT NULL,
    embedding DOUBLE PRECISION[] NOT NULL,
    filename TEXT NOT NULL,
    strategy_split TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Run the Script

The script accepts a PDF or DOCX file and a chunking strategy.
Available chunking strategies:

- fixed: fixed-size chunks with overlap (default)
- sentence: sentence-based splitting
- paragraph: paragraph-based splitting

Example 1: 
PDF with fixed-size chunks
python index_documents.py test-pdf.pdf

Example 2: 
DOCX with sentence-based splitting
python index_documents.py test-docx.docx --strategy sentence

## Engineering Decisions

### Embedding Storage and Database Selection

For this assignment, PostgreSQL was chosen as the storage layer instead of a dedicated *vector database*. 
Since the assignment does not require similarity search, a vector database would add unnecessary complexity. 

In a *production-grade system* where embedding-based retrieval or similarity search is required, a vector database (or the pgvector PostgreSQL extension) would be the appropriate choice.

The Gemini Embeddings API returns dense numerical vectors. To preserve precision, embeddings are stored in PostgreSQL using the `DOUBLE PRECISION[]` data type.

Relevant docs:  
https://www.postgresql.org/docs/current/datatype-numeric.html  

## Text Normalization
Text normalization is applied after chunking to preserve sentence and paragraph boundaries required by different splitting strategies.

## Token Safety and Input Size Limits

Each call to the Gemini Embeddings API may include **at most 2,048 input tokens**.
To guarantee that every chunk sent to Gemini complies with this constraint, a safety mechanism is applied before embedding generation.

Oversized chunks are split into smaller sub-chunks, regardless of the selected chunking strategy.

For Gemini models, **one token corresponds to approximately four characters**. Based on this approximation, a safe maximum chunk size of: `MAX_CHARS'` = 6000 is used.

Relevant docs:  
Gemini token counting: https://ai.google.dev/gemini-api/docs/tokens?lang=python

## Example Output

Example Console Output: 

<img width="816" height="154" alt="image" src="https://github.com/user-attachments/assets/b04bc6c7-4303-4cec-9cda-5632f26386ea" />

Example Database Record:

<img width="764" height="169" alt="image" src="https://github.com/user-attachments/assets/cf4eca70-b450-47e6-8212-2c585174c739" />






