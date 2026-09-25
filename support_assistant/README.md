# Zepto Support Assistant

## Overview

An offline GenAI-style support assistant for Zepto policies.

The system uses local sentence-transformer embeddings, ChromaDB
for vector retrieval, LangGraph for routing, Pydantic for
structured output, and FastAPI for the API.

The default mode uses a deterministic mock response and does
not require an external LLM API.

## Architecture

The data flow is:

1. Ingestion
2. Embedding
3. Retrieval
4. Generation
5. Structured response
6. FastAPI API

### 1. Ingestion

The policy documents are stored in:

`docs/`

The eight documents are:

- doc_01.txt
- doc_02.txt
- doc_03.txt
- doc_04.txt
- doc_05.txt
- doc_06.txt
- doc_07.txt
- doc_08.txt

`ingest.py` loads the documents, splits them into chunks,
creates embeddings, and stores them in ChromaDB.

### 2. Embedding

The local model used is:

`all-MiniLM-L6-v2`

No external embedding API is required.

### 3. Retrieval

ChromaDB stores the document chunks and embeddings.

For policy questions, the query is embedded and the top 3
similar chunks are retrieved using vector similarity.

### 4. Generation

`main.py` contains three LangGraph nodes:

- `classify_intent`
- `retrieve_and_answer`
- `direct_answer`

Policy questions are routed to retrieval.

General questions are routed directly to the canned response.

### 5. MOCK_LLM

The application checks the environment variable:

`MOCK_LLM`

Default:

`MOCK_LLM=1`

In mock mode, the application uses deterministic responses
and does not make an external LLM call.

If `MOCK_LLM=0`, the code provides the branch for an optional
real LLM extension.

### 6. Structured Output

The API response follows the Pydantic schema:

```json
{
  "answer": "string",
  "sources": ["chunk_id"],
  "confidence": 1.0
}