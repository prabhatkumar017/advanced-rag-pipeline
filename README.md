# Advanced RAG Pipeline

A production-style Retrieval-Augmented Generation (RAG) system built from scratch. This project covers the full lifecycle — document ingestion, hybrid retrieval, reranking, LLM-based answer generation, a REST API, a frontend UI, evaluation, and a feedback loop.

---

## Table of Contents

1. [What is RAG?](#what-is-rag)
2. [Project Architecture](#project-architecture)
3. [Folder Structure](#folder-structure)
4. [Component Deep Dive](#component-deep-dive)
   - [Ingestion](#1-ingestion)
   - [Retrieval](#2-retrieval)
   - [Generation](#3-generation)
   - [API](#4-api)
   - [Frontend](#5-frontend)
   - [Evaluation](#6-evaluation)
   - [Feedback](#7-feedback)
5. [How to Run](#how-to-run)
6. [Environment Variables](#environment-variables)
7. [Interview Concepts Cheatsheet](#interview-concepts-cheatsheet)

---

## What is RAG?

**RAG (Retrieval-Augmented Generation)** is a technique that improves LLM responses by grounding them in external knowledge.

Instead of relying solely on what the LLM was trained on, RAG:
1. **Retrieves** relevant documents from a knowledge base at query time
2. **Augments** the LLM prompt with that retrieved context
3. **Generates** an answer grounded in the retrieved content

**Why RAG over fine-tuning?**
- No expensive retraining needed when knowledge changes
- LLM answers stay grounded — reduces hallucination
- You can cite sources
- Works well for domain-specific or up-to-date knowledge

---

## Project Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                     INGESTION                        │
│  Load Docs → Chunk → Embed → FAISS Index + BM25     │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                     RETRIEVAL                        │
│  Vector Search + BM25 Search → Hybrid → Reranker    │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                    GENERATION                        │
│  Prompt Template + Context → OpenAI GPT → Answer    │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│               API + FRONTEND + LOGGING               │
│  FastAPI /ask → Streamlit UI → logs.json            │
└─────────────────────────────────────────────────────┘
```

---

## Folder Structure

```
advanced-rag-pipeline/
├── main.py                          # CLI entry point — full pipeline loop
├── requirements.txt
├── .env                             # API keys (not committed)
│
├── ingestion/
│   ├── document_loader.py           # Load documents from data source
│   ├── chunking.py                  # Split documents into chunks
│   └── embedding.py                 # Convert text to vectors
│
├── retrieval/
│   ├── vector_search.py             # FAISS semantic search
│   ├── bm25_search.py               # BM25 keyword search
│   ├── hybrid_search.py             # Merge vector + BM25 results
│   └── reranker.py                  # Cross-encoder reranking
│
├── generation/
│   ├── answer_generator.py          # Send prompt to OpenAI, get answer
│   └── prompt_templates.py          # Prompt engineering
│
├── api/
│   └── rag_api.py                   # FastAPI REST endpoint
│
├── frontend/
│   └── streamlit_app.py             # Streamlit chat UI
│
├── evaluation/
│   ├── ragas_eval.py                # RAGAS metrics evaluation
│   └── hallucination_detection.py   # LLM-as-judge hallucination check
│
└── feedback/
    ├── interaction_logger.py        # Log every query + answer to logs.json
    └── improvement_loop.py          # Analyze logs for usage patterns
```

---

## Component Deep Dive

### 1. Ingestion

The ingestion pipeline prepares raw documents for retrieval. It has three stages:

#### `document_loader.py`
Loads the source dataset. Currently uses the **CNN/DailyMail** dataset (200 news articles) via HuggingFace `datasets`.

```python
dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:200]")
```

In production this would support PDFs, CSVs, databases, web scrapes, etc.

#### `chunking.py`
LLMs have context length limits. Long documents must be split into smaller **chunks** before embedding.

Uses LangChain's `RecursiveCharacterTextSplitter`:
- `chunk_size=500` — each chunk is max 500 characters
- `chunk_overlap=100` — 100 characters overlap between chunks to preserve context at boundaries

**Why overlap?** Without overlap, a sentence split across two chunks loses its meaning in both.

#### `embedding.py`
Converts text chunks into dense vector representations using `sentence-transformers`.

Model used: **`BAAI/bge-small-en`**
- Lightweight, fast
- Good quality for English semantic similarity
- Runs locally — no API cost

```python
model = SentenceTransformer("BAAI/bge-small-en")
embeddings = model.encode(texts)  # returns numpy array of shape (N, 384)
```

---

### 2. Retrieval

Retrieval is the most critical part of RAG. Poor retrieval = poor answers, regardless of the LLM quality.

#### `vector_search.py`
Uses **FAISS** (Facebook AI Similarity Search) to perform fast approximate nearest-neighbor search in vector space.

```python
index = faiss.IndexFlatL2(dim)   # L2 = Euclidean distance
index.add(embeddings)
distances, indices = index.search(query_embedding, k=5)
```

**Why FAISS?**
- Extremely fast even with millions of vectors
- Runs locally, no external DB needed
- `IndexFlatL2` is exact (brute-force) — good for small datasets

#### `bm25_search.py`
**BM25 (Best Match 25)** is a classical keyword-based ranking algorithm used by search engines like Elasticsearch.

It scores documents based on term frequency and inverse document frequency (TF-IDF family).

**Why BM25 alongside vector search?**
- Vector search is good at *semantic* similarity ("car" ≈ "vehicle")
- BM25 is good at *exact* keyword matches ("GPT-4o" must appear in the doc)
- Together they cover each other's blind spots

#### `hybrid_search.py`
Combines results from both vector search and BM25 using a simple **union merge** strategy:

```python
combined = list(set(vector_results.tolist() + bm25_results))
```

In production, **Reciprocal Rank Fusion (RRF)** is a more principled approach to hybrid merging.

#### `reranker.py`
After retrieval, a **cross-encoder** reranks the top results for higher precision.

Model: **`cross-encoder/ms-marco-MiniLM-L-6-v2`**

**Bi-encoder vs Cross-encoder:**
| | Bi-encoder (embedding model) | Cross-encoder (reranker) |
|---|---|---|
| Input | Query and doc encoded separately | Query + doc encoded together |
| Speed | Fast (pre-compute doc embeddings) | Slow (must run per pair) |
| Accuracy | Good | Better |
| Use case | First-stage retrieval (top-k) | Second-stage reranking (top-5) |

The two-stage approach (retrieve many → rerank to few) gives the best accuracy/speed tradeoff.

---

### 3. Generation

#### `prompt_templates.py`
Defines the prompt structure sent to the LLM.

```
You are a news assistant trained on CNN/DailyMail-style articles.
Answer the question using ONLY the provided context.

Context:
{context}

Question:
{question}

Answer:
```

**Why "ONLY the provided context"?** Forces the LLM to stay grounded and not hallucinate from training data.

#### `answer_generator.py`
Sends the formatted prompt to **OpenAI GPT-4o-mini** and returns the response.

- Loads API key from `.env` via `python-dotenv`
- Initializes the OpenAI client per call (keeps it stateless and testable)
- Model: `gpt-4o-mini` — cost-efficient, fast, good quality

---

### 4. API

#### `api/rag_api.py`
A **FastAPI** REST API that exposes the full RAG pipeline over HTTP.

**Startup:** The pipeline (doc loading, embedding, index building) runs once when the server starts — not per request. This avoids re-embedding thousands of documents on every call.

**Endpoint:**
```
GET /ask?question=<your question>
→ {"answer": "..."}
```

Run with:
```bash
uvicorn api.rag_api:app --reload
```

---

### 5. Frontend

#### `frontend/streamlit_app.py`
A minimal **Streamlit** web UI that calls the FastAPI backend.

- User types a question
- Hits Submit
- UI calls `GET http://localhost:8000/ask?question=...`
- Displays the answer

Run with:
```bash
streamlit run frontend/streamlit_app.py
```

> Both the API server and the Streamlit app must be running at the same time.

---

### 6. Evaluation

#### `evaluation/ragas_eval.py`
Uses the **RAGAS** framework to evaluate RAG quality with three metrics:

| Metric | What it measures |
|---|---|
| `faithfulness` | Is the answer supported by the retrieved context? |
| `answer_relevancy` | Does the answer actually address the question? |
| `context_precision` | Are the retrieved chunks relevant to the question? |

RAGAS uses an LLM internally to score these metrics.

#### `evaluation/hallucination_detection.py`
A lightweight LLM-as-judge approach: asks GPT-4o-mini to check whether the answer is supported by the provided context. Returns `YES` or `NO`.

---

### 7. Feedback

#### `feedback/interaction_logger.py`
Logs every query, answer, and context to `logs.json` as newline-delimited JSON:

```json
{"timestamp": "2026-04-16T10:00:00", "query": "...", "answer": "...", "context": "..."}
```

#### `feedback/improvement_loop.py`
Reads `logs.json` and surfaces:
- Total number of interactions
- Top 10 most frequent query terms
- Last 5 queries with timestamps

Use this to identify what topics users ask about most and where retrieval may be failing.

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up environment variables
Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-...
```

### 3. Run CLI pipeline
```bash
python main.py
```

### 4. Run API server
```bash
uvicorn api.rag_api:app --reload
```

### 5. Run frontend
```bash
streamlit run frontend/streamlit_app.py
```

### 6. Analyze logs
```python
from feedback.improvement_loop import analyze_logs
analyze_logs()
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key for generation and hallucination detection |

---

## Interview Concepts Cheatsheet

### RAG vs Fine-tuning
- **RAG**: Inject knowledge at inference time. No retraining. Knowledge can be updated cheaply. Better for dynamic or proprietary data.
- **Fine-tuning**: Bake knowledge into model weights. Expensive. Better for style/behavior changes, not factual knowledge.

### Chunking strategy matters
- Too large → irrelevant content dilutes the context
- Too small → loses surrounding context needed for meaning
- Overlap prevents boundary artifacts

### Why hybrid search?
- Vector search alone misses exact keyword matches
- BM25 alone misses paraphrased or semantic queries
- Hybrid gives better recall than either alone

### Two-stage retrieval
- Stage 1 (bi-encoder): Fast, retrieve top-20 candidates
- Stage 2 (cross-encoder): Slow but precise, rerank to top-5
- This is how modern search engines work (e.g., Google, Bing)

### FAISS index types
- `IndexFlatL2` — exact brute-force, accurate, slow at scale
- `IndexIVFFlat` — clusters vectors, faster but approximate
- `IndexHNSW` — graph-based, very fast approximate search

### Hallucination in RAG
- Can happen if retrieved context doesn't contain the answer but LLM fills in the gap
- Mitigated by: strong prompt constraints, faithfulness scoring (RAGAS), LLM-as-judge checks

### RAGAS metrics
- **Faithfulness**: answer ⊆ context (no made-up facts)
- **Answer relevancy**: answer addresses the question
- **Context precision**: retrieved chunks are on-topic

### Common RAG failure modes
| Failure | Cause | Fix |
|---|---|---|
| Wrong chunks retrieved | Poor embedding model or chunking | Better chunker, reranker |
| Answer ignores context | Weak prompt | Stronger instruction in prompt |
| Hallucinated answer | LLM adds info not in context | Add faithfulness check |
| Slow response | Embedding on every request | Pre-compute and cache embeddings |
