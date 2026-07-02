# FuseRAG

![FuseRAG Frontend](screenshot.png)

FuseRAG is a production-minded RAG system for chatting with documents, code, markdown, and web content without treating retrieval like an afterthought.

It is called **FuseRAG** because the core of the system is **hybrid retrieval**: it fuses semantic vector search with keyword search so the model can find both conceptual matches and exact language.  
The name also reflects the broader design of the project: multiple retrieval signals, reranking, and evaluation are fused into one end-to-end pipeline.

## Why I Built This

Most RAG demos look convincing in screenshots, but fall apart on real queries.

They usually rely on embeddings alone, retrieve loosely relevant chunks, miss exact terms, and give you no real way to measure whether the final answer was actually grounded in the source material.

FuseRAG was built to fix that. The goal was not just to make document chat work, but to make retrieval more reliable, answers more grounded, and quality more measurable.

## What Makes FuseRAG Better Than A Basic RAG Demo

FuseRAG is stronger than a typical embeddings-only RAG pipeline in a few important ways:

- **Hybrid retrieval** combines dense vector search with BM25 keyword matching instead of trusting embeddings alone.
- **Reciprocal Rank Fusion** merges those retrieval signals so exact-match relevance and semantic similarity both matter.
- **Cross-encoder reranking** improves the final ordering of chunks before generation.
- **Evaluation is built in** with retrieval and answer-quality metrics instead of relying on intuition.
- **Multiple ingestion paths** support PDFs, code, markdown, and web pages in one system.
- **A real web app** makes the full pipeline usable: ingest, query, browse documents, and run evaluations.

In short: this is not a "chat with PDF" demo. It is a retrieval system with a chat interface on top.

## How It Works

```text
Documents -> Parsing -> Chunking -> Embeddings + Keyword Index
                                      |
                                      v
                         Hybrid Retrieval (Dense + Sparse)
                                      |
                                      v
                            Reciprocal Rank Fusion
                                      |
                                      v
                           Cross-Encoder Reranking
                                      |
                                      v
                         Prompt Construction + LLM Answer
                                      |
                                      v
                         Evaluation of Retrieval + Output
```

## What We Built

FuseRAG includes:

- A FastAPI backend with API routes, middleware, and server-rendered views
- A document ingestion pipeline for PDFs, markdown, code files, and web sources
- Multiple chunking strategies, including semantic and syntax-aware chunking
- Dense retrieval with Pinecone-compatible vector search
- Sparse retrieval with BM25 keyword search
- Reciprocal Rank Fusion for hybrid retrieval
- Cross-encoder reranking for better final relevance
- Streaming answer generation with OpenAI and Anthropic support
- An evaluation harness for retrieval quality, answer correctness, and faithfulness
- A lightweight web UI built with Jinja2 and HTMX

## Why The Retrieval Is Stronger

The biggest weakness in many RAG systems is that they assume the first retrieval pass is "good enough." In practice, it often is not.

FuseRAG improves that in three layers:

1. **Vector search** captures semantic meaning.
2. **BM25** catches exact keywords, identifiers, and terminology that embeddings often underweight.
3. **Reranking** scores query-document pairs directly, which helps surface the most useful chunks before generation.

That combination makes the system more dependable than a naive top-k similarity search, especially for technical content and mixed-format corpora.

## Evaluation First, Not Vibes First

One of the most important parts of this project is the evaluation layer.

FuseRAG does not stop at "the answer sounds good." It includes a framework for measuring:

- **Precision@k**
- **Recall@k**
- **MRR**
- **nDCG**
- **Faithfulness**
- **Correctness**
- **Latency**
- **Token usage and cost**

This matters because the real difference between a demo and a usable RAG system is whether you can prove the system is retrieving the right context and staying grounded in it.

## Product Experience

The web frontend lets you:

- upload documents or ingest them from source URLs
- ask questions and receive streamed answers
- inspect ingested documents
- run evaluations and review historical runs

It is intentionally simple, but it covers the full workflow end to end.

## Tech Stack

| Component | Technology |
|-----------|------------|
| API Framework | FastAPI |
| Vector Retrieval | Pinecone-compatible vector store |
| Keyword Retrieval | BM25 |
| Reranking | Sentence Transformers cross-encoder |
| LLM Providers | OpenAI and Anthropic |
| Metadata Storage | PostgreSQL / pgvector |
| Frontend | Jinja2 + HTMX |

## Quick Start

```bash
cd rag-system
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

Visit `http://localhost:8000/docs` for the API and `http://localhost:8000/web/query` for the web app.

## Example API Usage

Upload a document:

```bash
curl -X POST "http://localhost:8000/api/v1/ingest/upload" \
  -F "file=@document.pdf" \
  -F "chunking_strategy=recursive"
```

Ask a question:

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main conclusions?",
    "top_k": 5,
    "stream": false
  }'
```

## Project Structure

```text
rag-system/
|-- src/
|   |-- api/
|   |-- ingestion/
|   |-- retrieval/
|   |-- generation/
|   `-- evaluation/
|-- tests/
`-- scripts/
```

## Roadmap

Natural next steps for FuseRAG:

- conversation memory and follow-up context
- OCR and scanned-document support
- stronger domain-specific embedding strategies
- richer retrieval experiments and baseline comparisons
- GraphRAG-style relationship exploration

---

FuseRAG is built around a simple idea: better answers start with better retrieval.
