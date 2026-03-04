# RAG System Architecture

## Overview

The RAG (Retrieval-Augmented Generation) system is a full-stack solution that combines document retrieval with language model generation to provide accurate, context-aware answers from a knowledge base.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Client Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Web UI    │  │    API      │  │  WebSocket  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        API Layer (FastAPI)                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Middleware                                │   │
│  │  • LoggingMiddleware (UUID, latency tracking)              │   │
│  │  • ErrorHandlingMiddleware (consistent error format)       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      Routes                                 │   │
│  │  • /health - Health check                                  │   │
│  │  • /api/v1/query - Synchronous query                       │   │
│  │  • /api/v1/query/stream - WebSocket streaming             │   │
│  │  • /api/v1/ingest - Document ingestion                    │   │
│  │  • /api/v1/ingest/upload - File upload                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Retrieval Layer                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  HybridRetriever                                           │   │
│  │  ├── VectorStore (Pinecone) - Dense semantic search       │   │
│  │  └── KeywordSearch (BM25) - Sparse keyword search          │   │
│  │       │                                                     │   │
│  │       ▼                                                     │   │
│  │  Reranker (Cross-Encoder) - Reorder top-K results         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  EmbeddingService (OpenAI text-embedding-ada-002)         │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Generation Layer                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  LLMClient                                                  │   │
│  │  ├── OpenAI (GPT-3.5, GPT-4)                               │   │
│  │  └── Anthropic (Claude 3.5 Sonnet)                        │   │
│  │       │                                                     │   │
│  │       ▼                                                     │   │
│  │  ContextManager - Fit chunks into context window           │   │
│  │  PromptBuilder - Construct RAG prompts                      │   │
│  │  StreamingHandler - WebSocket token streaming               │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Storage Layer                                  │
│  ┌──────────────────────┐    ┌──────────────────────┐           │
│  │   PostgreSQL         │    │   Pinecone            │           │
│  │   (Metadata/Logs)    │    │   (Vectors)          │           │
│  └──────────────────────┘    └──────────────────────┘           │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. API Layer

**FastAPI Application**
- Async request handling
- WebSocket support for streaming
- Middleware for logging and error handling

**Endpoints:**
- `GET /health` - Health check
- `POST /api/v1/query` - Synchronous query with full response
- `WebSocket /api/v1/query/stream` - Streaming query
- `POST /api/v1/ingest` - Ingest document by URL or path
- `POST /api/v1/ingest/upload` - Upload and ingest PDF file

### 2. Retrieval Layer

**Hybrid Retriever**
Combines dense (vector) and sparse (keyword) retrieval methods for better coverage.

**Vector Search (Pinecone)**
- Semantic similarity search
- Supports metadata filtering
- HNSW indexing for fast queries

**Keyword Search (BM25)**
- Traditional TF-IDF based retrieval
- Complementary to vector search

**Reranker**
- Cross-encoder model for reordering results
- Improves precision at the cost of speed

### 3. Generation Layer

**LLM Clients**
- OpenAI: GPT-3.5-turbo, GPT-4, GPT-4o
- Anthropic: Claude 3.5 Sonnet

**Streaming Support**
- Token-by-token streaming over WebSocket
- Real-time progress updates

**Context Management**
- Intelligent chunk selection based on token limits
- Supports different context windows per model

### 4. Storage Layer

**PostgreSQL**
- Document metadata storage
- Query logs
- User data (future)

**Pinecone**
- Vector embeddings storage
- Similarity search

## Data Flow

### Query Flow

1. Client sends query to `/api/v1/query` or WebSocket
2. Query is embedded using EmbeddingService
3. HybridRetriever performs parallel vector + keyword search
4. Results are reranked using cross-encoder
5. ContextManager selects relevant chunks within context limit
6. PromptBuilder constructs RAG prompt
7. LLM generates response (streaming or batch)
8. Sources and metadata returned to client

### Ingestion Flow

1. Client uploads document (PDF/URL/path)
2. Parser extracts text based on document type
3. Chunker splits into appropriate chunks
4. EmbeddingService generates embeddings
5. Chunks upserted to Pinecone
6. Metadata stored in PostgreSQL
7. Confirmation returned to client

## Technology Stack

| Component | Technology |
|-----------|------------|
| API Framework | FastAPI |
| Web Server | Uvicorn |
| Database | PostgreSQL + pgvector |
| Vector Store | Pinecone |
| LLM Providers | OpenAI, Anthropic |
| Embeddings | OpenAI text-embedding-ada-002 |
| Testing | pytest |
| Logging | loguru |
| Deployment | Docker, Kubernetes |

## Security

- API key authentication via environment variables
- Request ID tracking for audit trails
- Input validation and sanitization
- Error message sanitization in production

## Performance Considerations

- Connection pooling for database
- Async/await throughout the stack
- Streaming responses to reduce latency
- Embedding batching for efficiency
- Vector index optimization (HNSW)
