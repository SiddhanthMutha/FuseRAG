# API Reference

## Base URL

```
http://localhost:8000
```

## Authentication

All endpoints require API keys to be set in environment variables:

- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `PINECONE_API_KEY` - Pinecone API key
- `DATABASE_URL` - PostgreSQL connection string

## Endpoints

### Health Check

#### GET /health

Check API health status.

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

---

### Query Endpoints

#### POST /api/v1/query

Query the RAG system (non-streaming).

**Request Body:**

```json
{
  "query": "What is RAG?",
  "top_k": 5,
  "model": "gpt-3.5-turbo",
  "filters": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| query | string | The question to ask |
| top_k | integer | Number of chunks to retrieve (default: 5) |
| model | string | LLM model to use (gpt-3.5-turbo, gpt-4, claude-3-5-sonnet) |
| filters | object | Optional metadata filters |

**Response:**

```json
{
  "query_id": "uuid-string",
  "answer": "RAG stands for Retrieval-Augmented Generation...",
  "sources": [
    {
      "chunk_id": "chunk-123",
      "content": "RAG is a technique...",
      "score": 0.95,
      "document_source": "doc.pdf",
      "metadata": {"page": 1}
    }
  ],
  "metadata": {
    "tokens_used": 1500,
    "latency_ms": 2500,
    "cost_usd": 0.003,
    "model": "gpt-3.5-turbo"
  }
}
```

#### WebSocket /api/v1/query/stream

Query the RAG system with streaming response.

**Client sends:**

```json
{
  "query": "What is RAG?",
  "top_k": 5,
  "model": "gpt-3.5-turbo"
}
```

**Server sends:**

```json
// Token stream
{"type": "token", "data": "RAG "}
{"type": "token", "data": "stands "}
{"type": "token", "data": "for "}
// ...

// Sources when done
{"type": "sources", "data": [...]}

// Completion signal
{"type": "done", "data": null}

// Error (if any)
{"type": "error", "data": "error message"}
```

---

### Ingestion Endpoints

#### POST /api/v1/ingest

Ingest a document by URL or file path.

**Request Body:**

```json
{
  "source": "https://example.com/article.pdf",
  "doc_type": "pdf",
  "chunking_strategy": "semantic",
  "metadata": {
    "author": "John Doe",
    "title": "Article Title"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| source | string | URL or file path |
| doc_type | string | pdf, web, code, markdown |
| chunking_strategy | string | fixed_size, semantic, syntax_aware |
| metadata | object | Optional metadata |

**Response:**

```json
{
  "document_id": "doc-uuid",
  "chunks_created": 25,
  "status": "success",
  "errors": []
}
```

#### POST /api/v1/ingest/upload

Upload and ingest a PDF file.

**Content-Type:** multipart/form-data

**Form Data:**

| Field | Type | Description |
|-------|------|-------------|
| file | binary | PDF file |
| chunking_strategy | string | semantic (default) |

**Response:**

```json
{
  "document_id": "doc-uuid",
  "chunks_created": 25,
  "status": "success",
  "errors": []
}
```

---

## Example Usage

### cURL Examples

**Query the system:**

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is a transformer in machine learning?",
    "top_k": 3,
    "model": "gpt-3.5-turbo"
  }'
```

**Ingest a document:**

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "source": "https://example.com/document.pdf",
    "doc_type": "pdf",
    "chunking_strategy": "semantic"
  }'
```

**Upload a file:**

```bash
curl -X POST http://localhost:8000/api/v1/ingest/upload \
  -F "file=@./document.pdf" \
  -F "chunking_strategy=semantic"
```

### Python Example

```python
import httpx

# Query
response = httpx.post(
    "http://localhost:8000/api/v1/query",
    json={
        "query": "What is RAG?",
        "top_k": 5,
        "model": "gpt-3.5-turbo"
    }
)
print(response.json())

# Ingest
response = httpx.post(
    "http://localhost:8000/api/v1/ingest",
    json={
        "source": "https://example.com/doc.pdf",
        "doc_type": "pdf"
    }
)
print(response.json())
```

### WebSocket Example (JavaScript)

```javascript
const ws = new WebSocket("ws://localhost:8000/api/v1/query/stream");

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === "token") {
    console.log("Token:", data.data);
  } else if (data.type === "sources") {
    console.log("Sources:", data.data);
  } else if (data.type === "done") {
    console.log("Complete!");
  } else if (data.type === "error") {
    console.error("Error:", data.data);
  }
};

ws.send(JSON.stringify({
  query: "What is RAG?",
  top_k: 5
}));
```

---

## Error Responses

All endpoints return consistent error responses:

```json
{
  "error": "Error message",
  "request_id": "uuid-string",
  "detail": "Detailed error info (production only)"
}
```

**Common Status Codes:**

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 422 | Validation Error |
| 500 | Internal Server Error |
| 502 | Bad Gateway (LLM error) |
