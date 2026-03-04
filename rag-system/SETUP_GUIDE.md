# RAG System Setup Guide

## Prerequisites

Before running the RAG system, you need to set up the following:

### 1. Required API Keys

Create a `.env` file in the `rag-system` directory with these keys:

```bash
# Required: OpenAI API Key (for LLM and embeddings)
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-your-openai-key-here

# Optional: Anthropic Claude (alternative LLM)
# Get from: https://console.anthropic.com/
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# Required: Pinecone (vector database)
# Get from: https://app.pinecone.io/
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=rag-system

# PostgreSQL Database URL
# Format: postgresql+asyncpg://username:password@host:port/database
DATABASE_URL=postgresql+asyncpg://raguser:ragpassword@localhost:5432/ragdb
```

### 2. Get API Keys

**OpenAI:**
1. Go to https://platform.openai.com/
2. Sign up/Login
3. Go to API Keys section
4. Create a new secret key

**Pinecone:**
1. Go to https://app.pinecone.io/
2. Sign up/Login
3. Create a new index with:
   - Name: `rag-system`
   - Dimension: `1536` (for ada-002) or `384` (for MiniLM)
   - Metric: `cosine`
   - Pod type: `p1` (starter)

### 3. Set Up PostgreSQL

**Option A - Docker (Recommended):**
```bash
cd rag-system/infra/docker
docker-compose up -d postgres
```

**Option B - Local Installation:**
```bash
# Install PostgreSQL 15 with pgvector
# Then create database:
createdb ragdb
```

## How to Run

### Option 1: Run Unit Tests Only (No API Keys Needed)

This verifies the core logic works:

```bash
cd rag-system
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run unit tests
pytest tests/unit/ -v
```

### Option 2: Run with Docker (Full System)

```bash
cd rag-system

# 1. Make sure your .env file is set up with API keys
# 2. Start PostgreSQL and the app
cd infra/docker
docker-compose up -d

# 3. Check if running
curl http://localhost:8000/health
```

### Option 3: Run Locally with Python

```bash
cd rag-system

# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up PostgreSQL (run Docker separately)
docker run -d -e POSTGRES_USER=raguser -e POSTGRES_PASSWORD=ragpassword -e POSTGRES_DB=ragdb -p 5432:5432 pgvector/pgvector:pg15

# 4. Run database setup
python scripts/setup_database.py

# 5. Start the API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

## Testing the API

Once the server is running:

### Health Check
```bash
curl http://localhost:8000/health
```

### Query the System
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "top_k": 3}'
```

### Ingest a Document
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"source": "https://example.com/document.pdf", "doc_type": "pdf"}'
```

## Troubleshooting

### "Database connection failed"
- Make sure PostgreSQL is running
- Check DATABASE_URL in .env

### "Pinecone connection error"
- Verify PINECONE_API_KEY is correct
- Check Pinecone index exists

### "OpenAI API error"
- Verify OPENAI_API_KEY is set
- Check API key has credits

### Tests fail with missing dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```
