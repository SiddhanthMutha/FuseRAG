# Deployment Guide

## Prerequisites

- Docker and Docker Compose
- Python 3.11+
- PostgreSQL 15 with pgvector
- Pinecone account (or self-hosted vector DB)

## Environment Variables

Create a `.env` file:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/ragdb

# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic (optional)
ANTHROPIC_API_KEY=sk-ant-...

# Pinecone
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-west1
PINECONE_INDEX_NAME=rag-index

# Application
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
ENABLE_QUERY_LOGGING=true
```

## Quick Start (Docker)

### 1. Clone and Setup

```bash
cd rag-system
cp .env.example .env
# Edit .env with your API keys
```

### 2. Start Infrastructure

```bash
cd infra/docker
docker-compose up -d
```

This starts:
- PostgreSQL with pgvector (port 5432)
- PGAdmin (port 5050) - optional

### 3. Run the Application

```bash
# Using Docker
docker build -f infra/docker/Dockerfile -t rag-system .
docker run -p 8000:8000 --env-file .env rag-system

# Or using Python directly
pip install -r requirements.txt
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 4. Verify

```bash
curl http://localhost:8000/health
```

## Development Setup

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Setup database
python scripts/setup_database.py

# Run tests
pytest tests/unit/ -v

# Start development server
uvicorn src.api.main:app --reload
```

### Using Docker Compose for Development

```bash
# Start all services including the app
cd infra/docker
docker-compose -f docker-compose.yml up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

## Production Deployment

### Docker Production Build

```bash
# Build production image
docker build -f infra/docker/Dockerfile -t rag-system:prod .

# Run with environment file
docker run -d \
  --name rag-system \
  -p 8000:8000 \
  --env-file .env \
  rag-system:prod
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f infra/k8s/

# Check deployment status
kubectl get pods -n rag-system

# View logs
kubectl logs -n rag-system -l app=rag-system
```

## Database Setup

### PostgreSQL with pgvector

The system uses pgvector for efficient vector similarity search.

```bash
# Using Docker
docker run -d \
  -e POSTGRES_USER=raguser \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=ragdb \
  -p 5432:5432 \
  pgvector/pgvector:pg15

# Run setup script
python scripts/setup_database.py
```

### Pinecone Setup

1. Create a Pinecone account
2. Create an index with:
   - Dimension: 1536 (for ada-002) or 384 (for smaller models)
   - Metric: cosine
   - Pod type: p1 (or p2 for production)

## Configuration

### Application Settings

| Variable | Default | Description |
|----------|---------|-------------|
| DATABASE_URL | - | PostgreSQL connection string |
| OPENAI_API_KEY | - | OpenAI API key |
| ANTHROPIC_API_KEY | - | Anthropic API key |
| PINECONE_API_KEY | - | Pinecone API key |
| DEFAULT_CHUNK_SIZE | 500 | Default chunk size in tokens |
| DEFAULT_CHUNK_OVERLAP | 50 | Chunk overlap in tokens |
| DEFAULT_TOP_K | 5 | Default number of results |
| ENABLE_QUERY_LOGGING | true | Log queries to database |

### Model Pricing

| Model | Input ($/1K) | Output ($/1K) |
|-------|-------------|---------------|
| gpt-3.5-turbo | 0.0015 | 0.002 |
| gpt-4 | 0.03 | 0.06 |
| gpt-4o | 0.005 | 0.015 |
| claude-3-5-sonnet | 0.003 | 0.015 |

## Monitoring

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health (if enabled)
curl http://localhost:8000/health/detailed
```

### Logs

Logs are output in JSON format via loguru:

```bash
# View logs
tail -f logs/app.log

# Search logs
grep "ERROR" logs/app.log
```

### Metrics

The system tracks:
- Query latency (p50, p95, p99)
- Token usage per query
- Cost per query
- Retrieval precision

## Troubleshooting

### Common Issues

**Database Connection Failed**
```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# Check logs
docker logs postgres
```

**Pinecone Connection Error**
```bash
# Verify API key
echo $PINECONE_API_KEY

# Check index exists
curl -X GET "https://controller.us-west1.pinecone.io/indexes" \
  -H "Api-Key: $PINECONE_API_KEY"
```

**LLM API Errors**
```bash
# Check API keys are set
echo $OPENAI_API_KEY

# Verify key works
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Performance Issues

- Increase `PINECONE_TOP_K` for more results
- Use smaller chunk sizes for faster retrieval
- Enable caching for repeated queries
- Scale horizontally with more instances

## Security

### Production Checklist

- [ ] Use strong database passwords
- [ ] Enable SSL/TLS for database connections
- [ ] Use secrets management (Kubernetes secrets, AWS Secrets Manager)
- [ ] Enable API authentication
- [ ] Configure CORS properly
- [ ] Use non-root user in Docker
- [ ] Enable request rate limiting

## Backup and Recovery

### Database Backup

```bash
# Backup PostgreSQL
docker exec postgres pg_dump -U raguser ragdb > backup.sql

# Restore
docker exec -i postgres psql -U raguser ragdb < backup.sql
```

### Vector Index Backup

Pinecone handles indexing automatically. For self-hosted solutions, use standard backup procedures.
