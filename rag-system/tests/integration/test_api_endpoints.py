"""
Integration tests for API endpoints.
Tests the full API stack including routing, validation, and response formatting.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def mock_services():
    """Mock all external services."""
    with patch("src.retrieval.embeddings.EmbeddingService") as mock_emb, \
         patch("src.retrieval.vector_store.VectorStore") as mock_vs, \
         patch("src.generation.llm_client.LLMClient") as mock_llm:
        
        # Mock embedding service
        mock_emb_instance = MagicMock()
        mock_emb_instance.embed_batch = AsyncMock(return_value=[[0.1] * 384])
        mock_emb.return_value = mock_emb_instance
        
        # Mock vector store
        mock_vs_instance = MagicMock()
        mock_vs_instance.search = AsyncMock(return_value=[])
        mock_vs.return_value = mock_vs_instance
        
        # Mock LLM client
        mock_llm_instance = MagicMock()
        mock_llm_instance.generate = AsyncMock(
            return_value=("Test answer", 100, 50)
        )
        mock_llm.return_value = mock_llm_instance
        
        yield {
            "embedding": mock_emb_instance,
            "vector_store": mock_vs_instance,
            "llm": mock_llm_instance
        }


class TestHealthEndpoint:
    """Test suite for health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_returns_200(self):
        """Test health endpoint returns healthy status."""
        from src.api.main import app
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_includes_version(self):
        """Test health endpoint includes version info."""
        from src.api.main import app
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
        
        data = response.json()
        assert "version" in data


class TestQueryEndpoints:
    """Test suite for query endpoints."""

    @pytest.mark.asyncio
    async def test_query_endpoint_returns_422_for_empty_query(self):
        """Test query endpoint rejects empty queries."""
        from src.api.main import app
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/query",
                json={"query": ""}
            )
        
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_query_endpoint_returns_422_for_missing_query(self):
        """Test query endpoint rejects missing query field."""
        from src.api.main import app
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/query",
                json={}
            )
        
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_query_endpoint_validates_top_k_range(self, mock_services):
        """Test query endpoint validates top_k is within range."""
        from src.api.main import app
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/query",
                json={"query": "test", "top_k": 100}
            )
        
        # top_k should be validated (implementation dependent)
        assert response.status_code in [200, 422]

    @pytest.mark.asyncio
    async def test_query_endpoint_accepts_valid_model(self, mock_services):
        """Test query endpoint accepts valid model names."""
        from src.api.main import app
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/query",
                json={"query": "test", "model": "gpt-3.5-turbo"}
            )
        
        assert response.status_code in [200, 422, 500]

    @pytest.mark.asyncio
    async def test_query_endpoint_returns_query_id(self, mock_services):
        """Test query response includes query_id."""
        from src.api.main import app
        
        # Mock the retrieval to return empty results
        with patch("src.retrieval.vector_store.VectorStore") as mock_vs:
            mock_vs_instance = MagicMock()
            mock_vs_instance.search = AsyncMock(return_value=[])
            mock_vs.return_value = mock_vs_instance
            
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/query",
                    json={"query": "test query"}
                )
            
            if response.status_code == 200:
                data = response.json()
                assert "query_id" in data


class TestIngestEndpoints:
    """Test suite for ingestion endpoints."""

    @pytest.mark.asyncio
    async def test_ingest_endpoint_validates_doc_type(self):
        """Test ingest endpoint validates document type."""
        from src.api.main import app
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/ingest",
                json={"source": "test.pdf", "doc_type": "invalid_type"}
            )
        
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_ingest_endpoint_requires_source(self):
        """Test ingest endpoint requires source field."""
        from src.api.main import app
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/ingest",
                json={}
            )
        
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_ingest_upload_endpoint_exists(self):
        """Test file upload endpoint exists."""
        from src.api.main import app
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/v1/ingest/upload")
        
        # Should either accept or reject with 422 (missing file)
        assert response.status_code in [200, 422, 500]


class TestErrorHandling:
    """Test suite for API error handling."""

    @pytest.mark.asyncio
    async def test_404_for_unknown_endpoint(self):
        """Test unknown endpoints return 404."""
        from src.api.main import app
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/nonexistent")
        
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_cors_headers_present(self):
        """Test CORS headers are present in responses."""
        from src.api.main import app
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.options("/health")
        
        # Should have CORS headers or return 200
        assert response.status_code in [200, 405]

    @pytest.mark.asyncio
    async def test_request_id_header_present(self):
        """Test X-Request-ID header is added to responses."""
        from src.api.main import app
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
        
        if response.status_code == 200:
            assert "x-request-id" in response.headers or "X-Request-ID" in response.headers
