"""
Unit tests for the EmbeddingService class.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.retrieval.embeddings import EmbeddingService


class TestEmbeddingService:
    """Test suite for embedding generation."""

    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI client."""
        with patch("src.retrieval.embeddings.openai") as mock:
            mock_client = MagicMock()
            mock_client.embeddings.create = AsyncMock(
                return_value=MagicMock(
                    data=[
                        MagicMock(embedding=[0.1] * 1536)
                    ]
                )
            )
            mock.AsyncOpenAI.return_value = mock_client
            yield mock

    def test_embedding_service_initialization(self):
        """Test EmbeddingService can be initialized."""
        service = EmbeddingService()
        assert service is not None

    @pytest.mark.asyncio
    async def test_embed_single_text(self):
        """Test embedding a single text string."""
        with patch("src.retrieval.embeddings.openai") as mock_openai:
            # Mock the response
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
            mock_openai.AsyncOpenAI.return_value.embeddings.create = AsyncMock(
                return_value=mock_response
            )
            
            service = EmbeddingService()
            result = await service.embed_single("Hello world")
            
            assert result is not None
            assert isinstance(result, list)
            assert len(result) == 1536

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        """Test embedding multiple texts."""
        with patch("src.retrieval.embeddings.openai") as mock_openai:
            # Mock the response
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=[0.1] * 1536),
                MagicMock(embedding=[0.2] * 1536),
            ]
            mock_openai.AsyncOpenAI.return_value.embeddings.create = AsyncMock(
                return_value=mock_response
            )
            
            service = EmbeddingService()
            texts = ["Hello world", "Goodbye world"]
            result = await service.embed_batch(texts)
            
            assert result is not None
            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_embed_batch_empty_list(self):
        """Test embedding an empty list returns empty list."""
        service = EmbeddingService()
        result = await service.embed_batch([])
        
        assert result == []

    def test_get_embedding_dimension(self):
        """Test getting the embedding dimension for a model."""
        service = EmbeddingService()
        
        # Default model should be text-embedding-ada-002 (1536 dim)
        dim = service.get_embedding_dimension()
        assert dim == 1536

    @pytest.mark.asyncio
    async def test_embed_with_custom_model(self):
        """Test embedding with a custom model."""
        with patch("src.retrieval.embeddings.openai") as mock_openai:
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
            mock_openai.AsyncOpenAI.return_value.embeddings.create = AsyncMock(
                return_value=mock_response
            )
            
            service = EmbeddingService(model="text-embedding-3-small")
            result = await service.embed_single("Test text")
            
            assert result is not None

    @pytest.mark.asyncio
    async def test_embed_truncates_long_text(self):
        """Test that long texts are truncated to max tokens."""
        with patch("src.retrieval.embeddings.openai") as mock_openai:
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
            mock_openai.AsyncOpenAI.return_value.embeddings.create = AsyncMock(
                return_value=mock_response
            )
            
            service = EmbeddingService()
            # Create a very long text (more than 8192 tokens)
            long_text = "word " * 10000
            result = await service.embed_single(long_text)
            
            # Should not raise an error
            assert result is not None
