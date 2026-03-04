"""
Unit tests for the VectorStore and HybridRetriever classes.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.models import RetrievalResult


class TestVectorStore:
    """Test suite for VectorStore."""

    @pytest.fixture
    def mock_pinecone(self):
        """Mock Pinecone client."""
        with patch("src.retrieval.vector_store.pinecone") as mock:
            mock_index = MagicMock()
            mock_index.query = AsyncMock(
                return_value={
                    "matches": [
                        {"id": "chunk_1", "score": 0.95, "metadata": {"content": "test content 1"}},
                        {"id": "chunk_2", "score": 0.85, "metadata": {"content": "test content 2"}},
                    ]
                }
            )
            mock_index.upsert = AsyncMock(return_value={"upserted_count": 2})
            mock.Index.return_value = mock_index
            yield mock

    def test_vector_store_initialization(self):
        """Test VectorStore can be initialized."""
        store = VectorStore()
        assert store is not None

    @pytest.mark.asyncio
    async def test_search_returns_results(self, mock_pinecone):
        """Test search returns retrieval results."""
        store = VectorStore()
        query_embedding = [0.1] * 384
        
        results = await store.search(query_embedding=query_embedding, top_k=5)
        
        assert results is not None
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_with_filters(self, mock_pinecone):
        """Test search with metadata filters."""
        store = VectorStore()
        query_embedding = [0.1] * 384
        
        results = await store.search(
            query_embedding=query_embedding,
            top_k=5,
            filters={"source": "test.pdf"}
        )
        
        assert results is not None

    @pytest.mark.asyncio
    async def test_upsert_chunks(self, mock_pinecone):
        """Test upserting chunks to vector store."""
        store = VectorStore()
        
        mock_chunks = [
            MagicMock(
                id="chunk_1",
                content="Test content",
                embedding=[0.1] * 384,
                metadata={"source": "test.pdf"}
            )
        ]
        
        result = await store.upsert_chunks(mock_chunks)
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_search_empty_embedding(self, mock_pinecone):
        """Test search with empty embedding returns empty list."""
        store = VectorStore()
        
        results = await store.search(query_embedding=[], top_k=5)
        
        assert results == []


class TestHybridRetriever:
    """Test suite for HybridRetriever."""

    @pytest.fixture
    def mock_vector_store(self):
        """Mock VectorStore."""
        with patch("src.retrieval.hybrid_retriever.VectorStore") as mock:
            mock_instance = MagicMock()
            mock_instance.search = AsyncMock(
                return_value=[
                    RetrievalResult(
                        chunk_id="chunk_1",
                        content="Vector result 1",
                        score=0.9,
                        document_source="doc1.pdf",
                        metadata={}
                    )
                ]
            )
            mock.return_value = mock_instance
            yield mock

    @pytest.fixture
    def mock_keyword_search(self):
        """Mock keyword search."""
        with patch("src.retrieval.hybrid_retriever.KeywordSearch") as mock:
            mock_instance = MagicMock()
            mock_instance.search = AsyncMock(
                return_value=[
                    RetrievalResult(
                        chunk_id="chunk_2",
                        content="Keyword result 1",
                        score=0.8,
                        document_source="doc2.pdf",
                        metadata={}
                    )
                ]
            )
            mock.return_value = mock_instance
            yield mock

    def test_hybrid_retriever_initialization(self):
        """Test HybridRetriever can be initialized."""
        retriever = HybridRetriever()
        assert retriever is not None

    @pytest.mark.asyncio
    async def test_retrieve_combines_results(self, mock_vector_store, mock_keyword_search):
        """Test that hybrid retrieval combines vector and keyword results."""
        retriever = HybridRetriever()
        
        results = await retriever.retrieve(
            query="test query",
            top_k=5,
            vector_weight=0.7,
            keyword_weight=0.3
        )
        
        assert results is not None
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_retrieve_with_filters(self, mock_vector_store, mock_keyword_search):
        """Test retrieval with metadata filters."""
        retriever = HybridRetriever()
        
        results = await retriever.retrieve(
            query="test query",
            top_k=5,
            filters={"source": "test.pdf"}
        )
        
        assert results is not None


class TestRetrievalResult:
    """Test suite for RetrievalResult model."""

    def test_retrieval_result_creation(self):
        """Test creating a RetrievalResult."""
        result = RetrievalResult(
            chunk_id="test_chunk",
            content="Test content",
            score=0.95,
            document_source="test.pdf",
            metadata={"key": "value"}
        )
        
        assert result.chunk_id == "test_chunk"
        assert result.content == "Test content"
        assert result.score == 0.95
        assert result.document_source == "test.pdf"
        assert result.metadata["key"] == "value"

    def test_retrieval_result_score_ordering(self):
        """Test that results can be sorted by score."""
        results = [
            RetrievalResult(chunk_id="c1", content="content1", score=0.5, document_source="d1", metadata={}),
            RetrievalResult(chunk_id="c2", content="content2", score=0.9, document_source="d2", metadata={}),
            RetrievalResult(chunk_id="c3", content="content3", score=0.7, document_source="d3", metadata={}),
        ]
        
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        
        assert sorted_results[0].chunk_id == "c2"
        assert sorted_results[1].chunk_id == "c3"
        assert sorted_results[2].chunk_id == "c1"
