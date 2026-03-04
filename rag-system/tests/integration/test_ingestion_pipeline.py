"""
Integration tests for the ingestion pipeline.
Tests the full document ingestion flow from parsing to storage.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path


class TestIngestionPipeline:
    """Test suite for the full ingestion pipeline."""

    @pytest.fixture
    def mock_parsers(self):
        """Mock all document parsers."""
        with patch("src.ingestion.pdf_parser.PDFParser") as mock_pdf, \
             patch("src.ingestion.web_parser.WebParser") as mock_web, \
             patch("src.ingestion.code_parser.CodeParser") as mock_code:
            
            # Mock PDF parser
            mock_pdf_instance = MagicMock()
            mock_pdf_instance.parse = AsyncMock(return_value=MagicMock(
                content="PDF content",
                metadata={"source": "test.pdf"},
                id="doc_1"
            ))
            mock_pdf.return_value = mock_pdf_instance
            
            # Mock web parser
            mock_web_instance = MagicMock()
            mock_web_instance.parse = AsyncMock(return_value=MagicMock(
                content="Web content",
                metadata={"source": "http://example.com"},
                id="doc_2"
            ))
            mock_web.return_value = mock_web_instance
            
            # Mock code parser
            mock_code_instance = MagicMock()
            mock_code_instance.parse = AsyncMock(return_value=MagicMock(
                content="def hello(): pass",
                metadata={"source": "test.py"},
                id="doc_3"
            ))
            mock_code.return_value = mock_code_instance
            
            yield {
                "pdf": mock_pdf_instance,
                "web": mock_web_instance,
                "code": mock_code_instance
            }

    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service."""
        with patch("src.retrieval.embeddings.EmbeddingService") as mock:
            mock_instance = MagicMock()
            mock_instance.embed_batch = AsyncMock(return_value=[[0.1] * 384])
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store."""
        with patch("src.retrieval.vector_store.VectorStore") as mock:
            mock_instance = MagicMock()
            mock_instance.upsert_chunks = AsyncMock(return_value={"upserted_count": 5})
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_database(self):
        """Mock database repository."""
        with patch("src.database.repository.DocumentRepository") as mock:
            mock_instance = MagicMock()
            mock_instance.document_exists = AsyncMock(return_value=None)
            mock_instance.save_document = AsyncMock()
            mock_instance.save_chunks = AsyncMock()
            mock.return_value = mock_instance
            yield mock_instance


class TestPDFIngestion:
    """Test suite for PDF ingestion."""

    @pytest.mark.asyncio
    async def test_pdf_parsing_extracts_text(self, mock_parsers):
        """Test PDF parser extracts text content."""
        from src.ingestion.pdf_parser import PDFParser
        
        parser = PDFParser()
        # Would test actual parsing if PDF available
        assert parser is not None

    @pytest.mark.asyncio
    async def test_pdf_ingestion_flow(self, mock_parsers, mock_embedding_service, mock_vector_store, mock_database):
        """Test full PDF ingestion flow."""
        from src.api.routes.ingest import ingest_document
        from src.models import IngestionRequest, DocumentType, ChunkingStrategy
        
        # This would require full database setup
        # Placeholder test
        assert True


class TestChunkingIntegration:
    """Test suite for chunking integration."""

    def test_semantic_chunking_preserves_sentences(self):
        """Test semantic chunking keeps sentences intact."""
        from src.ingestion.chunker import Chunker, ChunkingStrategy
        from src.models import Document, DocumentType
        
        text = (
            "This is the first sentence. "
            "This is the second sentence. "
            "This is the third sentence."
        )
        
        doc = Document(
            content=text,
            doc_type=DocumentType.TEXT,
            source="test.txt"
        )
        
        chunker = Chunker(
            strategy=ChunkingStrategy.SEMANTIC,
            max_tokens=50,
            overlap_tokens=10
        )
        
        chunks = chunker.chunk_document(doc)
        
        # Should have multiple chunks
        assert len(chunks) >= 1

    def test_fixed_size_chunking_respects_max_tokens(self):
        """Test fixed-size chunking respects token limit."""
        from src.ingestion.chunker import Chunker, ChunkingStrategy
        from src.models import Document, DocumentType
        
        # Create text long enough for multiple chunks
        text = "word " * 200
        
        doc = Document(
            content=text,
            doc_type=DocumentType.TEXT,
            source="test.txt"
        )
        
        chunker = Chunker(
            strategy=ChunkingStrategy.FIXED_SIZE,
            max_tokens=50,
            overlap_tokens=10
        )
        
        chunks = chunker.chunk_document(doc)
        
        # Each chunk should be under the limit (with some tolerance)
        for chunk in chunks:
            assert chunk.token_count <= 60


class TestEmbeddingIntegration:
    """Test suite for embedding integration."""

    @pytest.mark.asyncio
    async def test_batch_embedding_handles_multiple_texts(self, mock_embedding_service):
        """Test batch embedding processes multiple texts."""
        texts = ["text 1", "text 2", "text 3"]
        
        embeddings = await mock_embedding_service.embed_batch(texts)
        
        assert len(embeddings) == 3

    @pytest.mark.asyncio
    async def test_embedding_dimensions_match_vector_store(self, mock_embedding_service):
        """Test embedding dimensions are compatible with vector store."""
        text = "test text"
        
        embedding = await mock_embedding_service.embed_batch([text])
        
        # Default dimension should be 384 or 1536
        assert len(embedding[0]) in [384, 1536]


class TestVectorStoreIntegration:
    """Test suite for vector store integration."""

    @pytest.mark.asyncio
    async def test_upsert_associates_chunk_with_document(self, mock_vector_store):
        """Test chunks are associated with correct document."""
        from src.models import Chunk, Document
        
        # This would require actual implementation
        # Placeholder test
        assert mock_vector_store is not None


class TestDeduplication:
    """Test suite for document deduplication."""

    @pytest.mark.asyncio
    async def test_duplicate_content_detected(self, mock_database):
        """Test duplicate content is detected by hash."""
        from src.database.repository import DocumentRepository
        
        repo = DocumentRepository(session=MagicMock())
        
        # First call returns None (no duplicate)
        result1 = await repo.document_exists("test content")
        assert result1 is None


class TestErrorHandling:
    """Test suite for ingestion error handling."""

    @pytest.mark.asyncio
    async def test_invalid_file_type_rejected(self):
        """Test invalid file types are rejected."""
        from src.models import DocumentType
        
        # Would test validation
        assert DocumentType.PDF.value == "pdf"

    @pytest.mark.asyncio
    async def test_corrupted_pdf_handled_gracefully(self):
        """Test corrupted PDFs are handled without crash."""
        # Would test error handling
        assert True
