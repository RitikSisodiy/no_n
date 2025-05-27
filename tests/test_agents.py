import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from app.core.agents import QueryProcessor
from app.core.agents.retriever import RetrieverAgent
from app.core.agents.metadata import MetadataAgent
from app.core.agents.reflection import ReflectionAgent
from app.core.agents.web_search import WebSearchAgent
from app.models.schemas import DocumentMetadata, DocumentType

@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = Mock()
    llm.complete.return_value = Mock(text='{"answer": "Test response"}')
    return llm

@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing."""
    store = Mock()
    store.search.return_value = []
    store.hybrid_search.return_value = []
    return store

@pytest.fixture
def query_processor(mock_llm, mock_vector_store):
    """Create a QueryProcessor instance with mocked dependencies."""
    with patch('app.core.agents.VectorStore', return_value=mock_vector_store):
        processor = QueryProcessor()
        processor.llm = mock_llm
        processor.vector_store = mock_vector_store
        return processor

@pytest.mark.asyncio
async def test_query_processing(query_processor):
    """Test basic query processing."""
    # Test query
    query = "What is the test answer?"
    
    # Process query
    response = await query_processor.process_query(
        query=query,
        metadata_filters={"document_type": "pdf"},
        search_web=False
    )
    
    # Verify response structure
    assert response.answer is not None
    assert isinstance(response.sources, list)
    assert isinstance(response.confidence, float)
    assert isinstance(response.processing_time, float)
    assert isinstance(response.metadata, dict)

@pytest.mark.asyncio
async def test_metadata_agent(mock_llm):
    """Test metadata agent processing."""
    agent = MetadataAgent(mock_llm)
    
    # Test metadata processing
    filters = {
        "author": "Test Author",
        "date": "2024-02-20",
        "document_type": "pdf"
    }
    
    processed = await agent.process_filters(
        query="Test query",
        filters=filters
    )
    
    assert isinstance(processed, dict)
    assert "document_type" in processed

@pytest.mark.asyncio
async def test_reflection_agent(mock_llm):
    """Test reflection agent evaluation."""
    agent = ReflectionAgent(mock_llm)
    
    # Test response evaluation
    query = "Test query"
    response = "Test response"
    nodes = []
    
    improved = await agent.evaluate_response(
        query=query,
        response=response,
        nodes=nodes
    )
    
    assert isinstance(improved, str)
    assert len(improved) > 0

@pytest.mark.asyncio
async def test_web_search_agent(mock_llm):
    """Test web search agent."""
    agent = WebSearchAgent(mock_llm)
    
    # Test web search
    results = await agent.search("Test query")
    
    assert isinstance(results, list)
    if results:
        assert isinstance(results[0], dict)
        assert "summary" in results[0]
        assert "sources" in results[0]

@pytest.mark.asyncio
async def test_retriever_agent(mock_llm, mock_vector_store):
    """Test retriever agent."""
    agent = RetrieverAgent(
        vector_store=mock_vector_store,
        llm=mock_llm
    )
    
    # Test retrieval
    nodes = await agent.retrieve(
        query="Test query",
        metadata_filters={"document_type": "pdf"}
    )
    
    assert isinstance(nodes, list)
    
    # Test response generation
    response = await agent.generate_response(
        query="Test query",
        nodes=nodes
    )
    
    assert isinstance(response, str)
    assert len(response) > 0

def test_document_metadata():
    """Test document metadata model."""
    metadata = DocumentMetadata(
        title="Test Document",
        author="Test Author",
        date=datetime.utcnow(),
        document_type=DocumentType.PDF,
        tags=["test", "example"],
        description="Test document description"
    )
    
    assert metadata.title == "Test Document"
    assert metadata.author == "Test Author"
    assert metadata.document_type == DocumentType.PDF
    assert len(metadata.tags) == 2
    assert metadata.description == "Test document description" 