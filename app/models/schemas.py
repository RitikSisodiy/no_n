from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    TXT = "txt"

class DocumentMetadata(BaseModel):
    """Metadata for uploaded documents."""
    title: str
    author: Optional[str] = None
    date: Optional[datetime] = None
    document_type: DocumentType
    tags: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)

class SourceType(str, Enum):
    DOCUMENT = "document"
    WEB = "web"

class DocumentSource(BaseModel):
    """Source attribution for document-based answers."""
    type: SourceType = SourceType.DOCUMENT
    document_id: str
    document_name: str
    page_number: int
    chunk_id: str
    heading: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)

class WebSource(BaseModel):
    """Source attribution for web-based answers."""
    type: SourceType = SourceType.WEB
    url: str
    title: str
    snippet: str
    retrieved_at: datetime
    confidence: float = Field(ge=0.0, le=1.0)

class Source(BaseModel):
    """Union type for all source types."""
    document: Optional[DocumentSource] = None
    web: Optional[WebSource] = None

class QueryRequest(BaseModel):
    """Query request model."""
    query: str = Field(..., min_length=1, description="The natural language query")
    metadata_filters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional metadata filters for the query"
    )
    search_web: bool = Field(
        default=False,
        description="Whether to search the web if no document matches are found"
    )

class QueryResponse(BaseModel):
    """Query response model with answer and source attribution."""
    answer: str = Field(..., description="The generated answer to the query")
    sources: List[Source] = Field(
        default_factory=list,
        description="List of sources used to generate the answer"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence score for the answer"
    )
    processing_time: float = Field(
        ...,
        description="Time taken to process the query in seconds"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the query processing"
    )

class DocumentStatus(BaseModel):
    """Status information for a processed document."""
    document_id: str
    title: str
    status: str
    processed_at: datetime
    metadata: DocumentMetadata
    chunk_count: int
    error: Optional[str] = None

class DocumentList(BaseModel):
    """List of processed documents with their status."""
    documents: List[DocumentStatus]
    total_count: int
    page: int
    page_size: int

class FeedbackRequest(BaseModel):
    """Feedback request model."""
    document_id: str = Field(..., description="ID of the document")
    query: str = Field(..., description="Original query")
    response: str = Field(..., description="Response that was provided")
    rating: int = Field(
        ...,
        ge=1,
        le=5,
        description="Rating from 1 to 5"
    )
    feedback: Optional[str] = Field(
        None,
        description="Optional feedback text"
    )

class FeedbackAnalytics(BaseModel):
    """Feedback analytics model."""
    total_feedback: int = Field(..., description="Total number of feedback entries")
    average_rating: float = Field(..., ge=0.0, le=5.0, description="Average rating")
    rating_distribution: Dict[int, int] = Field(
        ...,
        description="Distribution of ratings"
    )
    common_feedback: List[Dict[str, Any]] = Field(
        ...,
        description="List of common feedback themes"
    )
    document_stats: Optional[Dict[str, Any]] = Field(
        None,
        description="Document-specific statistics"
    )
    time_series: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Time series data of feedback"
    ) 