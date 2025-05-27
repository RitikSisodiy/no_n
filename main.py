from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import os
from datetime import datetime, timedelta
import json

from app.core.parsers import DocumentParser
from app.core.agents import QueryProcessor
from app.core.storage import DocumentStorage
from app.models.schemas import QueryRequest, QueryResponse, DocumentMetadata, FeedbackRequest, DocumentStatus, FeedbackAnalytics
from app.core.analytics import AnalyticsService
from app.api.auth import router as auth_router
from app.core.auth import AuthService
from app.models.auth import User, UserCreate, UserUpdate, Token
from app.core.cache import CacheService
from app.core.versioning import VersioningService
from app.core.websocket import WebSocketService
from app.models.document import Document, DocumentCreate, DocumentUpdate, DocumentQuery
from app.models.analytics import AnalyticsResponse
from app.core.versioning import DocumentVersion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agentic RAG System",
    description="An intelligent RAG system with multi-agent orchestration and source attribution",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core components
document_parser = DocumentParser()
query_processor = QueryProcessor()
document_storage = DocumentStorage()
analytics_service = AnalyticsService(document_storage)
cache_service = CacheService()
versioning_service = VersioningService(document_storage)
websocket_service = WebSocketService()

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Include routers
app.include_router(auth_router)

# Dependency for getting current user
async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get the current authenticated user."""
    try:
        user = await auth_service.get_user_from_token(token)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user
    except Exception as e:
        logger.error(f"Error getting current user: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "agentic-rag"}

@app.post("/documents/upload")
async def upload_document(
    document: DocumentCreate,
    user: User = Depends(get_current_user)
):
    """Upload a document with versioning."""
    try:
        # Upload document
        doc = await document_storage.upload_document(document)
        
        # Create initial version
        version = await versioning_service.create_version(
            document_id=doc.id,
            content=document.content,
            metadata=document.metadata,
            user_id=user.id,
            changes={"type": "initial_version"}
        )
        
        # Cache document metadata
        await cache_service.cache_document_metadata(doc.id, doc.dict())
        
        # Notify subscribers about the new document
        await websocket_service.notify_system_update(
            update_type="document_uploaded",
            data={
                "document_id": doc.id,
                "title": doc.title,
                "uploaded_by": user.id,
                "version_id": version.version_id
            }
        )
        
        return doc
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def process_query(
    query_request: QueryRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Process a natural language query with optional metadata filters.
    
    Args:
        query_request: Query request containing the question and optional filters
        current_user: The current authenticated user
    
    Returns:
        Query response with answer and source attribution
    """
    try:
        response = await query_processor.process_query(
            query=query_request.query,
            metadata_filters=query_request.metadata_filters,
            search_web=query_request.search_web
        )
        return response
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=list[DocumentStatus])
async def list_documents(
    current_user: User = Depends(get_current_user),
    skip: int = 0,
    limit: int = 10
):
    """List all processed documents with their metadata."""
    try:
        documents = await document_parser.list_documents(skip=skip, limit=limit)
        return {"documents": documents}
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    user: User = Depends(get_current_user)
):
    """Delete a document with cache invalidation."""
    try:
        # Delete document
        success = await document_storage.delete_document(document_id)
        
        if success:
            # Invalidate cache
            await cache_service.invalidate_document_cache(document_id)
            
            # Notify subscribers about the deletion
            await websocket_service.notify_system_update(
                update_type="document_deleted",
                data={
                    "document_id": document_id,
                    "deleted_by": user.id
                }
            )
            
            return {"message": "Document deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def save_feedback(
    feedback: Dict[str, Any],
    user: User = Depends(get_current_user)
):
    """Save feedback with real-time notification."""
    try:
        # Save feedback
        result = await document_storage.save_feedback(feedback)
        
        # Notify subscribers about the feedback
        if "document_id" in feedback:
            await websocket_service.notify_document_update(
                document_id=feedback["document_id"],
                update_type="feedback_received",
                data={
                    "feedback_id": result["feedback_id"],
                    "user_id": user.id,
                    "rating": feedback.get("rating"),
                    "timestamp": datetime.utcnow().isoformat()
                },
                user_id=user.id
            )
        
        return result
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/feedback", response_model=FeedbackAnalytics)
async def get_feedback_analytics(
    current_user: User = Depends(get_current_user),
    document_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Get comprehensive analytics data.
    
    Args:
        document_id: Optional document ID to filter by
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        current_user: The current authenticated user
    
    Returns:
        Analytics data including query performance, document usage,
        user feedback, and system health metrics
    """
    try:
        analytics_data = analytics_service.get_analytics(
            document_id=document_id,
            start_date=start_date,
            end_date=end_date
        )
        return {
            "status": "success",
            "data": analytics_data
        }
    except Exception as e:
        logger.error(f"Error getting analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint (no authentication required)
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check Redis connection
        redis_health = await cache_service.check_health()
        
        # Check database connection
        db_health = await document_storage.check_health()
        
        return {
            "status": "healthy",
            "redis": redis_health,
            "database": db_health,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time updates."""
    try:
        await websocket_service.handle_connection(websocket, user_id)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user {user_id}")
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {str(e)}")

# Document versioning endpoints
@app.post("/documents/{document_id}/versions", response_model=DocumentVersion)
async def create_document_version(
    document_id: str,
    content: str,
    metadata: Dict[str, Any],
    user: User = Depends(get_current_user)
):
    """Create a new version of a document."""
    try:
        version = await versioning_service.create_version(
            document_id=document_id,
            content=content,
            metadata=metadata,
            user_id=user.id
        )
        
        # Notify subscribers about the new version
        await websocket_service.notify_document_update(
            document_id=document_id,
            update_type="version_created",
            data={
                "version_id": version.version_id,
                "version_number": version.version_number,
                "created_by": user.id
            },
            user_id=user.id
        )
        
        return version
    except Exception as e:
        logger.error(f"Error creating document version: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{document_id}/versions", response_model=List[DocumentVersion])
async def list_document_versions(
    document_id: str,
    limit: int = 10,
    offset: int = 0,
    user: User = Depends(get_current_user)
):
    """List versions of a document."""
    try:
        versions = await versioning_service.list_versions(
            document_id=document_id,
            limit=limit,
            offset=offset
        )
        return versions
    except Exception as e:
        logger.error(f"Error listing document versions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{document_id}/versions/{version_id}", response_model=DocumentVersion)
async def get_document_version(
    document_id: str,
    version_id: str,
    user: User = Depends(get_current_user)
):
    """Get a specific version of a document."""
    try:
        version = await versioning_service.get_version(version_id)
        if not version or version.document_id != document_id:
            raise HTTPException(status_code=404, detail="Version not found")
        return version
    except Exception as e:
        logger.error(f"Error getting document version: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/{document_id}/versions/{version_id}/rollback")
async def rollback_document(
    document_id: str,
    version_id: str,
    user: User = Depends(get_current_user)
):
    """Rollback a document to a specific version."""
    try:
        version = await versioning_service.rollback_to_version(
            version_id=version_id,
            user_id=user.id
        )
        
        # Notify subscribers about the rollback
        await websocket_service.notify_document_update(
            document_id=document_id,
            update_type="document_rollback",
            data={
                "version_id": version.version_id,
                "version_number": version.version_number,
                "rolled_back_by": user.id
            },
            user_id=user.id
        )
        
        return {"message": "Document rolled back successfully", "version": version}
    except Exception as e:
        logger.error(f"Error rolling back document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{document_id}/versions/compare")
async def compare_versions(
    document_id: str,
    version_id1: str,
    version_id2: str,
    user: User = Depends(get_current_user)
):
    """Compare two versions of a document."""
    try:
        comparison = await versioning_service.compare_versions(
            version_id1=version_id1,
            version_id2=version_id2
        )
        return comparison
    except Exception as e:
        logger.error(f"Error comparing versions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/{document_id}/versions/{version_id}/tags")
async def add_version_tag(
    document_id: str,
    version_id: str,
    tag_name: str,
    description: Optional[str] = None,
    user: User = Depends(get_current_user)
):
    """Add a tag to a document version."""
    try:
        success = await versioning_service.add_version_tag(
            version_id=version_id,
            tag_name=tag_name,
            description=description,
            user_id=user.id
        )
        
        if success:
            # Notify subscribers about the new tag
            await websocket_service.notify_document_update(
                document_id=document_id,
                update_type="version_tagged",
                data={
                    "version_id": version_id,
                    "tag_name": tag_name,
                    "tagged_by": user.id
                },
                user_id=user.id
            )
            
            return {"message": "Tag added successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to add tag")
    except Exception as e:
        logger.error(f"Error adding version tag: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{document_id}/versions/tags/{tag_name}", response_model=DocumentVersion)
async def get_version_by_tag(
    document_id: str,
    tag_name: str,
    user: User = Depends(get_current_user)
):
    """Get a document version by its tag."""
    try:
        version = await versioning_service.get_version_by_tag(
            document_id=document_id,
            tag_name=tag_name
        )
        if not version:
            raise HTTPException(status_code=404, detail="Version not found")
        return version
    except Exception as e:
        logger.error(f"Error getting version by tag: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Update existing endpoints to use caching and real-time notifications
@app.post("/query")
async def query_documents(
    query: DocumentQuery,
    user: User = Depends(get_current_user)
):
    """Query documents with caching."""
    try:
        # Try to get from cache first
        cache_key = f"query:{query.query}:{query.document_id if query.document_id else 'all'}"
        cached_result = await cache_service.get_query_result(cache_key)
        
        if cached_result:
            return cached_result
        
        # If not in cache, process query
        result = await document_storage.process_query(query)
        
        # Cache the result
        await cache_service.cache_query_result(cache_key, result)
        
        # Notify subscribers about the query
        if query.document_id:
            await websocket_service.notify_document_update(
                document_id=query.document_id,
                update_type="document_queried",
                data={
                    "query": query.query,
                    "user_id": user.id,
                    "timestamp": datetime.utcnow().isoformat()
                },
                user_id=user.id
            )
        
        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/feedback")
async def get_analytics(
    document_id: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    user: User = Depends(get_current_user)
) -> AnalyticsResponse:
    """Get analytics with caching."""
    try:
        # Try to get from cache first
        cache_key = f"analytics:{document_id}:{start_date}:{end_date}"
        cached_result = await cache_service.get_analytics(cache_key)
        
        if cached_result:
            return cached_result
        
        # If not in cache, get analytics
        analytics = await analytics_service.get_analytics(
            document_id=document_id,
            start_date=start_date,
            end_date=end_date
        )
        
        # Cache the result
        await cache_service.cache_analytics(cache_key, analytics)
        
        return analytics
    except Exception as e:
        logger.error(f"Error getting analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True
    ) 