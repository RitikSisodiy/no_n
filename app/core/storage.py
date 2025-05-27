import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from app.models.schemas import DocumentStatus, DocumentMetadata

logger = logging.getLogger(__name__)

class DocumentStorage:
    """SQLite-based storage for document status and audit logs."""
    
    def __init__(self, db_path: str = "data/documents.db"):
        """Initialize the storage system."""
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the database with required tables."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Document status table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_status (
                    document_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    processed_at TIMESTAMP NOT NULL,
                    metadata TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Document feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    rating INTEGER CHECK (rating BETWEEN 1 AND 5),
                    feedback TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES document_status(document_id)
                )
            """)
            
            # Audit logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    document_id TEXT,
                    user_id TEXT,
                    details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES document_status(document_id)
                )
            """)
            
            # Cache table for query results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_cache (
                    cache_key TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL
                )
            """)
            
            conn.commit()
    
    def save_document_status(self, status: DocumentStatus) -> None:
        """Save document processing status."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO document_status (
                        document_id, title, status, processed_at,
                        metadata, chunk_count, error, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    status.document_id,
                    status.title,
                    status.status,
                    status.processed_at.isoformat(),
                    json.dumps(status.metadata.dict()),
                    status.chunk_count,
                    status.error,
                    datetime.utcnow().isoformat()
                ))
                conn.commit()
                
                # Log the action
                self._log_audit(
                    action="save_document_status",
                    document_id=status.document_id,
                    details=f"Status: {status.status}, Chunks: {status.chunk_count}"
                )
                
        except Exception as e:
            logger.error(f"Error saving document status: {str(e)}")
            raise
    
    def get_document_status(self, document_id: str) -> Optional[DocumentStatus]:
        """Get document status by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM document_status WHERE document_id = ?
                """, (document_id,))
                row = cursor.fetchone()
                
                if row:
                    return DocumentStatus(
                        document_id=row[0],
                        title=row[1],
                        status=row[2],
                        processed_at=datetime.fromisoformat(row[3]),
                        metadata=DocumentMetadata(**json.loads(row[4])),
                        chunk_count=row[5],
                        error=row[6]
                    )
                return None
                
        except Exception as e:
            logger.error(f"Error getting document status: {str(e)}")
            raise
    
    def list_document_statuses(
        self,
        page: int = 1,
        page_size: int = 10
    ) -> List[DocumentStatus]:
        """List document statuses with pagination."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                offset = (page - 1) * page_size
                
                cursor.execute("""
                    SELECT * FROM document_status
                    ORDER BY processed_at DESC
                    LIMIT ? OFFSET ?
                """, (page_size, offset))
                
                return [
                    DocumentStatus(
                        document_id=row[0],
                        title=row[1],
                        status=row[2],
                        processed_at=datetime.fromisoformat(row[3]),
                        metadata=DocumentMetadata(**json.loads(row[4])),
                        chunk_count=row[5],
                        error=row[6]
                    )
                    for row in cursor.fetchall()
                ]
                
        except Exception as e:
            logger.error(f"Error listing document statuses: {str(e)}")
            raise
    
    def delete_document_status(self, document_id: str) -> None:
        """Delete document status."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM document_status WHERE document_id = ?",
                    (document_id,)
                )
                conn.commit()
                
                # Log the action
                self._log_audit(
                    action="delete_document_status",
                    document_id=document_id
                )
                
        except Exception as e:
            logger.error(f"Error deleting document status: {str(e)}")
            raise
    
    def save_feedback(
        self,
        document_id: str,
        query: str,
        response: str,
        rating: int,
        feedback: Optional[str] = None
    ) -> None:
        """Save user feedback for a document response."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO document_feedback (
                        document_id, query, response, rating, feedback
                    ) VALUES (?, ?, ?, ?, ?)
                """, (document_id, query, response, rating, feedback))
                conn.commit()
                
                # Log the action
                self._log_audit(
                    action="save_feedback",
                    document_id=document_id,
                    details=f"Rating: {rating}"
                )
                
        except Exception as e:
            logger.error(f"Error saving feedback: {str(e)}")
            raise
    
    def cache_query_result(
        self,
        cache_key: str,
        query: str,
        response: Dict[str, Any],
        ttl_seconds: int = 3600
    ) -> None:
        """Cache a query result."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                expires_at = datetime.utcnow().timestamp() + ttl_seconds
                
                cursor.execute("""
                    INSERT OR REPLACE INTO query_cache (
                        cache_key, query, response, metadata, expires_at
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    cache_key,
                    query,
                    json.dumps(response),
                    json.dumps({"ttl": ttl_seconds}),
                    datetime.fromtimestamp(expires_at).isoformat()
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error caching query result: {str(e)}")
            raise
    
    def get_cached_result(
        self,
        cache_key: str
    ) -> Optional[Dict[str, Any]]:
        """Get a cached query result."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT response FROM query_cache
                    WHERE cache_key = ? AND expires_at > ?
                """, (cache_key, datetime.utcnow().isoformat()))
                
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
                return None
                
        except Exception as e:
            logger.error(f"Error getting cached result: {str(e)}")
            return None
    
    def _log_audit(
        self,
        action: str,
        document_id: Optional[str] = None,
        user_id: Optional[str] = None,
        details: Optional[str] = None
    ) -> None:
        """Log an audit event."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO audit_logs (
                        action, document_id, user_id, details
                    ) VALUES (?, ?, ?, ?)
                """, (action, document_id, user_id, details))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error logging audit event: {str(e)}")
            # Don't raise the exception to avoid disrupting the main operation 