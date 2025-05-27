import logging
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlite3 import Connection
import json
import hashlib

logger = logging.getLogger(__name__)

class DocumentVersion:
    """Model for document versions."""
    
    def __init__(
        self,
        version_id: str,
        document_id: str,
        version_number: int,
        content_hash: str,
        metadata: Dict[str, Any],
        created_at: datetime,
        created_by: str,
        changes: Optional[Dict[str, Any]] = None
    ):
        self.version_id = version_id
        self.document_id = document_id
        self.version_number = version_number
        self.content_hash = content_hash
        self.metadata = metadata
        self.created_at = created_at
        self.created_by = created_by
        self.changes = changes or {}

class VersioningService:
    """Service for handling document versioning."""
    
    def __init__(self, storage: 'DocumentStorage'):
        """Initialize the versioning service."""
        self.storage = storage
    
    def _init_db(self, conn: Connection) -> None:
        """Initialize versioning tables."""
        cursor = conn.cursor()
        
        # Document versions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_versions (
                version_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                created_by TEXT NOT NULL,
                changes TEXT,
                FOREIGN KEY (document_id) REFERENCES document_status(document_id),
                FOREIGN KEY (created_by) REFERENCES users(id),
                UNIQUE (document_id, version_number)
            )
        """)
        
        # Version tags table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS version_tags (
                tag_id TEXT PRIMARY KEY,
                version_id TEXT NOT NULL,
                tag_name TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP NOT NULL,
                created_by TEXT NOT NULL,
                FOREIGN KEY (version_id) REFERENCES document_versions(version_id),
                FOREIGN KEY (created_by) REFERENCES users(id),
                UNIQUE (version_id, tag_name)
            )
        """)
        
        conn.commit()
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate SHA-256 hash of document content."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def create_version(
        self,
        document_id: str,
        content: str,
        metadata: Dict[str, Any],
        user_id: str,
        changes: Optional[Dict[str, Any]] = None
    ) -> DocumentVersion:
        """Create a new document version."""
        try:
            with self.storage._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get current version number
                cursor.execute("""
                    SELECT MAX(version_number)
                    FROM document_versions
                    WHERE document_id = ?
                """, (document_id,))
                row = cursor.fetchone()
                version_number = (row[0] or 0) + 1
                
                # Create version
                version_id = str(uuid.uuid4())
                content_hash = self._calculate_content_hash(content)
                now = datetime.utcnow()
                
                cursor.execute("""
                    INSERT INTO document_versions (
                        version_id, document_id, version_number,
                        content_hash, metadata, created_at,
                        created_by, changes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    version_id,
                    document_id,
                    version_number,
                    content_hash,
                    json.dumps(metadata),
                    now.isoformat(),
                    user_id,
                    json.dumps(changes) if changes else None
                ))
                
                conn.commit()
                
                return DocumentVersion(
                    version_id=version_id,
                    document_id=document_id,
                    version_number=version_number,
                    content_hash=content_hash,
                    metadata=metadata,
                    created_at=now,
                    created_by=user_id,
                    changes=changes
                )
                
        except Exception as e:
            logger.error(f"Error creating version: {str(e)}")
            raise
    
    async def get_version(
        self,
        version_id: str
    ) -> Optional[DocumentVersion]:
        """Get a specific version."""
        try:
            with self.storage._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM document_versions
                    WHERE version_id = ?
                """, (version_id,))
                row = cursor.fetchone()
                
                if row:
                    return DocumentVersion(
                        version_id=row[0],
                        document_id=row[1],
                        version_number=row[2],
                        content_hash=row[3],
                        metadata=json.loads(row[4]),
                        created_at=datetime.fromisoformat(row[5]),
                        created_by=row[6],
                        changes=json.loads(row[7]) if row[7] else None
                    )
                return None
                
        except Exception as e:
            logger.error(f"Error getting version: {str(e)}")
            raise
    
    async def list_versions(
        self,
        document_id: str,
        limit: int = 10,
        offset: int = 0
    ) -> List[DocumentVersion]:
        """List versions of a document."""
        try:
            with self.storage._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM document_versions
                    WHERE document_id = ?
                    ORDER BY version_number DESC
                    LIMIT ? OFFSET ?
                """, (document_id, limit, offset))
                rows = cursor.fetchall()
                
                return [
                    DocumentVersion(
                        version_id=row[0],
                        document_id=row[1],
                        version_number=row[2],
                        content_hash=row[3],
                        metadata=json.loads(row[4]),
                        created_at=datetime.fromisoformat(row[5]),
                        created_by=row[6],
                        changes=json.loads(row[7]) if row[7] else None
                    )
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Error listing versions: {str(e)}")
            raise
    
    async def add_version_tag(
        self,
        version_id: str,
        tag_name: str,
        description: Optional[str],
        user_id: str
    ) -> bool:
        """Add a tag to a version."""
        try:
            with self.storage._get_connection() as conn:
                cursor = conn.cursor()
                
                tag_id = str(uuid.uuid4())
                now = datetime.utcnow()
                
                cursor.execute("""
                    INSERT INTO version_tags (
                        tag_id, version_id, tag_name,
                        description, created_at, created_by
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    tag_id,
                    version_id,
                    tag_name,
                    description,
                    now.isoformat(),
                    user_id
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error adding version tag: {str(e)}")
            raise
    
    async def get_version_by_tag(
        self,
        document_id: str,
        tag_name: str
    ) -> Optional[DocumentVersion]:
        """Get a version by its tag."""
        try:
            with self.storage._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT v.*
                    FROM document_versions v
                    JOIN version_tags t ON v.version_id = t.version_id
                    WHERE v.document_id = ? AND t.tag_name = ?
                """, (document_id, tag_name))
                row = cursor.fetchone()
                
                if row:
                    return DocumentVersion(
                        version_id=row[0],
                        document_id=row[1],
                        version_number=row[2],
                        content_hash=row[3],
                        metadata=json.loads(row[4]),
                        created_at=datetime.fromisoformat(row[5]),
                        created_by=row[6],
                        changes=json.loads(row[7]) if row[7] else None
                    )
                return None
                
        except Exception as e:
            logger.error(f"Error getting version by tag: {str(e)}")
            raise
    
    async def compare_versions(
        self,
        version_id1: str,
        version_id2: str
    ) -> Dict[str, Any]:
        """Compare two versions of a document."""
        try:
            version1 = await self.get_version(version_id1)
            version2 = await self.get_version(version_id2)
            
            if not version1 or not version2:
                raise ValueError("One or both versions not found")
            
            if version1.document_id != version2.document_id:
                raise ValueError("Versions must be from the same document")
            
            # Compare metadata
            metadata_diff = {
                key: {
                    "old": version1.metadata.get(key),
                    "new": version2.metadata.get(key)
                }
                for key in set(version1.metadata) | set(version2.metadata)
                if version1.metadata.get(key) != version2.metadata.get(key)
            }
            
            # Compare content hashes
            content_changed = version1.content_hash != version2.content_hash
            
            return {
                "version1": {
                    "version_id": version1.version_id,
                    "version_number": version1.version_number,
                    "created_at": version1.created_at.isoformat(),
                    "created_by": version1.created_by
                },
                "version2": {
                    "version_id": version2.version_id,
                    "version_number": version2.version_number,
                    "created_at": version2.created_at.isoformat(),
                    "created_by": version2.created_by
                },
                "metadata_diff": metadata_diff,
                "content_changed": content_changed,
                "changes": version2.changes
            }
            
        except Exception as e:
            logger.error(f"Error comparing versions: {str(e)}")
            raise
    
    async def rollback_to_version(
        self,
        version_id: str,
        user_id: str
    ) -> DocumentVersion:
        """Rollback document to a specific version."""
        try:
            version = await self.get_version(version_id)
            if not version:
                raise ValueError("Version not found")
            
            # Create new version with rollback metadata
            return await self.create_version(
                document_id=version.document_id,
                content=version.content_hash,  # This should be the actual content
                metadata=version.metadata,
                user_id=user_id,
                changes={
                    "type": "rollback",
                    "from_version": version.version_number,
                    "reason": "Manual rollback"
                }
            )
            
        except Exception as e:
            logger.error(f"Error rolling back version: {str(e)}")
            raise 