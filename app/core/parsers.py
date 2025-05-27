import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from pathlib import Path

from llama_parse import LlamaParse
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.schema import NodeWithScore, TextNode

from app.models.schemas import DocumentMetadata, DocumentStatus, DocumentType
from app.core.retrievers import VectorStore
from app.core.storage import DocumentStorage

logger = logging.getLogger(__name__)

class DocumentParser:
    """Handles document parsing and ingestion into the vector store."""
    
    def __init__(self):
        """Initialize the document parser with LLAMAParse."""
        self.parser = LlamaParse(
            api_key=os.getenv("LLAMA_API_KEY"),
            result_type="markdown"  # Use markdown for better structure preservation
        )
        self.vector_store = VectorStore()
        self.storage = DocumentStorage()
        self.node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512, 128]  # Multi-level chunking
        )
    
    async def process_document(
        self,
        file_path: str,
        metadata: Optional[DocumentMetadata] = None
    ) -> str:
        """
        Process a document and store it in the vector database.
        
        Args:
            file_path: Path to the document file
            metadata: Optional metadata about the document
            
        Returns:
            str: Document ID
        """
        try:
            # Generate document ID
            doc_id = str(uuid.uuid4())
            
            # Create initial status
            if metadata is None:
                metadata = DocumentMetadata(
                    title=Path(file_path).stem,
                    document_type=self._infer_document_type(file_path)
                )
            
            status = DocumentStatus(
                document_id=doc_id,
                title=metadata.title,
                status="processing",
                processed_at=datetime.utcnow(),
                metadata=metadata,
                chunk_count=0
            )
            self.storage.save_document_status(status)
            
            try:
                # Parse document
                logger.info(f"Parsing document: {file_path}")
                parsed_doc = await self.parser.parse_file(file_path)
                
                # Create nodes with metadata
                nodes = self.node_parser.get_nodes_from_documents([parsed_doc])
                for node in nodes:
                    node.metadata.update({
                        "document_id": doc_id,
                        "title": metadata.title,
                        "author": metadata.author,
                        "date": metadata.date.isoformat() if metadata.date else None,
                        "document_type": metadata.document_type.value,
                        "tags": metadata.tags,
                        "description": metadata.description,
                        **metadata.custom_metadata
                    })
                
                # Store in vector database
                await self.vector_store.add_nodes(nodes)
                
                # Update status
                status.status = "processed"
                status.chunk_count = len(nodes)
                self.storage.save_document_status(status)
                
                logger.info(f"Successfully processed document {doc_id}")
                return doc_id
                
            except Exception as e:
                # Update status with error
                status.status = "error"
                status.error = str(e)
                self.storage.save_document_status(status)
                raise
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise
    
    async def list_documents(
        self,
        page: int = 1,
        page_size: int = 10
    ) -> Dict[str, Any]:
        """
        List all processed documents with their status.
        
        Args:
            page: Page number
            page_size: Number of documents per page
            
        Returns:
            Dict containing document list and pagination info
        """
        try:
            # Get document statuses from storage
            statuses = self.storage.list_document_statuses(page, page_size)
            
            return {
                "documents": statuses,
                "total_count": len(statuses),  # TODO: Add count query
                "page": page,
                "page_size": page_size
            }
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            raise
    
    async def delete_document(self, document_id: str) -> None:
        """
        Delete a document and its associated data.
        
        Args:
            document_id: ID of the document to delete
        """
        try:
            # Delete from vector store
            await self.vector_store.delete_document(document_id)
            
            # Delete document status
            self.storage.delete_document_status(document_id)
            
            logger.info(f"Successfully deleted document {document_id}")
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            raise
    
    def _infer_document_type(self, file_path: str) -> DocumentType:
        """Infer document type from file extension."""
        ext = Path(file_path).suffix.lower()
        type_map = {
            ".pdf": DocumentType.PDF,
            ".docx": DocumentType.DOCX,
            ".html": DocumentType.HTML,
            ".txt": DocumentType.TXT
        }
        return type_map.get(ext, DocumentType.TXT) 