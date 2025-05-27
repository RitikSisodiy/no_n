import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)

class VectorStore:
    """Handles vector storage and retrieval using Qdrant."""
    
    def __init__(self):
        """Initialize the vector store with Qdrant client."""
        self.client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333))
        )
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "agentic_rag")
        self._ensure_collection()
        
        # Initialize storage context and index
        self.storage_context = StorageContext.from_defaults(
            vector_store=QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name
            )
        )
        self.index = VectorStoreIndex.from_vector_store(
            self.storage_context.vector_store
        )
    
    def _ensure_collection(self) -> None:
        """Ensure the Qdrant collection exists with proper configuration."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=1536,  # OpenAI embedding dimension
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring collection: {str(e)}")
            raise
    
    async def add_nodes(self, nodes: List[TextNode]) -> None:
        """
        Add nodes to the vector store.
        
        Args:
            nodes: List of nodes to add
        """
        try:
            self.index.insert_nodes(nodes)
            logger.info(f"Added {len(nodes)} nodes to vector store")
        except Exception as e:
            logger.error(f"Error adding nodes to vector store: {str(e)}")
            raise
    
    async def delete_document(self, document_id: str) -> None:
        """
        Delete all nodes associated with a document.
        
        Args:
            document_id: ID of the document to delete
        """
        try:
            # Delete points with matching document_id in metadata
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id)
                            )
                        ]
                    )
                )
            )
            logger.info(f"Deleted document {document_id} from vector store")
        except Exception as e:
            logger.error(f"Error deleting document from vector store: {str(e)}")
            raise
    
    async def search(
        self,
        query: str,
        metadata_filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[NodeWithScore]:
        """
        Search for relevant nodes using hybrid search.
        
        Args:
            query: Search query
            metadata_filters: Optional metadata filters
            top_k: Number of results to return
            
        Returns:
            List of nodes with relevance scores
        """
        try:
            # Create query bundle
            query_bundle = QueryBundle(query_str=query)
            
            # Build filter if metadata filters provided
            filter_conditions = []
            if metadata_filters:
                for key, value in metadata_filters.items():
                    if isinstance(value, (list, tuple)):
                        filter_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchAny(any=value)
                            )
                        )
                    else:
                        filter_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value)
                            )
                        )
            
            # Perform search
            if filter_conditions:
                retriever = self.index.as_retriever(
                    filters=models.Filter(must=filter_conditions),
                    similarity_top_k=top_k
                )
            else:
                retriever = self.index.as_retriever(
                    similarity_top_k=top_k
                )
            
            nodes = await retriever.aretrieve(query_bundle)
            logger.info(f"Retrieved {len(nodes)} nodes for query: {query}")
            return nodes
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise
    
    async def hybrid_search(
        self,
        query: str,
        metadata_filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        alpha: float = 0.5
    ) -> List[NodeWithScore]:
        """
        Perform hybrid search combining semantic and keyword search.
        
        Args:
            query: Search query
            metadata_filters: Optional metadata filters
            top_k: Number of results to return
            alpha: Weight for semantic vs keyword search (0-1)
            
        Returns:
            List of nodes with relevance scores
        """
        try:
            # TODO: Implement hybrid search using both semantic and keyword search
            # For now, fall back to semantic search
            return await self.search(query, metadata_filters, top_k)
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            raise 