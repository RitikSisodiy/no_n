import logging
from typing import List, Optional, Dict, Any

from llama_index.core.llms import LLM
from llama_index.core.schema import NodeWithScore
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.postprocessor import SimilarityPostprocessor

from app.core.retrievers import VectorStore

logger = logging.getLogger(__name__)

class RetrieverAgent:
    """Agent responsible for retrieving relevant document chunks and generating responses."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm: LLM,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize the retriever agent.
        
        Args:
            vector_store: Vector store instance
            llm: Language model instance
            top_k: Number of top results to retrieve
            similarity_threshold: Minimum similarity score threshold
        """
        self.vector_store = vector_store
        self.llm = llm
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        # Initialize response synthesizer
        self.response_synthesizer = get_response_synthesizer(
            llm=llm,
            response_mode="tree_summarize",
            streaming=True
        )
        
        # Initialize post-processor
        self.postprocessor = SimilarityPostprocessor(
            similarity_cutoff=similarity_threshold
        )
    
    async def retrieve(
        self,
        query: str,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[NodeWithScore]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: Search query
            metadata_filters: Optional metadata filters
            
        Returns:
            List of relevant nodes with scores
        """
        try:
            # Perform hybrid search
            nodes = await self.vector_store.hybrid_search(
                query=query,
                metadata_filters=metadata_filters,
                top_k=self.top_k * 2  # Retrieve more nodes for post-processing
            )
            
            # Post-process results
            filtered_nodes = self.postprocessor.postprocess_nodes(nodes)
            
            # Take top-k results
            final_nodes = filtered_nodes[:self.top_k]
            
            if not final_nodes:
                logger.warning(f"No relevant nodes found for query: {query}")
                return []
            
            logger.info(f"Retrieved {len(final_nodes)} relevant nodes")
            return final_nodes
            
        except Exception as e:
            logger.error(f"Error retrieving nodes: {str(e)}")
            raise
    
    async def generate_response(
        self,
        query: str,
        nodes: List[NodeWithScore]
    ) -> str:
        """
        Generate a response based on retrieved nodes.
        
        Args:
            query: Original query
            nodes: Retrieved nodes
            
        Returns:
            Generated response
        """
        try:
            if not nodes:
                return "I couldn't find any relevant information to answer your question."
            
            # Generate response using the synthesizer
            response = await self.response_synthesizer.asynthesize(
                query=query,
                nodes=nodes
            )
            
            return response.response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def _format_context(self, nodes: List[NodeWithScore]) -> str:
        """
        Format retrieved nodes into a context string.
        
        Args:
            nodes: Retrieved nodes
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, node in enumerate(nodes, 1):
            metadata = node.node.metadata
            context_parts.append(
                f"[Source {i}]\n"
                f"Document: {metadata.get('title', 'Unknown')}\n"
                f"Page: {metadata.get('page_number', 'N/A')}\n"
                f"Content: {node.node.text}\n"
            )
        return "\n".join(context_parts) 