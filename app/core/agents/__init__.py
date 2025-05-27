import os
import logging
import time
from typing import List, Optional, Dict, Any
from datetime import datetime

from llama_index.core import ServiceContext
from llama_index.core.llms import LLM
from llama_index.core.schema import NodeWithScore
from llama_index.llms.openai import OpenAI
from langchain.agents import Tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool

from app.core.retrievers import VectorStore
from app.models.schemas import QueryRequest, QueryResponse, Source, DocumentSource, WebSource
from app.core.agents.retriever import RetrieverAgent
from app.core.agents.metadata import MetadataAgent
from app.core.agents.reflection import ReflectionAgent
from app.core.agents.web_search import WebSearchAgent

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Orchestrates multiple agents to process queries and generate responses."""

    def __init__(self):
        """Initialize the query processor with all necessary agents."""
        # Initialize LLM
        self.llm = self._initialize_llm()

        # Initialize vector store
        self.vector_store = VectorStore()

        # Initialize agents
        self.retriever_agent = RetrieverAgent(
            vector_store=self.vector_store,
            llm=self.llm
        )
        self.metadata_agent = MetadataAgent(llm=self.llm)
        self.reflection_agent = ReflectionAgent(llm=self.llm)
        self.web_search_agent = WebSearchAgent(llm=self.llm)

        # Initialize service context
        self.service_context = ServiceContext.from_defaults(llm=self.llm)

    def _initialize_llm(self) -> LLM:
        """Initialize the language model."""
        try:
            # Use environment variables for configuration
            model_name = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")
            api_base = os.getenv("LLM_API_BASE")
            api_key = os.getenv("LLM_API_KEY")

            return OpenAI(
                model=model_name,
                api_base=api_base,
                api_key=api_key,
                temperature=0.1,  # Low temperature for more focused responses
                streaming=True
            )
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise

    async def process_query(
            self,
            query: str,
            metadata_filters: Optional[Dict[str, Any]] = None,
            search_web: bool = False
    ) -> QueryResponse:
        """
        Process a query using the agent orchestration system.

        Args:
            query: The natural language query
            metadata_filters: Optional metadata filters
            search_web: Whether to search the web if no document matches

        Returns:
            QueryResponse with answer and source attribution
        """
        start_time = time.time()
        try:
            # 1. Metadata Agent: Process metadata filters
            if metadata_filters:
                metadata_filters = await self.metadata_agent.process_filters(
                    query,
                    metadata_filters
                )

            # 2. Retriever Agent: Get relevant document chunks
            nodes = await self.retriever_agent.retrieve(
                query,
                metadata_filters
            )

            # 3. If no relevant documents found and web search enabled
            if not nodes and search_web:
                web_results = await self.web_search_agent.search(query)
                if web_results:
                    return self._create_web_response(
                        query,
                        web_results,
                        time.time() - start_time
                    )

            # 4. Generate initial response
            initial_response = await self.retriever_agent.generate_response(
                query,
                nodes
            )

            # 5. Reflection Agent: Evaluate and improve response
            final_response = await self.reflection_agent.evaluate_response(
                query,
                initial_response,
                nodes
            )

            # 6. Create response with source attribution
            return self._create_document_response(
                query,
                final_response,
                nodes,
                time.time() - start_time
            )

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    def _create_document_response(
            self,
            query: str,
            answer: str,
            nodes: List[NodeWithScore],
            processing_time: float
    ) -> QueryResponse:
        """Create a response with document source attribution."""
        sources = []
        for node in nodes:
            metadata = node.node.metadata
            source = DocumentSource(
                document_id=metadata["document_id"],
                document_name=metadata["title"],
                page_number=metadata.get("page_number", 1),
                chunk_id=metadata.get("chunk_id", ""),
                heading=metadata.get("heading"),
                confidence=float(node.score) if node.score is not None else 0.0
            )
            sources.append(Source(document=source))

        return QueryResponse(
            answer=answer,
            sources=sources,
            confidence=float(nodes[0].score) if nodes and nodes[0].score is not None else 0.0,
            processing_time=processing_time,
            metadata={
                "query": query,
                "source_type": "document",
                "node_count": len(nodes)
            }
        )

    def _create_web_response(
            self,
            query: str,
            web_results: List[Dict[str, Any]],
            processing_time: float
    ) -> QueryResponse:
        """Create a response with web source attribution."""
        sources = []
        for result in web_results:
            source = WebSource(
                url=result["url"],
                title=result["title"],
                snippet=result["snippet"],
                retrieved_at=datetime.utcnow(),
                confidence=result.get("confidence", 0.0)
            )
            sources.append(Source(web=source))

        return QueryResponse(
            answer=web_results[0]["answer"],
            sources=sources,
            confidence=web_results[0].get("confidence", 0.0),
            processing_time=processing_time,
            metadata={
                "query": query,
                "source_type": "web",
                "result_count": len(web_results)
            }
        )