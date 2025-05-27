import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from llama_index.core.llms import LLM
from llama_index.core.prompts import PromptTemplate
from serpapi import GoogleSearch
from tavily import TavilyClient

logger = logging.getLogger(__name__)

class WebSearchAgent:
    """Agent responsible for performing web searches and processing results."""
    
    def __init__(self, llm: LLM):
        """
        Initialize the web search agent.
        
        Args:
            llm: Language model instance
        """
        self.llm = llm
        
        # Initialize search clients
        self.serp_client = None
        self.tavily_client = None
        
        # Initialize API keys
        serp_api_key = os.getenv("SERPAPI_API_KEY")
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        if serp_api_key:
            self.serp_client = GoogleSearch
        if tavily_api_key:
            self.tavily_client = TavilyClient(api_key=tavily_api_key)
        
        # Initialize prompt template
        self.summary_prompt = PromptTemplate(
            """You are a web search result processor. Your task is to analyze and summarize web search results.

Query: {query}

Search Results:
{results}

Please analyze the results and:
1. Identify the most relevant and reliable information
2. Cross-reference information across sources
3. Filter out irrelevant or low-quality content
4. Generate a comprehensive summary
5. Provide source attribution

Format your response as JSON:
{{
    "summary": "Comprehensive summary of the findings...",
    "sources": [
        {{
            "url": "https://example.com",
            "title": "Source Title",
            "snippet": "Relevant snippet from the source",
            "relevance_score": 0.9,
            "reliability_score": 0.8
        }}
    ],
    "confidence": 0.85,
    "metadata": {{
        "total_sources": 5,
        "filtered_sources": 2,
        "search_time": "2024-02-20T12:00:00Z"
    }}
}}

If no relevant results are found, return:
{{
    "summary": "No relevant information found.",
    "sources": [],
    "confidence": 0.0,
    "metadata": {{
        "total_sources": 0,
        "filtered_sources": 0,
        "search_time": "2024-02-20T12:00:00Z"
    }}
}}
"""
        )
    
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform web search and process results.
        
        Args:
            query: Search query
            
        Returns:
            List of processed search results
        """
        try:
            # Perform search using available clients
            search_results = await self._perform_search(query)
            
            if not search_results:
                logger.warning(f"No web search results found for query: {query}")
                return []
            
            # Process and summarize results
            processed_results = await self._process_results(query, search_results)
            
            logger.info(f"Processed {len(processed_results)} web search results")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error performing web search: {str(e)}")
            return []
    
    async def _perform_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform search using available search clients.
        
        Args:
            query: Search query
            
        Returns:
            List of raw search results
        """
        results = []
        
        try:
            # Try Tavily first (if available)
            if self.tavily_client:
                tavily_results = await self._search_tavily(query)
                results.extend(tavily_results)
            
            # Fall back to SerpAPI if needed
            if not results and self.serp_client:
                serp_results = await self._search_serpapi(query)
                results.extend(serp_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
            return []
    
    async def _search_tavily(self, query: str) -> List[Dict[str, Any]]:
        """Search using Tavily API."""
        try:
            response = await self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=5
            )
            
            return [{
                "url": result["url"],
                "title": result["title"],
                "snippet": result["content"],
                "source": "tavily"
            } for result in response["results"]]
            
        except Exception as e:
            logger.error(f"Error in Tavily search: {str(e)}")
            return []
    
    async def _search_serpapi(self, query: str) -> List[Dict[str, Any]]:
        """Search using SerpAPI."""
        try:
            search = self.serp_client({
                "q": query,
                "api_key": os.getenv("SERPAPI_API_KEY"),
                "num": 5
            })
            results = search.get_dict()
            
            return [{
                "url": result["link"],
                "title": result["title"],
                "snippet": result.get("snippet", ""),
                "source": "serpapi"
            } for result in results.get("organic_results", [])]
            
        except Exception as e:
            logger.error(f"Error in SerpAPI search: {str(e)}")
            return []
    
    async def _process_results(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process and summarize search results.
        
        Args:
            query: Original query
            results: Raw search results
            
        Returns:
            List of processed results
        """
        try:
            # Format results for prompt
            results_str = "\n\n".join([
                f"Source {i+1}:\n"
                f"URL: {r['url']}\n"
                f"Title: {r['title']}\n"
                f"Content: {r['snippet']}"
                for i, r in enumerate(results)
            ])
            
            # Generate prompt
            prompt = self.summary_prompt.format(
                query=query,
                results=results_str
            )
            
            # Get LLM response
            response = await self.llm.complete(prompt)
            
            # Parse response
            processed = self._parse_response(response.text)
            
            # Add search metadata
            processed["metadata"]["search_time"] = datetime.utcnow().isoformat()
            
            return [processed]
            
        except Exception as e:
            logger.error(f"Error processing search results: {str(e)}")
            return []
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response as JSON.
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed response dictionary
        """
        try:
            import json
            # Extract JSON from response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            
            # Fallback to simple parsing
            return {
                "summary": response,
                "sources": [],
                "confidence": 0.0,
                "metadata": {
                    "total_sources": 0,
                    "filtered_sources": 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return {
                "summary": "Error processing search results.",
                "sources": [],
                "confidence": 0.0,
                "metadata": {
                    "total_sources": 0,
                    "filtered_sources": 0
                }
            } 