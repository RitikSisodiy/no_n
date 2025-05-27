import logging
from typing import Dict, Any, Optional
from datetime import datetime

from llama_index.core.llms import LLM
from llama_index.core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

class MetadataAgent:
    """Agent responsible for processing and validating metadata filters."""
    
    def __init__(self, llm: LLM):
        """
        Initialize the metadata agent.
        
        Args:
            llm: Language model instance
        """
        self.llm = llm
        
        # Initialize prompt template
        self.filter_prompt = PromptTemplate(
            """You are a metadata processing agent. Your task is to analyze and validate metadata filters for a query.

Query: {query}
Current filters: {filters}

Please analyze the filters and:
1. Validate their format and values
2. Convert any date strings to ISO format
3. Normalize any text values
4. Remove any invalid filters
5. Add any implicit filters based on the query

Return a JSON object with the processed filters. For example:
{{
    "author": "John Doe",
    "date_range": {{
        "start": "2023-01-01",
        "end": "2023-12-31"
    }},
    "document_type": ["pdf", "docx"]
}}

If no valid filters are found, return an empty object {{}}.
"""
        )
    
    async def process_filters(
        self,
        query: str,
        filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process and validate metadata filters.
        
        Args:
            query: The original query
            filters: Raw metadata filters
            
        Returns:
            Processed and validated filters
        """
        try:
            # Format filters for prompt
            filters_str = str(filters)
            
            # Generate prompt
            prompt = self.filter_prompt.format(
                query=query,
                filters=filters_str
            )
            
            # Get LLM response
            response = await self.llm.complete(prompt)
            
            # Parse response as JSON
            processed_filters = self._parse_response(response.text)
            
            # Validate and normalize filters
            validated_filters = self._validate_filters(processed_filters)
            
            logger.info(f"Processed metadata filters: {validated_filters}")
            return validated_filters
            
        except Exception as e:
            logger.error(f"Error processing metadata filters: {str(e)}")
            return {}
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response as JSON.
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed filters dictionary
        """
        try:
            import json
            # Extract JSON from response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            return {}
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return {}
    
    def _validate_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize filter values.
        
        Args:
            filters: Raw filters dictionary
            
        Returns:
            Validated and normalized filters
        """
        validated = {}
        
        for key, value in filters.items():
            try:
                # Handle date ranges
                if key in ["date", "date_range"]:
                    if isinstance(value, dict):
                        if "start" in value:
                            value["start"] = self._normalize_date(value["start"])
                        if "end" in value:
                            value["end"] = self._normalize_date(value["end"])
                    else:
                        value = self._normalize_date(value)
                
                # Handle document types
                elif key == "document_type":
                    if isinstance(value, str):
                        value = [value.lower()]
                    elif isinstance(value, list):
                        value = [v.lower() for v in value]
                
                # Handle numeric ranges
                elif key in ["page", "page_range"]:
                    if isinstance(value, dict):
                        if "min" in value:
                            value["min"] = int(value["min"])
                        if "max" in value:
                            value["max"] = int(value["max"])
                    else:
                        value = int(value)
                
                # Handle text fields
                elif isinstance(value, str):
                    value = value.strip()
                
                validated[key] = value
                
            except Exception as e:
                logger.warning(f"Invalid filter value for {key}: {str(e)}")
                continue
        
        return validated
    
    def _normalize_date(self, date_str: str) -> str:
        """
        Normalize date string to ISO format.
        
        Args:
            date_str: Date string to normalize
            
        Returns:
            Normalized date string in ISO format
        """
        try:
            # Try parsing with different formats
            formats = [
                "%Y-%m-%d",
                "%d/%m/%Y",
                "%m/%d/%Y",
                "%B %d, %Y",
                "%d %B %Y"
            ]
            
            for fmt in formats:
                try:
                    date_obj = datetime.strptime(date_str, fmt)
                    return date_obj.strftime("%Y-%m-%d")
                except ValueError:
                    continue
            
            # If no format matches, try LLM-based parsing
            prompt = f"Convert this date to YYYY-MM-DD format: {date_str}"
            response = self.llm.complete(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error normalizing date {date_str}: {str(e)}")
            return date_str 