import logging
from typing import List, Dict, Any

from llama_index.core.llms import LLM
from llama_index.core.schema import NodeWithScore
from llama_index.core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

class ReflectionAgent:
    """Agent responsible for evaluating and improving responses."""
    
    def __init__(self, llm: LLM):
        """
        Initialize the reflection agent.
        
        Args:
            llm: Language model instance
        """
        self.llm = llm
        
        # Initialize evaluation prompt
        self.eval_prompt = PromptTemplate(
            """You are a response evaluation agent. Your task is to evaluate and improve a response to a query.

Query: {query}

Retrieved Context:
{context}

Initial Response:
{response}

Please evaluate the response on the following criteria:
1. Accuracy: Is the response factually correct based on the context?
2. Completeness: Does it fully answer the query?
3. Clarity: Is it clear and well-structured?
4. Relevance: Is it directly relevant to the query?
5. Consistency: Are there any contradictions?

For each criterion, provide:
- Score (0-1)
- Explanation
- Suggested improvements

Then, provide an improved version of the response that addresses any issues found.

Format your response as JSON:
{{
    "evaluation": {{
        "accuracy": {{ "score": 0.9, "explanation": "...", "improvements": [...] }},
        "completeness": {{ "score": 0.8, "explanation": "...", "improvements": [...] }},
        "clarity": {{ "score": 0.95, "explanation": "...", "improvements": [...] }},
        "relevance": {{ "score": 0.85, "explanation": "...", "improvements": [...] }},
        "consistency": {{ "score": 1.0, "explanation": "...", "improvements": [...] }}
    }},
    "overall_score": 0.9,
    "improved_response": "..."
}}

If the initial response is already excellent (overall_score >= 0.95), you may return it unchanged.
"""
        )
    
    async def evaluate_response(
        self,
        query: str,
        response: str,
        nodes: List[NodeWithScore]
    ) -> str:
        """
        Evaluate and improve a response.
        
        Args:
            query: Original query
            response: Initial response to evaluate
            nodes: Retrieved nodes used for the response
            
        Returns:
            Improved response
        """
        try:
            # Format context from nodes
            context = self._format_context(nodes)
            
            # Generate evaluation prompt
            prompt = self.eval_prompt.format(
                query=query,
                context=context,
                response=response
            )
            
            # Get LLM response
            eval_response = await self.llm.complete(prompt)
            
            # Parse evaluation
            evaluation = self._parse_evaluation(eval_response.text)
            
            # Log evaluation results
            self._log_evaluation(evaluation)
            
            # Return improved response if score is low, otherwise return original
            if evaluation["overall_score"] < 0.95:
                return evaluation["improved_response"]
            return response
            
        except Exception as e:
            logger.error(f"Error evaluating response: {str(e)}")
            return response  # Return original response on error
    
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
                f"Relevance Score: {node.score:.2f}\n"
                f"Content: {node.node.text}\n"
            )
        return "\n".join(context_parts)
    
    def _parse_evaluation(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM evaluation response.
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed evaluation dictionary
        """
        try:
            import json
            # Extract JSON from response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            
            # Fallback to simple parsing if JSON extraction fails
            return {
                "evaluation": {},
                "overall_score": 1.0,
                "improved_response": response
            }
            
        except Exception as e:
            logger.error(f"Error parsing evaluation: {str(e)}")
            return {
                "evaluation": {},
                "overall_score": 1.0,
                "improved_response": response
            }
    
    def _log_evaluation(self, evaluation: Dict[str, Any]) -> None:
        """
        Log evaluation results.
        
        Args:
            evaluation: Evaluation dictionary
        """
        try:
            # Log overall score
            logger.info(
                f"Response evaluation - Overall Score: {evaluation['overall_score']:.2f}"
            )
            
            # Log individual criteria
            for criterion, details in evaluation.get("evaluation", {}).items():
                logger.info(
                    f"{criterion.capitalize()}: {details['score']:.2f} - "
                    f"{details['explanation']}"
                )
                if details.get("improvements"):
                    logger.info(f"Improvements: {', '.join(details['improvements'])}")
                    
        except Exception as e:
            logger.error(f"Error logging evaluation: {str(e)}")
    
    def _calculate_confidence(self, evaluation: Dict[str, Any]) -> float:
        """
        Calculate confidence score from evaluation.
        
        Args:
            evaluation: Evaluation dictionary
            
        Returns:
            Confidence score (0-1)
        """
        try:
            # Use overall score as confidence
            return float(evaluation.get("overall_score", 1.0))
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 1.0 