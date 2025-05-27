import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import psutil
import json
from collections import Counter
from sqlite3 import Connection
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

from app.core.storage import DocumentStorage

logger = logging.getLogger(__name__)

class AnalyticsService:
    """Service for collecting and processing analytics data."""
    
    def __init__(self, storage: DocumentStorage):
        """Initialize the analytics service."""
        self.storage = storage
        self._init_nltk()
    
    def _init_nltk(self):
        """Initialize NLTK resources."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
    
    def get_analytics(
        self,
        document_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get comprehensive analytics data."""
        try:
            with self.storage._get_connection() as conn:
                return {
                    "query_performance": self._get_query_performance(conn),
                    "document_usage": self._get_document_usage(conn),
                    "user_feedback": self._get_feedback_analytics(
                        conn, document_id, start_date, end_date
                    ),
                    "system_health": self._get_system_health()
                }
        except Exception as e:
            logger.error(f"Error getting analytics: {str(e)}")
            raise
    
    def _get_query_performance(self, conn: Connection) -> Dict[str, Any]:
        """Get query performance metrics."""
        try:
            cursor = conn.cursor()
            
            # Get total queries and growth
            cursor.execute("""
                SELECT COUNT(*) as total,
                       COUNT(CASE WHEN created_at >= datetime('now', '-1 day')
                            THEN 1 END) as recent
                FROM audit_logs
                WHERE action = 'process_query'
            """)
            total, recent = cursor.fetchone()
            
            # Get average response time
            cursor.execute("""
                SELECT AVG(CAST(json_extract(details, '$.processing_time') AS FLOAT))
                FROM audit_logs
                WHERE action = 'process_query'
                AND details LIKE '%processing_time%'
            """)
            avg_time = cursor.fetchone()[0] or 0
            
            # Get success rate
            cursor.execute("""
                SELECT COUNT(*) as total,
                       COUNT(CASE WHEN details LIKE '%"status": "success"%'
                            THEN 1 END) as successful
                FROM audit_logs
                WHERE action = 'process_query'
            """)
            total_queries, successful = cursor.fetchone()
            success_rate = successful / total if total > 0 else 0
            
            # Get query time series
            cursor.execute("""
                SELECT date(created_at) as date,
                       COUNT(*) as count
                FROM audit_logs
                WHERE action = 'process_query'
                GROUP BY date(created_at)
                ORDER BY date DESC
                LIMIT 30
            """)
            time_series = [
                {"timestamp": row[0], "count": row[1]}
                for row in cursor.fetchall()
            ]
            
            return {
                "total_queries": total,
                "query_growth": f"{((recent / total) * 100):.1f}%" if total > 0 else "0%",
                "avg_response_time": avg_time,
                "response_time_change": self._get_response_time_change(conn),
                "success_rate": success_rate,
                "query_time_series": time_series
            }
        except Exception as e:
            logger.error(f"Error getting query performance: {str(e)}")
            return {}
    
    def _get_document_usage(self, conn: Connection) -> Dict[str, Any]:
        """Get document usage statistics."""
        try:
            cursor = conn.cursor()
            
            # Get document counts
            cursor.execute("""
                SELECT COUNT(*) as total,
                       SUM(chunk_count) as chunks
                FROM document_status
            """)
            total_docs, total_chunks = cursor.fetchone()
            
            # Get document type distribution
            cursor.execute("""
                SELECT json_extract(metadata, '$.document_type') as type,
                       COUNT(*) as count
                FROM document_status
                GROUP BY type
            """)
            doc_types = dict(cursor.fetchall())
            
            # Get most accessed documents
            cursor.execute("""
                SELECT document_id,
                       COUNT(*) as access_count
                FROM audit_logs
                WHERE action = 'retrieve_document'
                GROUP BY document_id
                ORDER BY access_count DESC
                LIMIT 10
            """)
            top_docs = []
            for doc_id, count in cursor.fetchall():
                cursor.execute("""
                    SELECT title FROM document_status
                    WHERE document_id = ?
                """, (doc_id,))
                title = cursor.fetchone()[0]
                top_docs.append({
                    "document_name": title,
                    "access_count": count
                })
            
            # Calculate storage used
            cursor.execute("""
                SELECT SUM(length(metadata)) + SUM(length(error))
                FROM document_status
            """)
            storage_bytes = cursor.fetchone()[0] or 0
            
            return {
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "storage_used_mb": storage_bytes / (1024 * 1024),
                "document_types": doc_types,
                "top_documents": top_docs
            }
        except Exception as e:
            logger.error(f"Error getting document usage: {str(e)}")
            return {}
    
    def _get_feedback_analytics(
        self,
        conn: Connection,
        document_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get feedback analytics."""
        try:
            cursor = conn.cursor()
            
            # Build query conditions
            conditions = []
            params = []
            if document_id:
                conditions.append("document_id = ?")
                params.append(document_id)
            if start_date:
                conditions.append("created_at >= ?")
                params.append(start_date)
            if end_date:
                conditions.append("created_at <= ?")
                params.append(end_date)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            # Get feedback metrics
            cursor.execute(f"""
                SELECT COUNT(*) as total,
                       AVG(rating) as avg_rating
                FROM document_feedback
                WHERE {where_clause}
            """, params)
            total, avg_rating = cursor.fetchone()
            
            # Get rating distribution
            cursor.execute(f"""
                SELECT rating, COUNT(*) as count
                FROM document_feedback
                WHERE {where_clause}
                GROUP BY rating
                ORDER BY rating
            """, params)
            rating_dist = dict(cursor.fetchall())
            
            # Get common feedback themes
            cursor.execute(f"""
                SELECT feedback
                FROM document_feedback
                WHERE {where_clause}
                AND feedback IS NOT NULL
            """, params)
            feedback_texts = [row[0] for row in cursor.fetchall()]
            
            # Process feedback text
            stop_words = set(stopwords.words('english'))
            words = []
            for text in feedback_texts:
                tokens = word_tokenize(text.lower())
                words.extend([w for w in tokens if w.isalnum() and w not in stop_words])
            
            common_feedback = [word for word, _ in Counter(words).most_common(50)]
            
            return {
                "total_feedback": total,
                "average_rating": avg_rating or 0,
                "rating_distribution": rating_dist,
                "common_feedback": common_feedback
            }
        except Exception as e:
            logger.error(f"Error getting feedback analytics: {str(e)}")
            return {}
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics."""
        try:
            # Get system resource usage
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # Get cache hit rate
            with self.storage._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) as total,
                           COUNT(CASE WHEN expires_at > datetime('now')
                                THEN 1 END) as hits
                    FROM query_cache
                """)
                total, hits = cursor.fetchone()
                cache_hit_rate = hits / total if total > 0 else 0
            
            return {
                "api_response_time": self._get_avg_api_response_time(),
                "vector_store_status": "healthy",  # TODO: Implement actual check
                "cache_hit_rate": cache_hit_rate,
                "resource_usage": {
                    "cpu_usage": cpu_percent,
                    "memory_usage": memory.percent,
                    "memory_available": memory.available / (1024 * 1024 * 1024)  # GB
                }
            }
        except Exception as e:
            logger.error(f"Error getting system health: {str(e)}")
            return {}
    
    def _get_response_time_change(self, conn: Connection) -> float:
        """Calculate the change in average response time."""
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT AVG(CAST(json_extract(details, '$.processing_time') AS FLOAT))
                FROM audit_logs
                WHERE action = 'process_query'
                AND created_at >= datetime('now', '-1 day')
                AND details LIKE '%processing_time%'
            """)
            recent_avg = cursor.fetchone()[0] or 0
            
            cursor.execute("""
                SELECT AVG(CAST(json_extract(details, '$.processing_time') AS FLOAT))
                FROM audit_logs
                WHERE action = 'process_query'
                AND created_at >= datetime('now', '-2 days')
                AND created_at < datetime('now', '-1 day')
                AND details LIKE '%processing_time%'
            """)
            prev_avg = cursor.fetchone()[0] or 0
            
            return recent_avg - prev_avg
        except Exception as e:
            logger.error(f"Error calculating response time change: {str(e)}")
            return 0.0
    
    def _get_avg_api_response_time(self) -> float:
        """Calculate average API response time."""
        try:
            with self.storage._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT AVG(CAST(json_extract(details, '$.response_time') AS FLOAT))
                    FROM audit_logs
                    WHERE action = 'api_request'
                    AND created_at >= datetime('now', '-1 hour')
                """)
                return cursor.fetchone()[0] or 0
        except Exception as e:
            logger.error(f"Error calculating API response time: {str(e)}")
            return 0.0 