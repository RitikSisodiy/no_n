import json
import logging
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import aioredis
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache

logger = logging.getLogger(__name__)

class CacheService:
    """Service for handling Redis caching."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize the cache service."""
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        self._prefix = "agentic_rag:"
        
        # Cache expiration times (in seconds)
        self.expirations = {
            "query": 3600,  # 1 hour
            "document": 86400,  # 24 hours
            "user": 1800,  # 30 minutes
            "session": 604800,  # 7 days
            "analytics": 300,  # 5 minutes
        }
    
    async def init(self):
        """Initialize Redis connection and FastAPI cache."""
        try:
            self.redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            FastAPICache.init(
                RedisBackend(self.redis),
                prefix=self._prefix
            )
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {str(e)}")
            raise
    
    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
    
    def _get_key(self, category: str, key: str) -> str:
        """Get a namespaced cache key."""
        return f"{self._prefix}{category}:{key}"
    
    async def get(self, category: str, key: str) -> Optional[Any]:
        """Get a value from cache."""
        try:
            if not self.redis:
                return None
            
            cache_key = self._get_key(category, key)
            value = await self.redis.get(cache_key)
            
            if value:
                return json.loads(value)
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            return None
    
    async def set(
        self,
        category: str,
        key: str,
        value: Any,
        expire: Optional[int] = None
    ) -> bool:
        """Set a value in cache."""
        try:
            if not self.redis:
                return False
            
            cache_key = self._get_key(category, key)
            expire = expire or self.expirations.get(category, 3600)
            
            await self.redis.set(
                cache_key,
                json.dumps(value),
                ex=expire
            )
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")
            return False
    
    async def delete(self, category: str, key: str) -> bool:
        """Delete a value from cache."""
        try:
            if not self.redis:
                return False
            
            cache_key = self._get_key(category, key)
            await self.redis.delete(cache_key)
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error: {str(e)}")
            return False
    
    async def clear_category(self, category: str) -> bool:
        """Clear all keys in a category."""
        try:
            if not self.redis:
                return False
            
            pattern = self._get_key(category, "*")
            keys = await self.redis.keys(pattern)
            
            if keys:
                await self.redis.delete(*keys)
            return True
            
        except Exception as e:
            logger.error(f"Cache clear category error: {str(e)}")
            return False
    
    # Query caching
    @cache(
        expire=3600,
        namespace="query",
        key_builder=lambda *args, **kwargs: f"query:{hash(str(args) + str(kwargs))}"
    )
    async def cache_query_result(
        self,
        query: str,
        filters: Dict[str, Any],
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Cache a query result."""
        return result
    
    # Document caching
    async def cache_document(
        self,
        document_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Cache document metadata."""
        return await self.set("document", document_id, metadata)
    
    async def get_cached_document(
        self,
        document_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached document metadata."""
        return await self.get("document", document_id)
    
    # User session caching
    async def cache_user_session(
        self,
        user_id: str,
        session_data: Dict[str, Any]
    ) -> bool:
        """Cache user session data."""
        return await self.set("session", user_id, session_data)
    
    async def get_cached_session(
        self,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached user session."""
        return await self.get("session", user_id)
    
    # Analytics caching
    async def cache_analytics(
        self,
        category: str,
        data: Dict[str, Any],
        expire: Optional[int] = None
    ) -> bool:
        """Cache analytics data."""
        return await self.set("analytics", category, data, expire)
    
    async def get_cached_analytics(
        self,
        category: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached analytics data."""
        return await self.get("analytics", category)
    
    # Cache invalidation
    async def invalidate_document_cache(
        self,
        document_id: str
    ) -> bool:
        """Invalidate document-related caches."""
        try:
            # Delete document metadata
            await self.delete("document", document_id)
            
            # Delete related query caches
            pattern = self._get_key("query", "*")
            keys = await self.redis.keys(pattern)
            
            for key in keys:
                value = await self.redis.get(key)
                if value:
                    data = json.loads(value)
                    if any(
                        doc.get("document_id") == document_id
                        for doc in data.get("sources", [])
                    ):
                        await self.redis.delete(key)
            
            # Clear analytics cache
            await self.clear_category("analytics")
            
            return True
            
        except Exception as e:
            logger.error(f"Cache invalidation error: {str(e)}")
            return False
    
    async def invalidate_user_cache(
        self,
        user_id: str
    ) -> bool:
        """Invalidate user-related caches."""
        try:
            # Delete user session
            await self.delete("session", user_id)
            
            # Clear user-specific analytics
            await self.clear_category("analytics")
            
            return True
            
        except Exception as e:
            logger.error(f"User cache invalidation error: {str(e)}")
            return False 