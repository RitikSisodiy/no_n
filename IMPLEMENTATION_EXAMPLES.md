# Agentic RAG System Implementation Examples

## 1. Authentication Implementation

### 1.1 User Registration
```python
# app/core/auth.py
from datetime import datetime, timedelta
from typing import Optional
from uuid import uuid4
from fastapi import HTTPException, Security
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

class UserRegistration(BaseModel):
    email: EmailStr
    password: str
    full_name: str

class AuthService:
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
        
    async def register_user(self, user_data: UserRegistration) -> dict:
        # Check if user exists
        if await self.get_user_by_email(user_data.email):
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create user
        user_id = str(uuid4())
        hashed_password = self.pwd_context.hash(user_data.password)
        
        user = {
            "id": user_id,
            "email": user_data.email,
            "password_hash": hashed_password,
            "full_name": user_data.full_name,
            "role": "user",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "is_active": True
        }
        
        # Save to database
        await self.db.users.insert_one(user)
        
        # Generate tokens
        access_token = self.create_access_token(user_id)
        refresh_token = self.create_refresh_token(user_id)
        
        return {
            "user": {k: v for k, v in user.items() if k != "password_hash"},
            "access_token": access_token,
            "refresh_token": refresh_token
        }

# API Endpoint
@router.post("/register", response_model=UserResponse)
async def register(user_data: UserRegistration):
    auth_service = AuthService()
    return await auth_service.register_user(user_data)
```

### 1.2 User Login
```python
# app/core/auth.py
class AuthService:
    async def authenticate_user(self, email: str, password: str) -> Optional[dict]:
        user = await self.get_user_by_email(email)
        if not user:
            return None
        if not self.pwd_context.verify(password, user["password_hash"]):
            return None
        return user

    async def login(self, email: str, password: str) -> dict:
        user = await self.authenticate_user(email, password)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Incorrect email or password"
            )
        
        # Generate tokens
        access_token = self.create_access_token(user["id"])
        refresh_token = self.create_refresh_token(user["id"])
        
        # Update last login
        await self.db.users.update_one(
            {"id": user["id"]},
            {"$set": {"last_login": datetime.utcnow()}}
        )
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "user": {k: v for k, v in user.items() if k != "password_hash"}
        }

# API Endpoint
@router.post("/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    auth_service = AuthService()
    return await auth_service.login(form_data.username, form_data.password)
```

## 2. Document Management

### 2.1 Document Creation
```python
# app/core/document.py
from typing import List, Optional
from uuid import uuid4
from datetime import datetime
from pydantic import BaseModel

class DocumentCreate(BaseModel):
    title: str
    content: str
    metadata: dict

class DocumentService:
    async def create_document(
        self,
        user_id: str,
        document_data: DocumentCreate
    ) -> dict:
        document_id = str(uuid4())
        now = datetime.utcnow()
        
        document = {
            "id": document_id,
            "title": document_data.title,
            "content": document_data.content,
            "metadata": document_data.metadata,
            "created_by": user_id,
            "created_at": now,
            "updated_at": now,
            "version": 1,
            "status": "active"
        }
        
        # Save to database
        await self.db.documents.insert_one(document)
        
        # Create initial version
        await self.create_version(document_id, document, user_id)
        
        # Add to vector store
        await self.vector_store.add_documents([document])
        
        return document

    async def create_version(
        self,
        document_id: str,
        document: dict,
        user_id: str
    ) -> dict:
        version_id = str(uuid4())
        version = {
            "version_id": version_id,
            "document_id": document_id,
            "version_number": document["version"],
            "content_hash": self.calculate_content_hash(document["content"]),
            "metadata": document["metadata"],
            "created_at": datetime.utcnow(),
            "created_by": user_id,
            "changes": None  # For initial version
        }
        
        await self.db.document_versions.insert_one(version)
        return version

# API Endpoint
@router.post("/documents", response_model=DocumentResponse)
async def create_document(
    document: DocumentCreate,
    current_user: dict = Depends(get_current_user)
):
    document_service = DocumentService()
    return await document_service.create_document(current_user["id"], document)
```

### 2.2 Document Versioning
```python
# app/core/versioning.py
class VersioningService:
    async def get_document_versions(
        self,
        document_id: str,
        limit: int = 10,
        offset: int = 0
    ) -> List[dict]:
        versions = await self.db.document_versions.find(
            {"document_id": document_id}
        ).sort("version_number", -1).skip(offset).limit(limit).to_list(None)
        
        return versions

    async def compare_versions(
        self,
        version_id_1: str,
        version_id_2: str
    ) -> dict:
        v1 = await self.db.document_versions.find_one({"version_id": version_id_1})
        v2 = await self.db.document_versions.find_one({"version_id": version_id_2})
        
        if not v1 or not v2:
            raise HTTPException(status_code=404, detail="Version not found")
        
        # Get document content for both versions
        doc1 = await self.db.documents.find_one({"id": v1["document_id"]})
        doc2 = await self.db.documents.find_one({"id": v2["document_id"]})
        
        # Calculate differences
        diff = self.calculate_diff(doc1["content"], doc2["content"])
        
        return {
            "version1": v1,
            "version2": v2,
            "differences": diff
        }

    async def rollback_to_version(
        self,
        document_id: str,
        version_id: str,
        user_id: str
    ) -> dict:
        version = await self.db.document_versions.find_one({"version_id": version_id})
        if not version:
            raise HTTPException(status_code=404, detail="Version not found")
        
        # Get document
        document = await self.db.documents.find_one({"id": document_id})
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Create new version with rolled back content
        new_version = await self.create_version(
            document_id,
            {
                **document,
                "content": version["content"],
                "version": document["version"] + 1
            },
            user_id
        )
        
        # Update document
        await self.db.documents.update_one(
            {"id": document_id},
            {
                "$set": {
                    "content": version["content"],
                    "version": document["version"] + 1,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        return new_version
```

## 3. Caching Implementation

### 3.1 Redis Cache Service
```python
# app/core/cache.py
from typing import Any, Optional
import json
from redis import Redis
from datetime import timedelta

class CacheService:
    def __init__(self, redis_url: str):
        self.redis = Redis.from_url(redis_url, decode_responses=True)
        self.default_ttl = {
            "query": timedelta(hours=1),
            "document": timedelta(hours=24),
            "user": timedelta(minutes=30),
            "session": timedelta(days=7),
            "analytics": timedelta(minutes=5)
        }

    async def get(self, key: str, category: str = "query") -> Optional[Any]:
        """Get value from cache"""
        data = self.redis.get(f"{category}:{key}")
        if data:
            return json.loads(data)
        return None

    async def set(
        self,
        key: str,
        value: Any,
        category: str = "query",
        ttl: Optional[timedelta] = None
    ) -> bool:
        """Set value in cache"""
        ttl = ttl or self.default_ttl.get(category)
        return self.redis.setex(
            f"{category}:{key}",
            ttl,
            json.dumps(value)
        )

    async def delete(self, key: str, category: str = "query") -> bool:
        """Delete value from cache"""
        return bool(self.redis.delete(f"{category}:{key}"))

    async def clear_category(self, category: str) -> bool:
        """Clear all keys in a category"""
        keys = self.redis.keys(f"{category}:*")
        if keys:
            return bool(self.redis.delete(*keys))
        return True

# Usage in Document Service
class DocumentService:
    def __init__(self, cache_service: CacheService):
        self.cache_service = cache_service

    async def get_document(self, document_id: str) -> Optional[dict]:
        # Try cache first
        cached_doc = await self.cache_service.get(document_id, "document")
        if cached_doc:
            return cached_doc
        
        # Get from database
        document = await self.db.documents.find_one({"id": document_id})
        if document:
            # Cache for future use
            await self.cache_service.set(document_id, document, "document")
        
        return document
```

## 4. Vector Store Integration

### 4.1 ChromaDB Implementation
```python
# app/core/vector_store.py
from typing import List, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

class ChromaVectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Use OpenAI embeddings
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-ada-002"
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_function
        )

    async def add_documents(self, documents: List[dict]) -> List[str]:
        """Add documents to vector store"""
        ids = [doc["id"] for doc in documents]
        texts = [doc["content"] for doc in documents]
        metadatas = [{
            "title": doc["title"],
            "created_by": doc["created_by"],
            "created_at": doc["created_at"].isoformat(),
            "version": doc["version"]
        } for doc in documents]
        
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        
        return ids

    async def search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None
    ) -> List[dict]:
        """Search documents"""
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=filter
        )
        
        return [{
            "id": id,
            "content": doc,
            "metadata": meta,
            "distance": dist
        } for id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )]

    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from vector store"""
        try:
            self.collection.delete(ids=document_ids)
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
```

## 5. Real-time Updates

### 5.1 WebSocket Implementation
```python
# app/core/websocket.py
from typing import Dict, Set
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
import json

class ConnectionManager:
    def __init__(self):
        # Store active connections
        self.active_connections: Dict[str, WebSocket] = {}
        # Store document subscriptions
        self.document_subscribers: Dict[str, Set[str]] = {}
        # Store user subscriptions
        self.user_subscribers: Dict[str, Set[str]] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        # Clean up subscriptions
        for doc_id, subscribers in self.document_subscribers.items():
            subscribers.discard(user_id)
        for target_id, subscribers in self.user_subscribers.items():
            subscribers.discard(user_id)

    async def subscribe_document(self, user_id: str, document_id: str):
        if document_id not in self.document_subscribers:
            self.document_subscribers[document_id] = set()
        self.document_subscribers[document_id].add(user_id)

    async def unsubscribe_document(self, user_id: str, document_id: str):
        if document_id in self.document_subscribers:
            self.document_subscribers[document_id].discard(user_id)

    async def broadcast_document_update(
        self,
        document_id: str,
        action: str,
        data: dict
    ):
        if document_id not in self.document_subscribers:
            return
        
        message = {
            "type": "document_update",
            "action": action,
            "data": {
                "id": document_id,
                "changes": data,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        for user_id in self.document_subscribers[document_id]:
            if user_id in self.active_connections:
                try:
                    await self.active_connections[user_id].send_json(message)
                except WebSocketDisconnect:
                    self.disconnect(user_id)

# WebSocket Endpoint
@router.websocket("/ws/{user_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str,
    manager: ConnectionManager = Depends(get_connection_manager)
):
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "subscribe_document":
                await manager.subscribe_document(
                    user_id,
                    data["document_id"]
                )
            elif data["type"] == "unsubscribe_document":
                await manager.unsubscribe_document(
                    user_id,
                    data["document_id"]
                )
            
    except WebSocketDisconnect:
        manager.disconnect(user_id)
```

## 6. Query Processing

### 6.1 RAG Implementation
```python
# app/core/rag.py
from typing import List, Optional
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

class RAGService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = Chroma(
            collection_name="documents",
            embedding_function=self.embeddings
        )
        
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0
        )
        
        self.qa_prompt = PromptTemplate(
            template="""Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            Context: {context}
            
            Question: {question}
            
            Answer:""",
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 5}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.qa_prompt}
        )

    async def process_query(
        self,
        query: str,
        user_id: str,
        filter: Optional[dict] = None
    ) -> dict:
        # Log query
        await self.log_query(query, user_id)
        
        # Process query
        result = self.qa_chain({"query": query})
        
        # Cache result
        await self.cache_service.set(
            f"query:{hash(query)}",
            result,
            category="query"
        )
        
        return {
            "answer": result["result"],
            "sources": [
                {
                    "id": doc.metadata["id"],
                    "title": doc.metadata["title"],
                    "content": doc.page_content,
                    "score": doc.metadata.get("score", 0)
                }
                for doc in result["source_documents"]
            ]
        }

    async def log_query(self, query: str, user_id: str):
        """Log query for analytics"""
        log_entry = {
            "query": query,
            "user_id": user_id,
            "timestamp": datetime.utcnow(),
            "type": "query"
        }
        await self.db.query_logs.insert_one(log_entry)

# API Endpoint
@router.post("/query", response_model=QueryResponse)
async def process_query(
    query: QueryRequest,
    current_user: dict = Depends(get_current_user)
):
    rag_service = RAGService()
    return await rag_service.process_query(
        query.text,
        current_user["id"],
        query.filter
    )
```

## 7. Analytics Implementation

### 7.1 Analytics Service
```python
# app/core/analytics.py
from typing import Dict, List
from datetime import datetime, timedelta
from collections import Counter

class AnalyticsService:
    async def get_user_activity(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict[str, int]:
        """Get user activity statistics"""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Get document activities
        doc_activities = await self.db.document_activities.find({
            "user_id": user_id,
            "timestamp": {"$gte": start_date}
        }).to_list(None)
        
        # Get query activities
        query_activities = await self.db.query_logs.find({
            "user_id": user_id,
            "timestamp": {"$gte": start_date}
        }).to_list(None)
        
        return {
            "documents_created": len([a for a in doc_activities if a["action"] == "create"]),
            "documents_updated": len([a for a in doc_activities if a["action"] == "update"]),
            "queries_made": len(query_activities),
            "active_days": len(set(
                a["timestamp"].date() for a in doc_activities + query_activities
            ))
        }

    async def get_system_metrics(self) -> Dict[str, any]:
        """Get system-wide metrics"""
        # Get document counts
        doc_count = await self.db.documents.count_documents({})
        user_count = await self.db.users.count_documents({})
        
        # Get query statistics
        recent_queries = await self.db.query_logs.find({
            "timestamp": {
                "$gte": datetime.utcnow() - timedelta(hours=24)
            }
        }).to_list(None)
        
        # Calculate query metrics
        query_count = len(recent_queries)
        avg_response_time = sum(
            q.get("response_time", 0) for q in recent_queries
        ) / query_count if query_count > 0 else 0
        
        # Get cache statistics
        cache_stats = await self.cache_service.get_stats()
        
        return {
            "total_documents": doc_count,
            "total_users": user_count,
            "queries_last_24h": query_count,
            "avg_response_time": avg_response_time,
            "cache_hit_rate": cache_stats["hit_rate"],
            "cache_miss_rate": cache_stats["miss_rate"]
        }

    async def get_popular_queries(
        self,
        days: int = 7,
        limit: int = 10
    ) -> List[Dict[str, any]]:
        """Get most popular queries"""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Get recent queries
        queries = await self.db.query_logs.find({
            "timestamp": {"$gte": start_date}
        }).to_list(None)
        
        # Count query frequencies
        query_counter = Counter(q["query"] for q in queries)
        
        # Get top queries
        return [
            {
                "query": query,
                "count": count,
                "percentage": count / len(queries) * 100
            }
            for query, count in query_counter.most_common(limit)
        ]

# API Endpoints
@router.get("/analytics/user/{user_id}", response_model=UserAnalytics)
async def get_user_analytics(
    user_id: str,
    days: int = 30,
    current_user: dict = Depends(get_current_user)
):
    analytics_service = AnalyticsService()
    return await analytics_service.get_user_activity(user_id, days)

@router.get("/analytics/system", response_model=SystemAnalytics)
async def get_system_analytics(
    current_user: dict = Depends(get_current_admin)
):
    analytics_service = AnalyticsService()
    return await analytics_service.get_system_metrics()
```

## 8. Error Handling

### 8.1 Custom Exception Handlers
```python
# app/core/exceptions.py
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Union

class DocumentNotFoundError(HTTPException):
    def __init__(self, document_id: str):
        super().__init__(
            status_code=404,
            detail=f"Document {document_id} not found"
        )

class UnauthorizedError(HTTPException):
    def __init__(self, detail: str = "Not authorized"):
        super().__init__(
            status_code=401,
            detail=detail
        )

class ValidationError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=422,
            detail=detail
        )

async def http_exception_handler(
    request: Request,
    exc: HTTPException
) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "type": exc.__class__.__name__
            }
        }
    )

# Register exception handlers
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(DocumentNotFoundError, http_exception_handler)
app.add_exception_handler(UnauthorizedError, http_exception_handler)
app.add_exception_handler(ValidationError, http_exception_handler)
```

### 8.2 Error Logging
```python
# app/core/logging.py
import logging
from typing import Optional
from datetime import datetime
import traceback

class ErrorLogger:
    def __init__(self):
        self.logger = logging.getLogger("error_logger")
        self.logger.setLevel(logging.ERROR)
        
        # File handler
        fh = logging.FileHandler("error.log")
        fh.setLevel(logging.ERROR)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        
        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    async def log_error(
        self,
        error: Exception,
        context: Optional[dict] = None
    ):
        """Log error with context"""
        error_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": error.__class__.__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {}
        }
        
        self.logger.error(
            f"Error occurred: {error_data['error_type']} - "
            f"{error_data['error_message']}\n"
            f"Context: {error_data['context']}\n"
            f"Traceback: {error_data['traceback']}"
        )
        
        # Store in database for analysis
        await self.db.error_logs.insert_one(error_data)

# Usage in middleware
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        error_logger = ErrorLogger()
        await error_logger.log_error(e, {
            "path": request.url.path,
            "method": request.method,
            "client": request.client.host if request.client else None
        })
        raise
```

## 9. Testing Examples

### 9.1 Unit Tests
```python
# tests/test_auth.py
import pytest
from app.core.auth import AuthService
from app.models.user import UserRegistration

@pytest.mark.asyncio
async def test_user_registration():
    auth_service = AuthService()
    user_data = UserRegistration(
        email="test@example.com",
        password="Test123!",
        full_name="Test User"
    )
    
    result = await auth_service.register_user(user_data)
    
    assert result["user"]["email"] == user_data.email
    assert result["user"]["full_name"] == user_data.full_name
    assert "access_token" in result
    assert "refresh_token" in result

@pytest.mark.asyncio
async def test_user_login():
    auth_service = AuthService()
    
    # First register a user
    user_data = UserRegistration(
        email="test@example.com",
        password="Test123!",
        full_name="Test User"
    )
    await auth_service.register_user(user_data)
    
    # Try to login
    result = await auth_service.login(
        email="test@example.com",
        password="Test123!"
    )
    
    assert "access_token" in result
    assert "refresh_token" in result
    assert result["user"]["email"] == user_data.email

# tests/test_document.py
@pytest.mark.asyncio
async def test_document_creation():
    document_service = DocumentService()
    user_id = "test_user_id"
    
    document_data = DocumentCreate(
        title="Test Document",
        content="This is a test document",
        metadata={"category": "test"}
    )
    
    document = await document_service.create_document(
        user_id,
        document_data
    )
    
    assert document["title"] == document_data.title
    assert document["content"] == document_data.content
    assert document["created_by"] == user_id
    assert document["version"] == 1

@pytest.mark.asyncio
async def test_document_versioning():
    document_service = DocumentService()
    user_id = "test_user_id"
    
    # Create initial document
    doc = await document_service.create_document(
        user_id,
        DocumentCreate(
            title="Test Doc",
            content="Initial content",
            metadata={}
        )
    )
    
    # Update document
    updated_doc = await document_service.update_document(
        doc["id"],
        "Updated content",
        user_id
    )
    
    # Get versions
    versions = await document_service.get_document_versions(doc["id"])
    
    assert len(versions) == 2
    assert versions[0]["version_number"] == 2
    assert versions[1]["version_number"] == 1
```

### 9.2 Integration Tests
```python
# tests/integration/test_api.py
import pytest
from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

def test_document_workflow():
    # Register user
    register_response = client.post(
        "/auth/register",
        json={
            "email": "test@example.com",
            "password": "Test123!",
            "full_name": "Test User"
        }
    )
    assert register_response.status_code == 200
    tokens = register_response.json()
    
    # Create document
    create_response = client.post(
        "/documents",
        json={
            "title": "Test Document",
            "content": "Test content",
            "metadata": {"category": "test"}
        },
        headers={"Authorization": f"Bearer {tokens['access_token']}"}
    )
    assert create_response.status_code == 200
    document = create_response.json()
    
    # Query document
    query_response = client.post(
        "/query",
        json={"text": "What is the test document about?"},
        headers={"Authorization": f"Bearer {tokens['access_token']}"}
    )
    assert query_response.status_code == 200
    result = query_response.json()
    
    assert "answer" in result
    assert "sources" in result
    assert len(result["sources"]) > 0

def test_websocket_connection():
    with client.websocket_connect("/ws/test_user") as websocket:
        # Subscribe to document
        websocket.send_json({
            "type": "subscribe_document",
            "document_id": "test_doc_id"
        })
        
        # Receive subscription confirmation
        response = websocket.receive_json()
        assert response["type"] == "subscription_confirmed"
        
        # Simulate document update
        websocket.send_json({
            "type": "document_update",
            "document_id": "test_doc_id",
            "action": "update",
            "data": {"content": "Updated content"}
        })
        
        # Receive update notification
        update = websocket.receive_json()
        assert update["type"] == "document_update"
        assert update["data"]["id"] == "test_doc_id"
```

## 10. Deployment Configuration Examples

### 10.1 Docker Deployment

#### Dockerfile (FastAPI API)
```dockerfile
# Dockerfile for FastAPI API
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Dockerfile (Streamlit UI)
```dockerfile
# Dockerfile for Streamlit UI
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 10.2 Docker Compose

#### docker-compose.yml
```yaml
version: '3.8'
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: agentic_api
    env_file:
      - .env
    ports:
      - "8000:8000"
    depends_on:
      - redis
      - chromadb
    restart: always

  ui:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: agentic_ui
    command: ["streamlit", "run", "app/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
    ports:
      - "8501:8501"
    depends_on:
      - api
    restart: always

  redis:
    image: redis:6.2-alpine
    container_name: agentic_redis
    ports:
      - "6379:6379"
    restart: always

  chromadb:
    image: chromadb/chroma:latest
    container_name: agentic_chromadb
    ports:
      - "8001:8000"
    volumes:
      - ./chroma_data:/chroma/.chroma/index
    restart: always
```

### 10.3 Kubernetes Deployment

#### api-deployment.yaml
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentic-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: agentic-api
  template:
    metadata:
      labels:
        app: agentic-api
    spec:
      containers:
      - name: api
        image: your-dockerhub/agentic-api:latest
        ports:
        - containerPort: 8000
        envFrom:
        - secretRef:
            name: agentic-env
---
apiVersion: v1
kind: Service
metadata:
  name: agentic-api
spec:
  type: ClusterIP
  selector:
    app: agentic-api
  ports:
    - protocol: TCP
      port: 8000
      targetPort: 8000
```

#### ui-deployment.yaml
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentic-ui
spec:
  replicas: 1
  selector:
    matchLabels:
      app: agentic-ui
  template:
    metadata:
      labels:
        app: agentic-ui
    spec:
      containers:
      - name: ui
        image: your-dockerhub/agentic-ui:latest
        ports:
        - containerPort: 8501
---
apiVersion: v1
kind: Service
metadata:
  name: agentic-ui
spec:
  type: LoadBalancer
  selector:
    app: agentic-ui
  ports:
    - protocol: TCP
      port: 8501
      targetPort: 8501
```

#### redis-deployment.yaml
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentic-redis
spec:
  replicas: 1
  selector:
    matchLabels:
      app: agentic-redis
  template:
    metadata:
      labels:
        app: agentic-redis
    spec:
      containers:
      - name: redis
        image: redis:6.2-alpine
        ports:
        - containerPort: 6379
---
apiVersion: v1
kind: Service
metadata:
  name: agentic-redis
spec:
  type: ClusterIP
  selector:
    app: agentic-redis
  ports:
    - protocol: TCP
      port: 6379
      targetPort: 6379
```

### 10.4 Cloud Deployment (Example: AWS Elastic Beanstalk)

#### Dockerrun.aws.json (Multi-container)
```json
{
  "version": "3",
  "services": {
    "api": {
      "image": "your-dockerhub/agentic-api:latest",
      "essential": true,
      "memory": 512,
      "portMappings": [
        { "containerPort": 8000 }
      ]
    },
    "ui": {
      "image": "your-dockerhub/agentic-ui:latest",
      "essential": true,
      "memory": 512,
      "portMappings": [
        { "containerPort": 8501 }
      ]
    },
    "redis": {
      "image": "redis:6.2-alpine",
      "essential": false,
      "memory": 256,
      "portMappings": [
        { "containerPort": 6379 }
      ]
    }
  }
}
```

### 10.5 Environment Variables Example (.env)
```env
# .env
API_SECRET_KEY=your-secret-key
DATABASE_URL=sqlite:///./db.sqlite3
REDIS_URL=redis://redis:6379/0
CHROMA_PERSIST_DIRECTORY=/chroma/.chroma/index
OPENAI_API_KEY=your-openai-key
```

### 10.6 Production Tips
- Use a reverse proxy (e.g., Nginx) for SSL termination and routing.
- Set up environment secrets using your cloud provider's secret manager.
- Use managed Redis and managed database services for reliability.
- Enable health checks and auto-scaling in your orchestrator.
- Monitor logs and metrics using cloud-native or third-party tools. 