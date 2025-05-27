# Agentic RAG System Architecture

## System Overview

The Agentic RAG (Retrieval-Augmented Generation) system is a sophisticated document management and query system that combines advanced AI capabilities with robust document versioning, real-time updates, and user authentication. The system is built using a modern microservices architecture with FastAPI for the backend and Streamlit for the frontend.

## Architecture Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Streamlit UI   │◄────┤   FastAPI API   │◄────┤   Core Services │
│                 │     │                 │     │                 │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  WebSocket      │     │    Redis        │     │   SQLite DB     │
│  Real-time      │     │    Cache        │     │   Storage       │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Core Components

### 1. Frontend (Streamlit UI)
- **Location**: `app/ui/app.py`
- **Key Features**:
  - User authentication (login/register)
  - Document management interface
  - Real-time updates via WebSocket
  - Query interface with feedback
  - Analytics dashboard
  - User management (admin only)
- **State Management**:
  - Session-based authentication
  - Document subscriptions
  - Real-time notifications
  - WebSocket connection management

### 2. Backend API (FastAPI)
- **Location**: `app/api/main.py`
- **Key Features**:
  - RESTful API endpoints
  - Authentication middleware
  - WebSocket support
  - Document versioning
  - Query processing
  - Analytics endpoints
- **Security**:
  - JWT-based authentication
  - Role-based access control
  - Protected endpoints

### 3. Core Services

#### Authentication Service
- **Location**: `app/core/auth.py`
- **Features**:
  - User registration and login
  - JWT token management
  - Password hashing and verification
  - Role-based permissions

#### Cache Service
- **Location**: `app/core/cache.py`
- **Features**:
  - Redis-based caching
  - Query result caching
  - Document metadata caching
  - User session caching
  - Analytics data caching
- **Cache Categories**:
  - Query cache (1 hour)
  - Document cache (24 hours)
  - User cache (30 minutes)
  - Session cache (7 days)
  - Analytics cache (5 minutes)

#### Versioning Service
- **Location**: `app/core/versioning.py`
- **Features**:
  - Document version control
  - Version comparison
  - Rollback functionality
  - Version tagging
  - Change tracking
- **Database Tables**:
  - `document_versions`
  - `version_tags`

#### WebSocket Service
- **Location**: `app/core/websocket.py`
- **Features**:
  - Real-time updates
  - Document subscriptions
  - User subscriptions
  - System notifications
  - Connection management

### 4. Data Storage

#### SQLite Database
- **Features**:
  - Document storage
  - User management
  - Version control
  - Analytics data
- **Key Tables**:
  - `users`
  - `documents`
  - `document_versions`
  - `version_tags`
  - `feedback`
  - `analytics`

#### Redis Cache
- **Features**:
  - In-memory caching
  - Session management
  - Real-time data
  - Performance optimization

## API Endpoints

### Authentication
- `POST /auth/register` - User registration
- `POST /auth/login` - User login
- `GET /auth/users` - List users (admin only)

### Documents
- `POST /documents/upload` - Upload document
- `GET /documents` - List documents
- `DELETE /documents/{document_id}` - Delete document
- `POST /documents/{document_id}/versions` - Create version
- `GET /documents/{document_id}/versions` - List versions
- `POST /documents/{document_id}/versions/{version_id}/rollback` - Rollback version

### Query
- `POST /query` - Process query
- `POST /feedback` - Submit feedback

### Analytics
- `GET /analytics/feedback` - Get feedback analytics
- `GET /health` - System health check

### WebSocket
- `WS /ws/{user_id}` - WebSocket connection

## Dependencies

### Core Dependencies
- FastAPI (API framework)
- Streamlit (UI framework)
- SQLAlchemy (ORM)
- Redis (Caching)
- WebSockets (Real-time updates)
- JWT (Authentication)

### AI/ML Dependencies
- LangChain (LLM framework)
- LlamaIndex (RAG framework)
- Transformers (NLP models)
- ChromaDB/Qdrant (Vector stores)

### Utility Dependencies
- Pydantic (Data validation)
- Python-multipart (File uploads)
- Aiohttp (Async HTTP)
- Plotly (Visualization)

## Security Features

1. **Authentication**
   - JWT-based token authentication
   - Password hashing with bcrypt
   - Token expiration and refresh

2. **Authorization**
   - Role-based access control (ADMIN, EDITOR, VIEWER)
   - Protected endpoints
   - User session management

3. **Data Security**
   - Input validation
   - SQL injection prevention
   - XSS protection
   - CORS configuration

## Performance Optimizations

1. **Caching Strategy**
   - Multi-level caching (Redis)
   - Query result caching
   - Document metadata caching
   - Analytics data caching

2. **Real-time Updates**
   - WebSocket connections
   - Efficient message broadcasting
   - Connection pooling
   - Subscription management

3. **Database Optimization**
   - Indexed queries
   - Efficient joins
   - Connection pooling
   - Query optimization

## Monitoring and Analytics

1. **System Health**
   - Health check endpoints
   - Performance metrics
   - Resource utilization
   - Error tracking

2. **User Analytics**
   - Query performance
   - User feedback
   - Document usage
   - System metrics

## Development and Deployment

### Development Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment variables:
   ```bash
   API_URL=http://localhost:8000
   REDIS_URL=redis://localhost:6379
   ```

3. Start services:
   ```bash
   # Start Redis
   redis-server

   # Start API
   uvicorn app.api.main:app --reload

   # Start UI
   streamlit run app/ui/app.py
   ```

### Deployment Considerations
1. **Scaling**
   - Horizontal scaling of API servers
   - Redis cluster for caching
   - Load balancing
   - Database replication

2. **Monitoring**
   - Log aggregation
   - Performance monitoring
   - Error tracking
   - Resource utilization

3. **Security**
   - HTTPS/TLS
   - Rate limiting
   - API key management
   - Regular security audits

## Future Enhancements

1. **Planned Features**
   - Advanced document processing
   - Enhanced analytics
   - Multi-language support
   - Advanced search capabilities

2. **Technical Improvements**
   - Microservices architecture
   - Containerization
   - CI/CD pipeline
   - Automated testing

## Contributing

Please refer to the CONTRIBUTING.md file for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 