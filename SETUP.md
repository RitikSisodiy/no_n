# Agentic RAG System Setup Guide

## Prerequisites

### System Requirements
- Python 3.9+ (recommended: Python 3.11)
- Redis Server 6.0+
- 4GB+ RAM (8GB recommended)
- 10GB+ free disk space
- Git

### Operating System Support
- Linux (Ubuntu 20.04+, CentOS 8+)
- macOS (10.15+)
- Windows 10/11 (with WSL2 recommended)

## Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/agentic-rag.git
cd agentic-rag
```

### 2. Create and Activate Virtual Environment

#### Using venv (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

#### Using Conda
```bash
# Create conda environment
conda create -n agentic-rag python=3.11
conda activate agentic-rag
```

### 3. Install Dependencies

#### Core Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt

# Or install specific dependency groups
pip install -r requirements.txt --no-deps  # Core dependencies only
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118  # With CUDA support
```

#### Optional Dependencies
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install testing dependencies
pip install -r requirements-test.txt
```

### 4. Environment Configuration

Create a `.env` file in the project root:
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_RELOAD=true

# Database Configuration
DATABASE_URL=sqlite:///./app.db
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10

# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_POOL_SIZE=10
REDIS_POOL_TIMEOUT=30

# Authentication
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# AI/ML Configuration
OPENAI_API_KEY=your-openai-api-key
MODEL_NAME=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-ada-002
VECTOR_STORE_TYPE=chroma  # or qdrant

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
LOG_FILE=app.log

# Security
CORS_ORIGINS=["http://localhost:8501"]
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600
```

### 5. Database Setup

#### SQLite (Default)
```bash
# Initialize database
python -m app.core.db.init_db

# Run migrations
alembic upgrade head
```

#### PostgreSQL (Optional)
```bash
# Update DATABASE_URL in .env
DATABASE_URL=postgresql://user:password@localhost:5432/agentic_rag

# Install PostgreSQL dependencies
pip install psycopg2-binary

# Initialize database
python -m app.core.db.init_db
```

### 6. Redis Setup

#### Local Installation

##### Linux (Ubuntu)
```bash
# Install Redis
sudo apt update
sudo apt install redis-server

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Verify installation
redis-cli ping
```

##### macOS
```bash
# Install Redis using Homebrew
brew install redis

# Start Redis
brew services start redis

# Verify installation
redis-cli ping
```

##### Windows (WSL2)
```bash
# Install Redis in WSL2
sudo apt update
sudo apt install redis-server

# Start Redis
sudo service redis-server start

# Verify installation
redis-cli ping
```

#### Docker (Alternative)
```bash
# Pull Redis image
docker pull redis:latest

# Run Redis container
docker run --name agentic-redis -p 6379:6379 -d redis:latest

# Verify connection
docker exec -it agentic-redis redis-cli ping
```

### 7. Vector Store Setup

#### ChromaDB (Default)
```bash
# Initialize ChromaDB
python -m app.core.vector_store.init_chroma

# Verify setup
python -m app.core.vector_store.test_chroma
```

#### Qdrant (Optional)
```bash
# Update VECTOR_STORE_TYPE in .env
VECTOR_STORE_TYPE=qdrant

# Install Qdrant
docker pull qdrant/qdrant

# Run Qdrant
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Initialize Qdrant
python -m app.core.vector_store.init_qdrant
```

## Starting the Application

### 1. Start Redis
```bash
# If using system service
sudo systemctl start redis-server

# If using Docker
docker start agentic-redis
```

### 2. Start the API Server
```bash
# Development mode
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 3. Start the Streamlit UI
```bash
# Development mode
streamlit run app/ui/app.py

# Production mode
streamlit run app/ui/app.py --server.port 8501 --server.address 0.0.0.0
```

### 4. Verify Installation
```bash
# Check API health
curl http://localhost:8000/health

# Check Redis connection
redis-cli ping

# Check database connection
python -m app.core.db.test_connection
```

## Development Tools

### 1. Code Quality
```bash
# Run linters
flake8 app/
black app/
isort app/

# Run type checking
mypy app/

# Run pre-commit hooks
pre-commit run --all-files
```

### 2. Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_auth.py

# Run with coverage
pytest --cov=app tests/

# Run async tests
pytest --asyncio-mode=auto tests/
```

### 3. API Documentation
```bash
# Access Swagger UI
http://localhost:8000/docs

# Access ReDoc
http://localhost:8000/redoc
```

## Troubleshooting

### Common Issues

1. **Redis Connection Error**
   ```bash
   # Check Redis status
   sudo systemctl status redis-server
   
   # Check Redis logs
   sudo journalctl -u redis-server
   
   # Verify Redis connection
   redis-cli ping
   ```

2. **Database Connection Error**
   ```bash
   # Check database file permissions
   ls -l app.db
   
   # Verify database connection
   python -m app.core.db.test_connection
   
   # Check database logs
   tail -f app.log
   ```

3. **API Server Issues**
   ```bash
   # Check API logs
   tail -f app.log
   
   # Verify API health
   curl http://localhost:8000/health
   
   # Check port availability
   netstat -tulpn | grep 8000
   ```

4. **Streamlit Issues**
   ```bash
   # Check Streamlit logs
   streamlit run app/ui/app.py --logger.level=debug
   
   # Clear Streamlit cache
   streamlit cache clear
   
   # Verify port availability
   netstat -tulpn | grep 8501
   ```

### Performance Tuning

1. **Redis Optimization**
   ```bash
   # Edit Redis configuration
   sudo nano /etc/redis/redis.conf
   
   # Key settings to adjust:
   maxmemory 2gb
   maxmemory-policy allkeys-lru
   ```

2. **Database Optimization**
   ```bash
   # Run database optimization
   python -m app.core.db.optimize
   
   # Check database statistics
   python -m app.core.db.stats
   ```

3. **API Performance**
   ```bash
   # Adjust worker count
   uvicorn app.api.main:app --workers $(nproc)
   
   # Enable compression
   uvicorn app.api.main:app --compress
   ```

## Security Checklist

1. **Environment Variables**
   - [ ] JWT_SECRET_KEY is set and secure
   - [ ] Database credentials are properly configured
   - [ ] API keys are stored securely
   - [ ] CORS settings are properly configured

2. **Database Security**
   - [ ] Database file permissions are set correctly
   - [ ] Regular backups are configured
   - [ ] Sensitive data is encrypted

3. **API Security**
   - [ ] Rate limiting is enabled
   - [ ] CORS is properly configured
   - [ ] Input validation is implemented
   - [ ] Authentication is working

4. **Redis Security**
   - [ ] Redis is not exposed to public network
   - [ ] Redis password is set
   - [ ] Redis maxmemory is configured

## Monitoring Setup

1. **Logging Configuration**
   ```bash
   # Configure logging
   python -m app.core.logging.setup
   
   # View logs
   tail -f app.log
   ```

2. **Performance Monitoring**
   ```bash
   # Install monitoring tools
   pip install prometheus-client
   
   # Start metrics collection
   python -m app.core.monitoring.start
   ```

3. **Health Checks**
   ```bash
   # Run health check
   python -m app.core.health.check
   
   # Monitor system resources
   python -m app.core.monitoring.resources
   ```

## Backup and Recovery

1. **Database Backup**
   ```bash
   # Create backup
   python -m app.core.db.backup
   
   # Restore from backup
   python -m app.core.db.restore backup_file.db
   ```

2. **Redis Backup**
   ```bash
   # Create Redis backup
   redis-cli SAVE
   
   # Restore Redis data
   redis-cli FLUSHALL
   redis-cli --pipe < dump.rdb
   ```

3. **Configuration Backup**
   ```bash
   # Backup configuration
   cp .env .env.backup
   cp app/core/config.py app/core/config.py.backup
   ```

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Redis Documentation](https://redis.io/documentation)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [LlamaIndex Documentation](https://gpt-index.readthedocs.io/)

## Support

For additional support:
1. Check the [GitHub Issues](https://github.com/yourusername/agentic-rag/issues)
2. Join our [Discord Community](https://discord.gg/your-discord)
3. Contact the maintainers at support@yourdomain.com 