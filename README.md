# Agentic RAG System with Multi-Agent Orchestration

An end-to-end Retrieval-Augmented Generation (RAG) system that leverages multiple intelligent agents for enhanced document understanding, retrieval, and response generation with comprehensive source attribution.

## 🌟 Features

- **Multi-Format Document Processing**
  - PDF, DOCX, HTML support
  - Hierarchical document parsing
  - Metadata extraction
  - Page-level information preservation

- **Intelligent Agent System**
  - Retriever Agent: Smart document chunk retrieval
  - Metadata Agent: Context-aware filtering
  - Reflection Agent: Response quality assurance
  - Web Search Agent: External knowledge integration

- **Advanced RAG Capabilities**
  - Hybrid search (semantic + keyword + metadata)
  - Composable graph indices
  - Source attribution tracking
  - Confidence scoring

## 🏗️ Project Structure

```
agentic_rag/
├── app/                    # Application code
│   ├── api/               # FastAPI endpoints
│   ├── core/              # Core business logic
│   │   ├── agents/        # Agent implementations
│   │   ├── parsers/       # Document parsers
│   │   └── retrievers/    # Retrieval logic
│   ├── models/            # Data models
│   └── utils/             # Utility functions
├── config/                # Configuration files
├── data/                  # Data storage
│   ├── documents/         # Uploaded documents
│   └── vector_store/      # Vector database
├── tests/                 # Test suite
└── scripts/               # Utility scripts
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized deployment)
- LLAMA API key (for LLAMAParse)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/agentic-rag.git
   cd agentic-rag
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

### Running the System

1. Start the vector database:
   ```bash
   docker-compose up -d qdrant
   ```

2. Launch the API server:
   ```bash
   uvicorn app.api.main:app --reload
   ```

3. Access the API documentation at `http://localhost:8000/docs`

## 🛠️ Usage

### Document Ingestion

```python
from app.core.parsers import DocumentParser

parser = DocumentParser()
document = parser.parse("path/to/document.pdf")
```

### Query Processing

```python
from app.core.agents import QueryProcessor

processor = QueryProcessor()
response = processor.process_query(
    "What was the product launch date?",
    metadata_filters={"date_range": "2023"}
)
```

### Example Response

```json
{
    "answer": "The product was launched in March 2023.",
    "sources": [
        {
            "type": "document",
            "document": "Product_Overview.pdf",
            "page": 14,
            "chunk_id": "c_004",
            "heading": "Product Launch Details",
            "confidence": 0.95
        }
    ]
}
```

## 📊 Evaluation

The system is evaluated on multiple metrics:
- Retrieval accuracy (Precision@K)
- Answer quality (BLEU/ROUGE + human evaluation)
- Reflection agent correction rate
- Response latency
- Source attribution correctness

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- LLAMAIndex team for the excellent RAG framework
- Qdrant team for the vector database
- All contributors and users of this project 