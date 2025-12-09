# ScholarX - Research Paper RAG Pipeline

A production-ready Python RAG pipeline for semantic search and question answering over research papers. Built with ChromaDB, Sentence Transformers, and advanced retrieval techniques.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up .env (see .env.example)
cp .env.example .env
# Edit .env with your settings

# Add papers
python3 add_papers.py

# Query interactively
python3 query_interactive.py

# Manage collection
python3 manage_papers.py
```

## 📁 Project Structure

```
python-rag/
├── main.py                    # Core functions
├── add_papers.py              # Add papers (interactive)
├── query_interactive.py       # Query interface (with modes)
├── manage_papers.py           # Collection management
├── view_results.py            # View saved queries
├── scale_example.py           # Large-scale ingestion
├── feature_showcase.py        # Demo all features
├── advanced_features.py       # Advanced features demo
├── api/                       # Feature APIs
│   ├── main_api.py           # Unified API interface
│   ├── paper_api.py          # Paper details & summaries
│   ├── citations.py          # Citation graph
│   ├── summaries.py          # Auto-summarization
│   ├── authors.py            # Author graph & stats
│   ├── topics.py             # Topic clustering
│   ├── rag_modes.py          # Advanced RAG modes
│   ├── deduplication.py       # Duplicate detection
│   ├── similarity.py         # Similarity checking
│   ├── ranking.py            # Citation rankings
│   └── query_logger.py       # Query analytics
├── config/                    # Configuration
│   ├── settings.py
│   ├── chroma_client.py
│   └── openai_client.py
├── ingestion/                 # Paper loading
│   ├── pdf_loader.py
│   ├── paper_fetcher.py
│   ├── ingest_pipeline.py
│   ├── enhanced_metadata.py
│   └── text_cleaner.py
├── processing/                # Text processing
│   ├── chunker.py
│   ├── advanced_chunker.py
│   └── embeddings.py
├── vectorstore/               # Vector operations
│   ├── upsert.py
│   └── query.py
├── rag/                       # RAG pipeline
│   ├── pipeline.py
│   ├── retriever.py
│   ├── generator.py
│   ├── hybrid_search.py
│   ├── reranker.py
│   ├── quality_scorer.py
│   ├── query_expander.py
│   └── search_enhanced.py
└── utils/                     # Utilities
    ├── logger.py
    └── timers.py
```

## ✨ Features

### Core Features (MVP)
- ✅ **PDF Ingestion**: Load papers from URLs or Semantic Scholar/ArXiv
- ✅ **Semantic Search**: Vector-based similarity search
- ✅ **Metadata Storage**: Title, authors, abstract, year, DOI, etc.
- ✅ **RAG QA**: Question answering with citations

### Good Project Features
- ✅ **Hybrid Search**: Combines semantic + keyword search
- ✅ **Paper API**: Get paper details, summaries, chunks
- ✅ **Citation Graph**: Find related and citing papers
- ✅ **Chunk Provenance**: Track which chunk came from which paper
- ✅ **Query Logging**: Analytics on queries and usage

### Standout Features
- ✅ **Auto Summaries**: Generate short, medium, and bullet-point summaries
- ✅ **Author Graph**: Author statistics, co-author networks, profiles
- ✅ **Topic Clustering**: Organize papers by topics using K-Means
- ✅ **Advanced RAG Modes**: Concise, detailed, explain, compare, literature survey
- ✅ **Multi-Document Synthesis**: Query across multiple specific papers

### Optional Features
- ✅ **Citation Ranking**: Rank papers by citation metrics
- ✅ **Related Papers**: Find similar papers automatically
- ✅ **Deduplication**: Detect and merge duplicate papers
- ✅ **Similarity Checking**: Compare papers or check text similarity
- ✅ **ArXiv Version Normalization**: Handle paper versions

## 📖 Usage

### Add Papers
```bash
python3 add_papers.py
# Option 1: Fetch by topic
# Option 2: Add from PDF URL
```

### Query System
```bash
python3 query_interactive.py
# Ask questions naturally
# Results saved automatically
```

### Manage Collection
```bash
python3 manage_papers.py
# List, delete, export, view stats
```

### Batch Add Many Papers
```bash
python3 scale_example.py
# Pre-configured batch ingestion
```

### Advanced Features Demo
```bash
python3 feature_showcase.py
# See all features in action
```

### Use API Programmatically
```python
from api.main_api import api

# Get paper details
paper = api.get_paper("paper-id")

# Generate summary
summary = api.generate_summary("paper-id")

# Advanced RAG modes
result = api.rag_concise("What is attention?")
result = api.rag_compare("Compare transformer architectures")
result = api.rag_survey("neural machine translation")

# Author analysis
stats = api.get_author_statistics()
profile = api.get_author("John Doe")

# Topic clustering
clusters = api.cluster_topics(num_clusters=5)

# Find duplicates
duplicates = api.find_duplicates()

# Check similarity
similar = api.check_similarity("some text", threshold=0.8)
```

## ⚙️ Configuration

Edit `.env`:
```env
EMBEDDING_PROVIDER=sentence-transformers  # Free, local
LLM_PROVIDER=simple                        # Template-based
CHUNK_SIZE=1000
MAX_PAPERS_PER_QUERY=5
```

## 🔧 Requirements

- Python 3.10+
- See `requirements.txt` for dependencies
- No API keys needed for free mode!

## 📝 License

ISC
