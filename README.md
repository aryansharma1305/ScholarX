# ScholarX - Research Paper RAG Pipeline

A production-ready Python RAG pipeline for semantic search and question answering over research papers. Built with ChromaDB, Sentence Transformers, and advanced retrieval techniques.

## 🚀 Quick Start

### Option 1: Streamlit Web App (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app.py
# Or use the script:
./run_app.sh

# App opens at http://localhost:8501
```

### Option 2: Command Line
```bash
# Install dependencies
pip install -r requirements.txt

# Query interactively
python3 query_interactive.py

# Or use Python API
python3
>>> from main import query_rag
>>> result = query_rag("What is transformer architecture?")
```

## 📁 Project Structure

```
python-rag/
├── main.py                    # Core functions
├── add_papers.py              # Add papers (interactive)
├── query_interactive.py        # Query interface (with modes)
├── manage_papers.py            # Collection management
├── view_results.py            # View saved queries
├── streamlit_app.py           # Main Streamlit application
├── run_app.sh                 # Quick start script
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
│   ├── query_logger.py       # Query analytics
│   ├── recommendations.py    # Paper recommendations
│   ├── trends.py             # Research trend analysis
│   ├── research_gaps.py      # Research gap identification
│   ├── query_intent.py       # Query intent classification
│   ├── exports.py            # Export capabilities
│   ├── relevance_ranking.py  # Relevance ranking
│   └── search.py             # Search interface
├── config/                    # Configuration
│   ├── settings.py
│   ├── chroma_client.py
│   └── openai_client.py
├── ingestion/                 # Paper loading
│   ├── pdf_loader.py
│   ├── paper_fetcher.py
│   ├── ingest_pipeline.py
│   ├── enhanced_metadata.py
│   ├── arxiv_enhanced.py
│   ├── semantic_scholar_enhanced.py
│   ├── crossref_api.py
│   ├── openalex_api.py
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
├── evaluation/                # Evaluation framework
│   ├── metrics.py            # Evaluation metrics
│   ├── baselines.py          # Baseline systems
│   ├── datasets.py           # Dataset loading
│   ├── statistical_analysis.py # Statistical tests
│   ├── run_evaluation.py     # Evaluation runner
│   └── ablation_study.py     # Ablation study
└── utils/                     # Utilities
    ├── logger.py
    ├── timers.py
    └── cache.py
```

## ✨ Features

### Core Features
- ✅ **PDF Ingestion**: Load papers from URLs or Semantic Scholar/ArXiv
- ✅ **Semantic Search**: Vector-based similarity search
- ✅ **Metadata Storage**: Title, authors, abstract, year, DOI, etc.
- ✅ **RAG QA**: Question answering with citations

### Advanced Features
- ✅ **Hybrid Search**: Combines semantic + keyword search
- ✅ **Paper API**: Get paper details, summaries, chunks
- ✅ **Citation Graph**: Find related and citing papers
- ✅ **Auto Summaries**: Generate short, medium, and bullet-point summaries
- ✅ **Author Graph**: Author statistics, co-author networks, profiles
- ✅ **Topic Clustering**: Organize papers by topics using K-Means
- ✅ **Advanced RAG Modes**: Concise, detailed, explain, compare, literature survey
- ✅ **Multi-Document Synthesis**: Query across multiple specific papers
- ✅ **Citation Ranking**: Rank papers by citation metrics
- ✅ **Related Papers**: Find similar papers automatically
- ✅ **Deduplication**: Detect and merge duplicate papers
- ✅ **Similarity Checking**: Compare papers or check text similarity

### Enhanced API Features
- ✅ **Full ArXiv API**: Field searches, Boolean operators, date filters
- ✅ **Full Semantic Scholar API**: Autocomplete, batch lookup, enhanced search
- ✅ **Crossref API**: Metadata retrieval, DOI resolution
- ✅ **OpenAlex API**: Comprehensive paper metadata
- ✅ **Author Search**: With h-index, citations, affiliations
- ✅ **Advanced Filters**: Year ranges, categories, citation counts, open access

### Unique Features
- ✅ **Paper Recommendations**: Based on queries and reading history
- ✅ **Research Trend Analysis**: Topic popularity over time, future predictions
- ✅ **Research Gap Identification**: Find underexplored areas
- ✅ **Query Intent Classification**: Automatic intent detection and routing
- ✅ **Export Capabilities**: BibTeX, CSV, JSON, Markdown formats
- ✅ **Performance Caching**: Intelligent caching layer
- ✅ **Relevance Ranking**: Multi-factor ranking with visual indicators

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

# Recommendations
recommendations = api.recommend_papers(limit=10)
recommendations_for_query = api.recommend_for_query("transformer architecture", limit=5)

# Trend Analysis
trends = api.analyze_trends(years=[2020, 2021, 2022, 2023, 2024])
field_trend = api.get_field_trends("transformer")
future = api.predict_trends("transformer", years_ahead=3)

# Research Gaps
gaps = api.find_gaps("neural machine translation", min_papers=5)
combination_gap = api.find_combination_gaps("transformer", "computer vision")
directions = api.suggest_directions("attention mechanisms")

# Query Intent
intent = api.classify_intent("Compare transformer and RNN architectures")
routing = api.route_query("What are the latest trends in NLP?")

# Exports
api.export_bibtex(filename="my_papers.bib")
api.export_csv(filename="papers.csv")
api.export_markdown(filename="library.md")
```

## ⚙️ Configuration

Edit `.env`:
```env
EMBEDDING_PROVIDER=sentence-transformers  # Free, local
LLM_PROVIDER=simple                        # Template-based
CHUNK_SIZE=1000
MAX_PAPERS_PER_QUERY=5
SEMANTIC_SCHOLAR_API_KEY=your_key_here     # Optional
```

## 🔧 Requirements

- Python 3.10+
- See `requirements.txt` for dependencies
- No API keys needed for free mode!

## 📡 APIs Used

### ArXiv API (Free, No Key)
- **Base URL**: `http://export.arxiv.org/api/query`
- **Features**: Field searches, Boolean operators, date filtering, sorting
- **Rate Limits**: None (be respectful, 3s delay recommended)

### Semantic Scholar API (Free, Optional Key)
- **Base URL**: `https://api.semanticscholar.org/graph/v1`
- **Features**: Autocomplete, batch lookup, enhanced search, citations/references
- **Rate Limits**: 100 requests per 5 minutes (free), higher with API key

### Crossref API (Free, No Key)
- **Base URL**: `https://api.crossref.org`
- **Features**: Metadata retrieval, DOI resolution

### OpenAlex API (Free, No Key)
- **Base URL**: `https://api.openalex.org`
- **Features**: Comprehensive paper metadata

## 📝 License

ISC
