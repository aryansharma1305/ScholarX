# On-Demand Paper Fetching

## 🔄 How It Works

ScholarX now operates as a **fully dynamic system** that fetches papers from APIs on-demand when you ask a question.


```
User Query
    ↓
Query Processing (normalization, expansion)
    ↓
Fetch Papers from APIs (ArXiv + Semantic Scholar)
    ↓
Ingest Papers (extract text, chunk, embed)
    ↓
Search & Retrieve Relevant Chunks
    ↓
Generate Answer with Citations
```

## 📡 APIs Used Per Query

When you ask a question, the system:

1. **Searches ArXiv API** (free, no key required)
   - Searches by topic/keywords from your query
   - Fetches up to `max_papers_per_query` papers (default: 5)
   - Gets PDF URLs and metadata

2. **Searches Semantic Scholar API** (free, no key required)
   - Used if ArXiv doesn't provide enough results
   - Better metadata quality
   - Rate limited (100 req/5min)
3. **Ingests Papers**
   - Downloads PDFs
   - Extracts text
   - Chunks content
   - Generates embeddings (local, free)
   - Stores in ChromaDB

4. **Answers Your Question**
   - Searches through fetched papers
   - Retrieves relevant chunks
   - Generates answer with citations
## ⚙️ Configuration

In `.env` or `config/settings.py`:

```python
max_papers_per_query: int = 5  # Papers fetched per query
```
## 🎯 Benefits

- **Always Fresh**: Gets latest papers from APIs
- **No Pre-loading**: No need to manually add papers
- **Dynamic**: Adapts to any query topic
- **Free**: Uses free APIs (ArXiv, Semantic Scholar)
- **Cached**: Fetched papers are stored for future queries

## 📊 Current Behavior

- **Default**: Fetches 5 papers per query from APIs
- **Storage**: Papers are stored in ChromaDB after fetching
- **Reuse**: Previously fetched papers are reused if relevant
- **Fresh Fetch**: Always checks APIs first for new papers

## 🔍 Example

```python
from main import query_rag

# Ask any question - papers will be fetched automatically
result = query_rag("What is transformer architecture?")
# System will:
# 1. Search ArXiv for "transformer architecture"
# 2. Fetch 5 relevant papers
# 3. Ingest them
# 4. Answer your question using those papers
```

## 🚀 Performance

- **First Query**: ~30-60 seconds (fetching + ingestion)
- **Subsequent Queries**: Faster (papers already cached)
- **New Topics**: Always fetches fresh papers

