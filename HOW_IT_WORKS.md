# How ScholarX Works - On-Demand Paper Fetching

## 🎯 Overview

ScholarX is a **fully dynamic** research paper RAG system that **automatically fetches papers from APIs** when you ask a question. You don't need to pre-load papers - just ask and the system will find and process relevant papers for you.

## 🔄 How It Works

### When You Ask a Question:

1. **Query Processing**
   - Your question is normalized and expanded
   - System extracts key terms and concepts

2. **API Search** (Automatic)
   - **ArXiv API**: Searches for papers matching your query
   - **Semantic Scholar API**: Gets additional papers if needed
   - Fetches up to 5 papers per query (configurable)

3. **Paper Ingestion** (Automatic)
   - Downloads PDFs from fetched URLs
   - Extracts text content
   - Chunks the content intelligently
   - Generates embeddings (local, free)
   - Stores in ChromaDB

4. **Answer Generation**
   - Searches through ingested papers
   - Retrieves most relevant chunks
   - Generates answer with citations
   - Returns sources and context

## 📡 APIs Used

### Per Query:
- **ArXiv API** (Primary)
  - Free, no API key required
  - Unlimited requests
  - Searches by topic/keywords
  - Returns PDF URLs + metadata

- **Semantic Scholar API** (Secondary)
  - Free, no API key required
  - Rate limit: 100 requests per 5 minutes
  - Better metadata quality
  - Used if ArXiv doesn't provide enough results

### Total APIs: 2 External APIs (Both Free)

## 📊 Current Status

- **Papers in Collection**: 3 (from initial testing)
- **Behavior**: System will fetch new papers for each query
- **Storage**: Fetched papers are cached in ChromaDB for future queries

## 🚀 Example Usage

```python
from main import query_rag

# Ask any question - papers fetched automatically
result = query_rag("What is attention mechanism in transformers?")

# System will:
# 1. Search ArXiv for "attention mechanism transformers"
# 2. Fetch 5 relevant papers
# 3. Ingest them (extract, chunk, embed)
# 4. Answer your question using those papers
# 5. Return answer with citations
```

## ⚙️ Configuration

In `config/settings.py`:
- `max_papers_per_query = 5` - Papers fetched per query
- Can be changed via `.env` file: `MAX_PAPERS_PER_QUERY=10`

## 💡 Key Features

✅ **Fully Dynamic**: No pre-loading needed  
✅ **Always Fresh**: Gets latest papers from APIs  
✅ **Free APIs**: No API keys required  
✅ **Smart Caching**: Stores fetched papers for reuse  
✅ **Automatic**: Just ask questions, system handles the rest  

## 🔍 Why Only 3 Papers Currently?

The 3 papers in the collection are from initial testing. The system is now configured to:
- **Always fetch papers from APIs** when you ask a question
- **Not rely on pre-loaded papers**
- **Dynamically adapt to any query topic**

You can ask about **any topic** and the system will automatically fetch relevant papers!

