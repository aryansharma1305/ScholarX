# ScholarX - Data Sources & APIs

## 📊 Current Collection Status

**Total Papers in Collection:** 3 papers  
**Total Chunks:** 172 chunks  
**Total Characters:** 158,236 characters

### Paper Sources Breakdown:
- **ArXiv:** 172 chunks (100%)
- **Semantic Scholar:** 0 chunks (0%)
- **Direct PDF URLs:** 0 chunks (0%)

### Year Distribution:
- **2024:** 74 chunks (43%)
- **2022:** 33 chunks (19%)
- **2021:** 65 chunks (38%)

## 🔌 APIs Used

### 1. **ArXiv API** (Free, No API Key Required)
- **Base URL:** `http://export.arxiv.org/api/query`
- **Purpose:** Fetch research papers by topic
- **Rate Limits:** None (but be respectful)
- **Data Provided:**
  - Paper metadata (title, authors, abstract, year)
  - PDF URLs
  - ArXiv IDs
  - Publication dates
- **Usage:** Primary source for paper discovery

### 2. **Semantic Scholar API** (Free, No API Key Required)
- **Base URL:** `https://api.semanticscholar.org/graph/v1`
- **Purpose:** Fetch papers with rich metadata
- **Rate Limits:** 100 requests per 5 minutes (free tier)
- **Data Provided:**
  - Enhanced metadata
  - Citation information
  - Author details
  - PDF URLs (when available)
- **Usage:** Secondary source, used when ArXiv doesn't have enough results

### 3. **OpenAI API** (Optional, Requires API Key)
- **Purpose:** 
  - Embeddings generation (`text-embedding-3-large`)
  - LLM-based answer generation (`gpt-4o-mini`)
- **Current Status:** Not required (using free alternatives)
- **Free Alternatives:**
  - **Sentence Transformers** (for embeddings)
  - **Simple template-based** (for answer generation)

### 4. **ChromaDB** (Local, No API)
- **Purpose:** Vector database for storing embeddings
- **Type:** Local persistent storage
- **No external API calls**

## 📈 Paper Ingestion Sources

### Current Implementation:
1. **ArXiv** - Primary source
   - Searches by topic/keywords
   - Returns papers with PDF URLs
   - No authentication required

2. **Semantic Scholar** - Secondary source
   - Used when ArXiv doesn't provide enough results
   - Better metadata quality
   - Rate limited (100 req/5min)

3. **Direct PDF URLs** - Manual ingestion
   - Users can provide direct PDF URLs
   - Supports any publicly accessible PDF

## 🔄 Data Flow

```
User Query/Topic
    ↓
ArXiv API (primary)
    ↓
Semantic Scholar API (if needed)
    ↓
PDF Download & Text Extraction
    ↓
Text Processing & Chunking
    ↓
Embedding Generation (Sentence Transformers)
    ↓
ChromaDB Storage (local)
    ↓
Search & Retrieval
```

## 📊 Comparison with Google Scholar

| Feature | Google Scholar | ScholarX |
|---------|---------------|----------|
| **Paper Sources** | Multiple (publishers, ArXiv, etc.) | ArXiv, Semantic Scholar, Direct URLs |
| **Search Type** | Keyword-based | Semantic + Keyword (Hybrid) |
| **API Access** | Limited/Paid | Free (ArXiv, Semantic Scholar) |
| **Local Storage** | No | Yes (ChromaDB) |
| **RAG Answers** | No | Yes (with citations) |
| **Author Search** | Yes | Yes |
| **Year Filter** | Yes | Yes |
| **Citation Graph** | Yes | Yes (simplified) |
| **Paper Comparison** | No | Yes |
| **Topic Clustering** | No | Yes |
| **Auto Summaries** | No | Yes |
| **Offline Access** | No | Yes (cached papers) |

## 🎯 Current Limitations

1. **Paper Count:** Only 3 papers currently indexed
2. **Semantic Scholar Rate Limits:** 100 requests per 5 minutes
3. **No Publisher APIs:** Only open-access sources
4. **Citation Data:** Simplified (not full citation graph)

## 🚀 Scaling Potential

### To Scale to 1000+ Papers:
- **ArXiv:** Can fetch unlimited papers (no rate limits)
- **Semantic Scholar:** Need to respect rate limits (batch processing)
- **Storage:** ChromaDB can handle millions of vectors
- **Processing:** Batch ingestion recommended

### Recommended Approach:
1. Batch fetch from ArXiv (no limits)
2. Use Semantic Scholar for metadata enhancement (rate-limited)
3. Process in batches of 10-50 papers
4. Store locally in ChromaDB

## 📝 Summary

**APIs Used:** 2 external APIs (ArXiv, Semantic Scholar) + 1 optional (OpenAI)  
**Current Papers:** 3 papers  
**Storage:** Local ChromaDB (no external API)  
**Embeddings:** Local Sentence Transformers (no API)  
**LLM:** Template-based (no API) or optional OpenAI

**Total External API Calls:** Minimal (only for paper fetching)

