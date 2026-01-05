# ScholarX - Architecture Document
## High-Level & Low-Level Design

**Project Title:** ScholarX - Research Paper RAG Pipeline  
**Document Version:** 1.0  
**Date:** December 2025  
**Review Date:** January 6, 2026

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Architecture Choice](#2-architecture-choice)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Low-Level Design](#4-low-level-design)
5. [Database Schema Design](#5-database-schema-design)
6. [Data Exchange Contracts](#6-data-exchange-contracts)
7. [Component Details](#7-component-details)
8. [Deployment Architecture](#8-deployment-architecture)
9. [Diagrams Reference](#9-diagrams-reference)

---

## 1. Architecture Overview

### 1.1 System Purpose

ScholarX is a Retrieval-Augmented Generation (RAG) system designed to enable semantic search and intelligent question answering over research papers. The system processes academic papers from multiple sources, creates vector embeddings, and provides natural language query capabilities.

### 1.2 Key Requirements

- **Functional Requirements:**
  - Ingest papers from multiple sources (ArXiv, Semantic Scholar, etc.)
  - Perform semantic and keyword-based search
  - Generate answers to natural language queries
  - Provide advanced analytics (trends, gaps, recommendations)
  - Support multiple export formats

- **Non-Functional Requirements:**
  - Query latency: < 1 second
  - Support for 10,000+ papers
  - Scalable architecture
  - Modular and maintainable code
  - API-first design

---

## 2. Architecture Choice

### 2.1 Selected Architecture: **Modular Monolithic Architecture**

**Decision:** ScholarX uses a **Modular Monolithic Architecture** with clear separation of concerns.

### 2.2 Justification

#### Why Not Microservices?
1. **Project Scale**: Medium-scale application (10,000 papers, single team)
2. **Team Size**: Small development team (2-4 members)
3. **Deployment Complexity**: Microservices require orchestration, service discovery, distributed tracing
4. **Network Overhead**: In-process calls are faster than network calls
5. **Resource Constraints**: Single server deployment is more cost-effective
6. **Development Speed**: Faster development and testing in monolithic structure

#### Why Not Event-Driven?
1. **Synchronous Operations**: Most operations are request-response based
2. **Real-time Requirements**: Queries need immediate responses
3. **Complexity**: Event-driven adds complexity without clear benefits for this use case
4. **State Management**: Most operations are stateless

#### Why Not Serverless?
1. **Cold Start Issues**: Embedding models require warm-up time
2. **State Management**: Vector database needs persistent connections
3. **Cost**: Long-running processes are more expensive in serverless
4. **Control**: Need fine-grained control over resource allocation

#### Why Modular Monolithic?
1. **Clear Boundaries**: Well-defined modules (ingestion, processing, RAG, API)
2. **Maintainability**: Easy to understand and modify
3. **Performance**: Direct function calls, no network latency
4. **Testing**: Easier to test individual modules
5. **Future-Proof**: Can evolve to microservices if needed
6. **Resource Efficiency**: Single process, shared memory

### 2.3 Architecture Evolution Path

The current modular design allows for future evolution:

```
Current: Modular Monolithic
    ↓
Future Option 1: Service-Oriented (if scale requires)
    - Separate services: Ingestion, RAG, Analytics
    - Shared database
    - API Gateway

Future Option 2: Microservices (if team/scale grows)
    - Independent services
    - Service mesh
    - Distributed tracing
```

---

## 3. High-Level Architecture

### 3.1 System Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Streamlit UI │  │  CLI Tools   │  │  Python API  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ RAG Pipeline │  │ Feature APIs │  │  Evaluation  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    BUSINESS LOGIC LAYER                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Retrieval   │  │  Generation  │  │  Ingestion   │      │
│  │   Logic      │  │    Logic     │  │   Logic      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                      DATA LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  ChromaDB    │  │  Metadata    │  │  Query Logs  │      │
│  │ Vector Store │  │   Storage    │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                  EXTERNAL SERVICES LAYER                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   ArXiv      │  │ Semantic     │  │   Crossref   │      │
│  │     API      │  │  Scholar API │  │     API      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Module Organization

**Core Modules:**
1. **ingestion/** - Paper fetching and loading
2. **processing/** - Text processing, chunking, embeddings
3. **rag/** - RAG pipeline components
4. **api/** - Feature APIs
5. **vectorstore/** - Vector database operations
6. **config/** - Configuration management
7. **utils/** - Utility functions

**Key Principles:**
- **Separation of Concerns**: Each module has a single responsibility
- **Dependency Inversion**: High-level modules don't depend on low-level modules
- **Interface Segregation**: Clear interfaces between modules
- **Open/Closed**: Open for extension, closed for modification

### 3.3 Data Flow

**Query Flow:**
```
User Query → UI → RAG Pipeline → Retriever → Hybrid Search → 
ChromaDB → Reranker → Generator → Answer with Citations → UI
```

**Ingestion Flow:**
```
Paper Source → Ingestion Pipeline → PDF Loader → Text Cleaner → 
Chunker → Embedding Generator → ChromaDB + Metadata Store
```

---

## 4. Low-Level Design

### 4.1 RAG Pipeline Component

**Class: RAGPipeline**

```python
class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()
        self.reranker = Reranker()
        self.query_expander = QueryExpander()
    
    def run_pipeline(self, query: str, top_k: int = 5) -> RAGResponse:
        # 1. Normalize and expand query
        normalized = normalize_query(query)
        expanded = expand_query(normalized)
        
        # 2. Retrieve context
        candidates = self.retriever.retrieve(expanded, top_k * 2)
        
        # 3. Rerank results
        ranked = self.reranker.rerank(candidates, query)
        
        # 4. Generate answer
        answer = self.generator.generate(query, ranked[:top_k])
        
        return RAGResponse(query, answer, ranked[:top_k])
```

**Design Patterns:**
- **Strategy Pattern**: Different retrieval strategies (semantic, keyword, hybrid)
- **Template Method**: Pipeline defines algorithm skeleton
- **Factory Pattern**: Create different generator types

### 4.2 Retriever Component

**Class: Retriever**

```python
class Retriever:
    def __init__(self):
        self.vector_store = ChromaDBClient()
        self.embedding_model = EmbeddingGenerator()
        self.hybrid_search = HybridSearch()
    
    def retrieve(self, query: str, top_k: int) -> List[Chunk]:
        # 1. Generate query embedding
        query_embedding = self.embedding_model.generate(query)
        
        # 2. Hybrid search
        results = self.hybrid_search.search(
            query_embedding, 
            query_text=query,
            top_k=top_k
        )
        
        return results
```

**Design Patterns:**
- **Adapter Pattern**: Adapts ChromaDB interface to internal interface
- **Strategy Pattern**: Different search strategies

### 4.3 Hybrid Search Component

**Class: HybridSearch**

```python
class HybridSearch:
    def __init__(self, semantic_weight=0.7, keyword_weight=0.3):
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
    
    def search(self, query_embedding, query_text, top_k) -> List[Chunk]:
        # 1. Semantic search
        semantic_results = self.vector_store.similarity_search(
            query_embedding, top_k * 2
        )
        
        # 2. Keyword search
        keyword_results = self.vector_store.keyword_search(
            query_text, top_k * 2
        )
        
        # 3. Combine scores
        combined = self.combine_scores(
            semantic_results, keyword_results
        )
        
        return combined[:top_k]
```

### 4.4 Ingestion Pipeline Component

**Class: IngestionPipeline**

```python
class IngestionPipeline:
    def __init__(self):
        self.pdf_loader = PDFLoader()
        self.text_cleaner = TextCleaner()
        self.chunker = Chunker()
        self.embedding_gen = EmbeddingGenerator()
        self.vector_store = ChromaDBClient()
    
    def ingest(self, pdf_url: str, metadata: dict) -> str:
        # 1. Load PDF
        raw_text = self.pdf_loader.load(pdf_url)
        
        # 2. Clean text
        cleaned = self.text_cleaner.clean(raw_text)
        
        # 3. Chunk text
        chunks = self.chunker.chunk(cleaned)
        
        # 4. Generate embeddings
        embeddings = self.embedding_gen.batch_generate(chunks)
        
        # 5. Store in vector database
        paper_id = self.vector_store.upsert(chunks, embeddings, metadata)
        
        return paper_id
```

**Design Patterns:**
- **Pipeline Pattern**: Sequential processing stages
- **Builder Pattern**: Build complex paper objects step by step

---

## 5. Database Schema Design

### 5.1 ChromaDB Vector Store Schema

**Collection: rag-papers**

**Document Structure:**
```json
{
  "id": "chunk-uuid",
  "embedding": [384-dim vector],
  "metadata": {
    "paper_id": "paper-uuid",
    "chunk_text": "text content...",
    "chunk_index": 0,
    "paper_title": "Paper Title",
    "authors": "Author1, Author2",
    "year": 2024,
    "source": "arxiv",
    "doi": "10.1234/example"
  }
}
```

**Indexing:**
- HNSW (Hierarchical Navigable Small World) index
- Cosine similarity metric
- 384-dimensional vectors

### 5.2 Metadata Storage (SQLite/JSON)

**Paper Metadata:**
```json
{
  "paper_id": "uuid",
  "title": "string",
  "abstract": "text",
  "authors": ["author1", "author2"],
  "year": 2024,
  "doi": "string",
  "pdf_url": "string",
  "source": "arxiv|semantic_scholar|crossref|openalex|manual",
  "arxiv_id": "string (nullable)",
  "semantic_scholar_id": "string (nullable)",
  "citation_count": 0,
  "reference_count": 0,
  "created_at": "timestamp",
  "updated_at": "timestamp"
}
```

### 5.3 Query Logs Schema

**Query Log Entry:**
```json
{
  "query_id": "uuid",
  "query_text": "string",
  "answer": "text",
  "papers_used": ["paper_id1", "paper_id2"],
  "chunks_retrieved": 5,
  "model_used": "simple|openai|ollama",
  "time_taken": 0.5,
  "mode": "concise|detailed|explain|compare|survey",
  "created_at": "timestamp"
}
```

### 5.4 ER Diagram Summary

**Entities:**
- Paper (1) → (N) Chunk
- Paper (M) → (N) Author (via Paper_Author junction)
- Paper (1) → (N) Citation (self-referential)
- Paper (1) → (N) Query_Log
- Paper (M) → (N) Topic_Cluster (via Paper_Topic junction)

**See ERASER_AI_PROMPTS.md for detailed ER Diagram prompt.**

---

## 6. Data Exchange Contracts

### 6.1 Internal Data Exchange

**Format:** Python Objects (Dict/List), JSON for serialization

**Frequency:**
- Query Processing: Real-time (on-demand, 200-800ms)
- Paper Ingestion: Batch or on-demand (7-26 seconds per paper)
- Cache Updates: On-demand with TTL (1 hour default)
- Logging: Synchronous (every query, < 10ms overhead)

**Modes:**
- **Direct Function Calls**: In-process communication (primary)
- **JSON Serialization**: API responses, file exports
- **File-based**: Exports (BibTeX, CSV, JSON, Markdown)

### 6.2 External API Data Exchange

#### ArXiv API
- **Mode**: REST API (HTTPS GET)
- **Format**: XML/Atom Feed
- **Frequency**: On-demand
- **Rate Limit**: 3 seconds delay recommended (polite)
- **Contract**: 
  ```http
  GET http://export.arxiv.org/api/query?search_query=all:transformer&max_results=10
  ```
- **Response**: Atom feed with paper metadata

#### Semantic Scholar API
- **Mode**: REST API (HTTPS GET/POST)
- **Format**: JSON
- **Frequency**: On-demand
- **Rate Limit**: 100 requests per 5 minutes (free tier)
- **Contract**:
  ```http
  GET https://api.semanticscholar.org/graph/v1/paper/{paper_id}
  Authorization: Bearer {api_key}
  ```
- **Response**: JSON with paper details, citations, references

#### Crossref API
- **Mode**: REST API (HTTPS GET)
- **Format**: JSON
- **Frequency**: On-demand
- **Rate Limit**: Polite (with User-Agent header)
- **Contract**:
  ```http
  GET https://api.crossref.org/works/{doi}
  User-Agent: ScholarX/1.0
  ```
- **Response**: JSON with metadata

#### OpenAlex API
- **Mode**: REST API (HTTPS GET)
- **Format**: JSON
- **Frequency**: On-demand
- **Rate Limit**: Polite
- **Contract**:
  ```http
  GET https://api.openalex.org/works/{work_id}
  ```
- **Response**: JSON with comprehensive metadata

### 6.3 Data Sets

**Input Data:**
- Research Papers (PDF files): 1-50 MB per paper
- Paper Metadata (JSON): 5-10 KB per paper
- User Queries (Text): < 1000 characters
- API Responses: 10-100 KB per response

**Output Data:**
- Answers with Citations (Text + JSON): 1-5 KB
- Search Results (JSON array): 10-50 KB
- Exported Files:
  - BibTeX: 1-2 KB per paper
  - CSV: 0.5-1 KB per paper
  - JSON: 5-10 KB per paper
  - Markdown: 2-5 KB per paper
- Analytics Results (JSON): 5-20 KB

**Storage Requirements:**
- Vector Embeddings: 384 dim × 4 bytes = 1.5 KB per chunk
- Average chunks per paper: 50-200
- Total per paper: 75-300 KB vectors
- Paper Metadata: 5-10 KB per paper
- Query Logs: ~1 KB per query

**Scaling Estimates:**
- 1,000 papers: ~300 MB vectors + 10 MB metadata = ~310 MB
- 10,000 papers: ~3 GB vectors + 100 MB metadata = ~3.1 GB

---

## 7. Component Details

### 7.1 Presentation Layer Components

#### StreamlitApp
- **Purpose**: Web-based user interface
- **Technology**: Streamlit framework
- **Features**: 
  - Interactive query interface
  - Paper management
  - Analytics visualization
  - Export functionality

#### CLITools
- **Purpose**: Command-line interface
- **Technology**: Python argparse
- **Features**:
  - Query interface
  - Paper management commands
  - Batch operations

#### PythonAPI
- **Purpose**: Programmatic access
- **Technology**: Python classes and functions
- **Features**:
  - Unified API interface
  - All system capabilities exposed

### 7.2 Application Layer Components

#### RAGPipeline
- **Purpose**: Orchestrate RAG operations
- **Dependencies**: Retriever, Generator, Reranker
- **Responsibilities**:
  - Query processing
  - Pipeline orchestration
  - Response formatting

#### FeatureAPIs
- **Purpose**: Expose specific features
- **Components**:
  - CitationAPI
  - SummaryAPI
  - AuthorAPI
  - TopicAPI
  - TrendAPI
  - GapAnalysisAPI
  - RecommendationAPI
  - ExportAPI

### 7.3 Business Logic Layer Components

#### Retriever
- **Purpose**: Retrieve relevant context
- **Dependencies**: VectorStore, EmbeddingGenerator
- **Strategies**: Semantic, Keyword, Hybrid

#### Generator
- **Purpose**: Generate answers
- **Dependencies**: LLM Provider (optional)
- **Modes**: Template-based, LLM-based

#### IngestionPipeline
- **Purpose**: Process and ingest papers
- **Dependencies**: PDFLoader, Chunker, EmbeddingGenerator
- **Stages**: Load → Clean → Chunk → Embed → Store

### 7.4 Data Layer Components

#### ChromaDBClient
- **Purpose**: Vector storage and retrieval
- **Technology**: ChromaDB
- **Operations**: Upsert, Query, Delete

#### MetadataStore
- **Purpose**: Paper metadata storage
- **Technology**: SQLite/JSON files
- **Operations**: CRUD operations on metadata

---

## 8. Deployment Architecture

### 8.1 Development Environment

**Single Machine Deployment:**
- All components on developer machine
- ChromaDB in persistent mode
- Streamlit on localhost:8501
- Direct file access

### 8.2 Production Environment

**Recommended Deployment:**

```
┌─────────────────────────────────────────┐
│         Load Balancer (Optional)        │
└─────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
┌───────▼────────┐    ┌────────▼────────┐
│  Web Server    │    │  Web Server     │
│  (Streamlit)   │    │  (Streamlit)    │
│  Instance 1    │    │  Instance 2      │
└───────┬────────┘    └────────┬────────┘
        │                       │
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │   Application Server  │
        │  (RAG Pipeline, APIs) │
        └───────────┬───────────┘
                    │
    ┌───────────────┼───────────────┐
    │               │               │
┌───▼────┐   ┌─────▼─────┐   ┌────▼────┐
│ChromaDB│   │ Metadata  │   │  Cache  │
│ Server │   │  Database  │   │ (Redis) │
└────────┘   └────────────┘   └─────────┘
```

**Server Specifications:**
- **Web Server**: 2 CPU, 4GB RAM, Linux
- **Application Server**: 4 CPU, 8GB RAM, Linux, SSD
- **Vector DB Server**: 4 CPU, 8GB RAM, Linux, SSD (for 10K+ papers)
- **Metadata DB**: 2 CPU, 4GB RAM, Linux, HDD/SSD

### 8.3 Scalability Considerations

**Horizontal Scaling:**
- Multiple Streamlit instances behind load balancer
- Stateless application servers
- Shared vector database

**Vertical Scaling:**
- Increase CPU for embedding generation
- Increase RAM for larger collections
- SSD for faster vector search

**Caching Strategy:**
- Query result caching (Redis)
- Embedding model caching (in-memory)
- Metadata caching (in-memory)

---

## 9. Diagrams Reference

All architecture diagrams should be created using ERASER AI with the prompts provided in `ERASER_AI_PROMPTS.md`:

1. **ER Diagram** - Database schema and relationships
2. **Use Case Diagram** - System functionality from user perspective
3. **Class Diagram** - Object-oriented design
4. **DFD Level 0** - Context diagram
5. **DFD Level 1** - System decomposition
6. **Component Diagram** - System components and interfaces
7. **Sequence Diagram - Query Flow** - RAG query processing
8. **Sequence Diagram - Ingestion Flow** - Paper ingestion process
9. **Deployment Diagram** - System deployment architecture

**Legend for All Diagrams:**
- Use consistent color coding across diagrams
- Include legends explaining symbols and colors
- Add notes for complex interactions
- Maintain consistent naming conventions

---

## 10. Design Patterns Used

1. **Strategy Pattern**: Different search/retrieval strategies
2. **Template Method**: Pipeline algorithm skeleton
3. **Factory Pattern**: Create different generator types
4. **Adapter Pattern**: Adapt external APIs to internal interface
5. **Builder Pattern**: Build complex objects step by step
6. **Singleton Pattern**: Configuration and logger instances
7. **Observer Pattern**: Query logging and analytics

---

## 11. Security Considerations

1. **API Keys**: Stored in environment variables, never in code
2. **Input Validation**: All user inputs validated
3. **Rate Limiting**: Respect external API rate limits
4. **Error Handling**: Graceful error handling, no sensitive data exposure
5. **Logging**: No sensitive data in logs

---

## 12. Performance Optimization

1. **Caching**: Query results and embeddings cached
2. **Batch Processing**: Batch embedding generation
3. **Lazy Loading**: Embedding model loaded on demand
4. **Connection Pooling**: Reuse database connections
5. **Indexing**: HNSW index for fast vector search

---

**Document Status:** Complete  
**Next Review:** January 6, 2026  
**Maintained By:** Development Team

