# ERASER AI Prompts for ScholarX Architecture Diagrams

This document contains ERASER AI prompts for generating all required architecture diagrams for the ScholarX project.

---

## 1. ER Diagram (Entity Relationship Diagram)

### Prompt:

```
Create a comprehensive Entity Relationship Diagram (ERD) for ScholarX - a Research Paper RAG Pipeline system. 

Entities and Attributes:
1. **Paper** (Entity)
   - paper_id (PK, String, UUID)
   - title (String)
   - abstract (Text)
   - authors (String, comma-separated)
   - year (Integer)
   - doi (String, unique)
   - pdf_url (String)
   - source (String: 'arxiv', 'semantic_scholar', 'crossref', 'openalex', 'manual')
   - arxiv_id (String, nullable)
   - semantic_scholar_id (String, nullable)
   - citation_count (Integer)
   - reference_count (Integer)
   - created_at (Timestamp)
   - updated_at (Timestamp)

2. **Chunk** (Entity)
   - chunk_id (PK, String, UUID)
   - paper_id (FK → Paper)
   - chunk_text (Text)
   - chunk_index (Integer)
   - start_char (Integer)
   - end_char (Integer)
   - embedding_vector (Vector, 384 dimensions)
   - metadata (JSON)

3. **Author** (Entity)
   - author_id (PK, String, UUID)
   - name (String)
   - h_index (Integer, nullable)
   - citation_count (Integer, nullable)
   - affiliation (String, nullable)
   - email (String, nullable)

4. **Paper_Author** (Junction Entity - Many-to-Many)
   - paper_id (FK → Paper)
   - author_id (FK → Author)
   - author_order (Integer)

5. **Citation** (Entity)
   - citation_id (PK, String, UUID)
   - citing_paper_id (FK → Paper)
   - cited_paper_id (FK → Paper)
   - citation_context (Text, nullable)

6. **Query_Log** (Entity)
   - query_id (PK, String, UUID)
   - query_text (Text)
   - answer (Text)
   - papers_used (JSON array of paper_ids)
   - chunks_retrieved (Integer)
   - model_used (String)
   - time_taken (Float, seconds)
   - mode (String: 'concise', 'detailed', 'explain', 'compare', 'survey')
   - created_at (Timestamp)

7. **Topic_Cluster** (Entity)
   - cluster_id (PK, String, UUID)
   - cluster_name (String)
   - cluster_number (Integer)
   - created_at (Timestamp)

8. **Paper_Topic** (Junction Entity - Many-to-Many)
   - paper_id (FK → Paper)
   - cluster_id (FK → Topic_Cluster)
   - confidence_score (Float)

9. **Export_Session** (Entity)
   - export_id (PK, String, UUID)
   - format (String: 'bibtex', 'csv', 'json', 'markdown')
   - paper_ids (JSON array)
   - created_at (Timestamp)

Relationships:
- Paper 1:N Chunk (One paper has many chunks)
- Paper M:N Author (Many papers have many authors, via Paper_Author)
- Paper 1:N Citation (One paper can cite many papers, self-referential)
- Paper 1:N Query_Log (One paper can be used in many queries)
- Paper M:N Topic_Cluster (Many papers belong to many topic clusters, via Paper_Topic)
- Paper 1:N Export_Session (One paper can be in many export sessions)

Include:
- Primary keys (PK) marked
- Foreign keys (FK) with arrows showing relationships
- Cardinality indicators (1, N, M)
- Data types for all attributes
- Optional fields marked as nullable
- Use standard ERD notation with rectangles for entities and diamonds for relationships
- Color code: Papers (blue), Chunks (green), Authors (orange), Citations (red), Logs (purple)
```

---

## 2. Use Case Diagram

### Prompt:

```
Create a comprehensive Use Case Diagram for ScholarX - a Research Paper RAG Pipeline system.

Actors:
1. **Researcher** (Primary Actor)
2. **System Administrator** (Secondary Actor)
3. **External API Services** (External System - ArXiv, Semantic Scholar, Crossref, OpenAlex)

Use Cases (Grouped by Package):

**Package: Paper Management**
- Add Paper from URL
- Add Paper from ArXiv
- Add Paper from Semantic Scholar
- Search Papers
- View Paper Details
- Delete Paper
- Export Papers (BibTeX, CSV, JSON, Markdown)
- Manage Paper Collection

**Package: Query & Search**
- Ask Question (RAG Query)
- Semantic Search
- Keyword Search
- Hybrid Search
- Advanced RAG Modes:
  - Concise Answer
  - Detailed Answer
  - Explain Simply
  - Compare Papers
  - Literature Survey
  - Multi-Document Query

**Package: Analysis & Insights**
- View Citation Graph
- Get Paper Summary
- Analyze Research Trends
- Identify Research Gaps
- Get Paper Recommendations
- Cluster Papers by Topic
- Analyze Author Statistics
- View Author Profile

**Package: System Operations**
- View System Statistics
- View Query Logs
- Configure System Settings
- Monitor Performance
- Backup Database
- Restore Database

Relationships:
- Researcher → (all use cases in Paper Management, Query & Search, Analysis & Insights)
- System Administrator → (all use cases in System Operations)
- External API Services → (Add Paper from ArXiv, Add Paper from Semantic Scholar)
- Use "extends" relationships for: Advanced RAG Modes extend Ask Question
- Use "includes" relationships for: Hybrid Search includes Semantic Search and Keyword Search

Include:
- System boundary box labeled "ScholarX System"
- All actors outside the boundary
- Use cases grouped in packages with different colors
- Include/extend relationships shown with dashed arrows
- Association relationships shown with solid lines
- Use standard UML use case notation
```

---

## 3. Class Diagram

### Prompt:

```
Create a comprehensive Class Diagram for ScholarX - a Research Paper RAG Pipeline system using UML notation.

Classes and Attributes:

1. **ScholarXAPI** (Main API Class)
   - Methods: get_paper(), query_rag(), search_papers(), get_citations(), generate_summary(), etc.
   - Relationships: Uses PaperAPI, CitationAPI, SummaryAPI, etc.

2. **Paper** (Data Model)
   - Attributes: paper_id, title, abstract, authors, year, doi, pdf_url, source, citation_count
   - Methods: to_dict(), from_dict(), get_metadata()

3. **Chunk** (Data Model)
   - Attributes: chunk_id, paper_id, chunk_text, chunk_index, embedding_vector, metadata
   - Methods: to_dict(), get_embedding()

4. **RAGPipeline** (Core RAG Class)
   - Attributes: retriever, generator, reranker, query_expander
   - Methods: run_pipeline(), retrieve_context(), generate_answer()
   - Relationships: Uses Retriever, Generator, Reranker

5. **Retriever** (Retrieval Class)
   - Attributes: collection, embedding_model, top_k
   - Methods: retrieve(), hybrid_search(), semantic_search(), keyword_search()

6. **Generator** (Answer Generation Class)
   - Attributes: llm_provider, system_prompt, template
   - Methods: generate_answer(), format_response()

7. **HybridSearch** (Search Strategy Class)
   - Attributes: semantic_weight, keyword_weight
   - Methods: search(), combine_scores()

8. **Reranker** (Re-ranking Class)
   - Attributes: reranking_model, diversity_threshold
   - Methods: rerank(), ensure_diversity()

9. **IngestionPipeline** (Ingestion Class)
   - Attributes: pdf_loader, metadata_fetcher, text_cleaner
   - Methods: ingest_pdf(), fetch_metadata(), clean_text()

10. **PDFLoader** (PDF Processing Class)
    - Attributes: pdf_url, extraction_method
    - Methods: load_pdf(), extract_text(), extract_metadata()

11. **Chunker** (Text Chunking Class)
    - Attributes: chunk_size, chunk_overlap, strategy
    - Methods: chunk_text(), smart_chunk(), basic_chunk()

12. **EmbeddingGenerator** (Embedding Class)
    - Attributes: model_name, provider, model
    - Methods: generate_embedding(), batch_embed()

13. **ChromaDBClient** (Database Client)
    - Attributes: client, collection, persist_directory
    - Methods: get_collection(), upsert(), query(), delete()

14. **PaperFetcher** (External API Class)
    - Attributes: arxiv_client, semantic_scholar_client, crossref_client
    - Methods: fetch_from_arxiv(), fetch_from_semantic_scholar(), search_papers()

15. **CitationAPI** (Feature API)
    - Methods: get_citation_info(), find_citing_papers(), build_citation_graph()

16. **SummaryAPI** (Feature API)
    - Methods: generate_summary(), generate_short_summary(), generate_bullet_summary()

17. **TrendAnalysis** (Analytics Class)
    - Methods: analyze_trends(), get_field_popularity(), predict_future_trends()

18. **ResearchGapIdentifier** (Analytics Class)
    - Methods: identify_gaps(), find_underexplored_combinations(), suggest_directions()

19. **QueryLogger** (Logging Class)
    - Attributes: log_file, query_history
    - Methods: log_query(), get_query_stats(), get_history()

20. **Settings** (Configuration Class)
    - Attributes: embedding_provider, llm_provider, chunk_size, top_k
    - Methods: validate(), load_from_env()

Relationships:
- ScholarXAPI → uses → PaperAPI, CitationAPI, SummaryAPI, RAGPipeline
- RAGPipeline → uses → Retriever, Generator, Reranker, HybridSearch
- Retriever → uses → ChromaDBClient, EmbeddingGenerator
- IngestionPipeline → uses → PDFLoader, Chunker, EmbeddingGenerator
- PDFLoader → creates → Paper
- Chunker → creates → Chunk
- ChromaDBClient → stores → Chunk, Paper (metadata)
- PaperFetcher → creates → Paper

Include:
- All classes with attributes and methods
- Visibility markers: + (public), - (private), # (protected)
- Relationships: Association (→), Composition (◆→), Aggregation (◇→), Inheritance (▷)
- Use different colors for different layers: API (blue), Core (green), Data (yellow), External (red)
- Show method signatures with parameters and return types
```

---

## 4. Data Flow Diagram (DFD) - Level 0 (Context Diagram)

### Prompt:

```
Create a Level 0 (Context) Data Flow Diagram for ScholarX - a Research Paper RAG Pipeline system.

External Entities:
1. **Researcher** (User)
2. **ArXiv API** (External System)
3. **Semantic Scholar API** (External System)
4. **Crossref API** (External System)
5. **OpenAlex API** (External System)

Process:
- **ScholarX System** (Single process bubble in center)

Data Flows:
From Researcher to ScholarX:
- Query Request
- Paper URL
- Search Request
- Export Request
- Configuration Settings

From ScholarX to Researcher:
- Answer with Citations
- Paper Results
- Search Results
- Exported Files
- System Statistics
- Error Messages

From External APIs to ScholarX:
- Paper Metadata (from ArXiv, Semantic Scholar, Crossref, OpenAlex)
- PDF Files
- Citation Data
- Author Information

From ScholarX to External APIs:
- Search Queries
- Paper ID Requests
- API Keys

Include:
- Single process (ScholarX System) in center
- All external entities around the perimeter
- Data flows labeled with descriptive names
- Use standard DFD notation: circles for processes, rectangles for external entities, arrows for data flows
- Color code: External entities (blue), Process (green), Data flows (black)
```

---

## 5. Data Flow Diagram (DFD) - Level 1

### Prompt:

```
Create a Level 1 Data Flow Diagram for ScholarX - a Research Paper RAG Pipeline system, decomposing the main process into sub-processes.

External Entities:
1. **Researcher**
2. **ArXiv API**
3. **Semantic Scholar API**
4. **Crossref API**
5. **OpenAlex API**

Data Stores:
1. **Paper Database** (D1)
2. **Vector Store** (D2 - ChromaDB)
3. **Query Logs** (D3)
4. **Cache** (D4)

Processes:
1. **1.0 Process Query** (Query handling and routing)
2. **2.0 Retrieve Context** (Vector search and retrieval)
3. **3.0 Generate Answer** (Answer generation)
4. **4.0 Ingest Papers** (Paper ingestion pipeline)
5. **5.0 Process Text** (Text chunking and embedding)
6. **6.0 Analyze Data** (Trends, gaps, recommendations)
7. **7.0 Manage Collection** (Paper management operations)
8. **8.0 Export Data** (Export functionality)

Data Flows:
- Researcher → 1.0: Query Request
- 1.0 → 2.0: Search Request
- 2.0 → D2: Query Vectors
- D2 → 2.0: Retrieved Chunks
- 2.0 → 3.0: Context Chunks
- 3.0 → Researcher: Answer with Citations
- Researcher → 4.0: Paper URL/ID
- 4.0 → External APIs: API Requests
- External APIs → 4.0: Paper Metadata
- 4.0 → 5.0: Raw Text
- 5.0 → D2: Embeddings and Chunks
- 5.0 → D1: Paper Metadata
- Researcher → 6.0: Analysis Request
- 6.0 → D1: Read Papers
- 6.0 → Researcher: Analysis Results
- Researcher → 7.0: Management Request
- 7.0 → D1: Paper Operations
- Researcher → 8.0: Export Request
- 8.0 → D1: Read Papers
- 8.0 → Researcher: Exported Files
- 1.0 → D3: Query Log
- 2.0 → D4: Cache Check
- D4 → 2.0: Cached Results

Include:
- All processes numbered (1.0, 2.0, etc.)
- Data stores shown as open rectangles (D1, D2, etc.)
- External entities as rectangles
- All data flows labeled
- Use standard DFD notation
- Show data flow direction with arrows
- Color code: Processes (green), Data stores (yellow), External entities (blue)
```

---

## 6. Component Diagram

### Prompt:

```
Create a comprehensive Component Diagram for ScholarX - a Research Paper RAG Pipeline system showing all major components and their interfaces.

Components:

**User Interface Layer:**
1. **StreamlitApp** (Component)
   - Interface: WebUI
   - Provides: User interface for all operations

2. **CLITools** (Component)
   - Interface: CommandLineInterface
   - Provides: Command-line access

3. **PythonAPI** (Component)
   - Interface: ProgrammaticAPI
   - Provides: Python API for integration

**Application Layer:**
4. **RAGPipeline** (Component)
   - Interface: RAGInterface
   - Requires: RetrieverInterface, GeneratorInterface
   - Provides: Query processing and answer generation

5. **FeatureAPIs** (Component Package)
   - **CitationAPI** (Sub-component)
   - **SummaryAPI** (Sub-component)
   - **AuthorAPI** (Sub-component)
   - **TopicAPI** (Sub-component)
   - **TrendAPI** (Sub-component)
   - **GapAnalysisAPI** (Sub-component)
   - **RecommendationAPI** (Sub-component)
   - **ExportAPI** (Sub-component)

**Core Processing Layer:**
6. **Retriever** (Component)
   - Interface: RetrievalInterface
   - Requires: VectorStoreInterface, EmbeddingInterface
   - Provides: Context retrieval

7. **Generator** (Component)
   - Interface: GenerationInterface
   - Requires: LLMInterface
   - Provides: Answer generation

8. **HybridSearch** (Component)
   - Interface: SearchInterface
   - Provides: Hybrid search capabilities

9. **Reranker** (Component)
   - Interface: RerankingInterface
   - Provides: Result reranking

**Data Processing Layer:**
10. **IngestionPipeline** (Component)
    - Interface: IngestionInterface
    - Requires: PDFLoaderInterface, FetcherInterface
    - Provides: Paper ingestion

11. **PDFLoader** (Component)
    - Interface: PDFLoaderInterface
    - Provides: PDF text extraction

12. **Chunker** (Component)
    - Interface: ChunkingInterface
    - Provides: Text chunking

13. **EmbeddingGenerator** (Component)
    - Interface: EmbeddingInterface
    - Provides: Vector embeddings

**Data Layer:**
14. **ChromaDBClient** (Component)
    - Interface: VectorStoreInterface
    - Provides: Vector storage and retrieval

15. **MetadataStore** (Component)
    - Interface: MetadataInterface
    - Provides: Paper metadata storage

**External Integration Layer:**
16. **ArXivClient** (Component)
    - Interface: ArXivAPI
    - Requires: External ArXiv API
    - Provides: ArXiv paper access

17. **SemanticScholarClient** (Component)
    - Interface: SemanticScholarAPI
    - Requires: External Semantic Scholar API
    - Provides: Semantic Scholar access

18. **CrossrefClient** (Component)
    - Interface: CrossrefAPI
    - Requires: External Crossref API
    - Provides: Crossref access

19. **OpenAlexClient** (Component)
    - Interface: OpenAlexAPI
    - Requires: External OpenAlex API
    - Provides: OpenAlex access

**Utility Layer:**
20. **Logger** (Component)
    - Interface: LoggingInterface
    - Provides: Logging services

21. **Cache** (Component)
    - Interface: CacheInterface
    - Provides: Caching services

22. **Config** (Component)
    - Interface: ConfigInterface
    - Provides: Configuration management

Dependencies:
- StreamlitApp → PythonAPI → FeatureAPIs → RAGPipeline
- RAGPipeline → Retriever, Generator
- Retriever → ChromaDBClient, EmbeddingGenerator
- IngestionPipeline → PDFLoader, Chunker, EmbeddingGenerator
- FeatureAPIs → External API Clients
- All components → Logger, Config

Include:
- All components as rectangles with component icon
- Interfaces shown as "lollipop" (provided) or "socket" (required)
- Dependencies shown with dashed arrows
- Component packages shown with folder icon
- Use standard UML component notation
- Color code by layer: UI (blue), Application (green), Core (yellow), Data (orange), External (red), Utility (purple)
- Show port notation for interfaces
```

---

## 7. Sequence Diagram - RAG Query Flow

### Prompt:

```
Create a Sequence Diagram showing the complete flow of a RAG query in ScholarX system.

Actors/Lifelines:
1. **Researcher** (Actor)
2. **StreamlitApp** (UI Component)
3. **RAGPipeline** (Core Component)
4. **QueryExpander** (Component)
5. **Retriever** (Component)
6. **HybridSearch** (Component)
7. **ChromaDBClient** (Database)
8. **Reranker** (Component)
9. **Generator** (Component)
10. **QueryLogger** (Utility)

Sequence Flow:
1. Researcher → StreamlitApp: Enter Query
2. StreamlitApp → RAGPipeline: process_query(query)
3. RAGPipeline → QueryExpander: expand_query(query)
4. QueryExpander → RAGPipeline: expanded_queries
5. RAGPipeline → Retriever: retrieve_context(query, top_k)
6. Retriever → HybridSearch: hybrid_search(query)
7. HybridSearch → EmbeddingGenerator: generate_embedding(query)
8. EmbeddingGenerator → HybridSearch: query_embedding
9. HybridSearch → ChromaDBClient: vector_search(query_embedding)
10. ChromaDBClient → HybridSearch: candidate_chunks
11. HybridSearch → Retriever: search_results
12. Retriever → Reranker: rerank(candidate_chunks, query)
13. Reranker → Retriever: ranked_chunks
14. Retriever → RAGPipeline: context_chunks
15. RAGPipeline → Generator: generate_answer(query, context_chunks)
16. Generator → RAGPipeline: answer_with_citations
17. RAGPipeline → QueryLogger: log_query(query, answer, papers_used)
18. RAGPipeline → StreamlitApp: response
19. StreamlitApp → Researcher: Display Answer with Citations

Include:
- All lifelines as vertical dashed lines
- Activation boxes on lifelines during method execution
- Messages labeled with method names
- Return messages shown with dashed arrows
- Self-calls shown with loops
- Time flows downward
- Use standard UML sequence diagram notation
- Color code: User interactions (blue), Core processing (green), Data operations (yellow), Logging (purple)
- Add notes for important steps
- Show alternative flows with alt/opt frames if needed
```

---

## 8. Sequence Diagram - Paper Ingestion Flow

### Prompt:

```
Create a Sequence Diagram showing the complete flow of paper ingestion in ScholarX system.

Actors/Lifelines:
1. **Researcher** (Actor)
2. **StreamlitApp** (UI Component)
3. **IngestionPipeline** (Core Component)
4. **PaperFetcher** (Component)
5. **ArXivAPI** (External System)
6. **SemanticScholarAPI** (External System)
7. **PDFLoader** (Component)
8. **TextCleaner** (Component)
9. **Chunker** (Component)
10. **EmbeddingGenerator** (Component)
11. **ChromaDBClient** (Database)
12. **MetadataStore** (Database)

Sequence Flow:
1. Researcher → StreamlitApp: Add Paper (URL/ID/Search)
2. StreamlitApp → IngestionPipeline: ingest_paper(source, identifier)
3. IngestionPipeline → PaperFetcher: fetch_paper_metadata(source, identifier)
4. PaperFetcher → ArXivAPI: get_paper_metadata(arxiv_id) [if ArXiv]
5. ArXivAPI → PaperFetcher: paper_metadata
6. PaperFetcher → SemanticScholarAPI: get_paper_details(paper_id) [if Semantic Scholar]
7. SemanticScholarAPI → PaperFetcher: enhanced_metadata
8. PaperFetcher → IngestionPipeline: complete_metadata
9. IngestionPipeline → PDFLoader: load_pdf(pdf_url)
10. PDFLoader → IngestionPipeline: raw_text
11. IngestionPipeline → TextCleaner: clean_text(raw_text)
12. TextCleaner → IngestionPipeline: cleaned_text
13. IngestionPipeline → Chunker: chunk_text(cleaned_text)
14. Chunker → IngestionPipeline: text_chunks
15. IngestionPipeline → EmbeddingGenerator: generate_embeddings(text_chunks)
16. EmbeddingGenerator → IngestionPipeline: embeddings
17. IngestionPipeline → ChromaDBClient: upsert_chunks(chunks, embeddings, metadata)
18. ChromaDBClient → IngestionPipeline: success
19. IngestionPipeline → MetadataStore: save_paper_metadata(paper_metadata)
20. MetadataStore → IngestionPipeline: paper_id
21. IngestionPipeline → StreamlitApp: ingestion_complete(paper_id)
22. StreamlitApp → Researcher: Paper Added Successfully

Include:
- All lifelines as vertical dashed lines
- Activation boxes showing method execution
- Messages with method names and parameters
- Return messages with dashed arrows
- Parallel processing shown with par frames (chunking and embedding can be parallel)
- Error handling with opt frames
- Use standard UML sequence diagram notation
- Color code: User actions (blue), Ingestion (green), External APIs (red), Processing (yellow), Storage (orange)
- Add timing constraints if relevant
- Show loops for batch processing
```

---

## 9. Deployment Diagram

### Prompt:

```
Create a comprehensive Deployment Diagram for ScholarX - a Research Paper RAG Pipeline system showing the deployment architecture.

Nodes (Deployment Targets):

**Development Environment:**
1. **Developer Machine** (Node)
   - Artifacts: Python Application, ChromaDB, Streamlit Server
   - Components: All application components
   - Properties: OS: macOS/Linux/Windows, Python 3.10+

**Production Environment:**
2. **Web Server** (Node)
   - Artifacts: Streamlit Application, Python Runtime
   - Components: StreamlitApp, PythonAPI, FeatureAPIs
   - Properties: OS: Linux, Python 3.10+, Port: 8501

3. **Application Server** (Node)
   - Artifacts: Core Application, RAG Pipeline
   - Components: RAGPipeline, Retriever, Generator, IngestionPipeline
   - Properties: OS: Linux, Python 3.10+, CPU: 4+ cores, RAM: 8GB+

4. **Vector Database Server** (Node)
   - Artifacts: ChromaDB
   - Components: ChromaDBClient, VectorStore
   - Properties: OS: Linux, Storage: SSD, RAM: 4GB+

5. **Metadata Database** (Node)
   - Artifacts: SQLite/PostgreSQL
   - Components: MetadataStore, QueryLogger
   - Properties: OS: Linux, Storage: HDD/SSD

6. **Cache Server** (Node - Optional)
   - Artifacts: Redis/Memcached
   - Components: Cache
   - Properties: OS: Linux, RAM: 2GB+

**External Services:**
7. **ArXiv API** (Node - External)
   - Properties: Cloud Service, HTTPS

8. **Semantic Scholar API** (Node - External)
   - Properties: Cloud Service, HTTPS, API Key Required

9. **Crossref API** (Node - External)
   - Properties: Cloud Service, HTTPS

10. **OpenAlex API** (Node - External)
    - Properties: Cloud Service, HTTPS

Communication Links:
- Developer Machine → Web Server: HTTP/HTTPS (Development)
- Web Server → Application Server: Internal API Calls
- Application Server → Vector Database Server: ChromaDB Protocol
- Application Server → Metadata Database: SQL Protocol
- Application Server → Cache Server: Redis Protocol (if used)
- Application Server → External APIs: HTTPS
- Web Server → Researcher (Browser): HTTP/HTTPS, WebSocket

Deployment Specifications:
- Show nodes as 3D boxes
- Artifacts shown as rectangles within nodes
- Components shown as smaller rectangles
- Communication links shown as lines with protocol labels
- Use standard UML deployment diagram notation
- Color code: Internal nodes (green), External services (red), Development (blue)
- Show network boundaries with dashed boxes
- Include load balancer if multiple instances
- Show firewall between external and internal nodes
```

---

## 10. Architecture Choice: Microservices vs Monolithic

### Current Architecture: Modular Monolithic

**Selected Architecture:** Modular Monolithic Architecture

**Justification:**
1. **Project Scale**: ScholarX is a medium-scale application suitable for monolithic architecture
2. **Team Size**: Small development team benefits from simpler deployment
3. **Performance**: Direct in-process calls are faster than network calls
4. **Complexity**: Microservices would add unnecessary complexity for current requirements
5. **Resource Constraints**: Single server deployment is more cost-effective
6. **Development Speed**: Faster development and testing in monolithic structure

**Future Migration Path:**
- Can evolve to microservices if scale requires it
- Current modular design facilitates future decomposition
- Clear service boundaries already defined in code structure

---

## 11. Data Exchange Contracts

### 11.1 Internal Data Exchange

**Format:** Python Objects (Dict/List), JSON for serialization

**Frequency:**
- Query Processing: Real-time (on-demand)
- Paper Ingestion: Batch or on-demand (7-26 seconds per paper)
- Cache Updates: On-demand with TTL
- Logging: Synchronous (every query)

**Modes:**
- Direct function calls (in-process)
- JSON serialization for API responses
- File-based storage for exports

### 11.2 External API Data Exchange

**ArXiv API:**
- Mode: REST API (HTTPS)
- Format: XML/Atom Feed
- Frequency: On-demand, rate limit: 3s delay recommended
- Contract: Standard ArXiv API v1

**Semantic Scholar API:**
- Mode: REST API (HTTPS)
- Format: JSON
- Frequency: On-demand, rate limit: 100 req/5min (free tier)
- Contract: Semantic Scholar Graph API v1

**Crossref API:**
- Mode: REST API (HTTPS)
- Format: JSON
- Frequency: On-demand, rate limit: Polite (with User-Agent)
- Contract: Crossref REST API

**OpenAlex API:**
- Mode: REST API (HTTPS)
- Format: JSON
- Frequency: On-demand, rate limit: Polite
- Contract: OpenAlex REST API

### 11.3 Data Sets

**Input Data:**
- Research Papers (PDF files, 1-50 MB each)
- Paper Metadata (JSON, ~5-10 KB per paper)
- User Queries (Text, < 1000 characters)

**Output Data:**
- Answers with Citations (Text + JSON, ~1-5 KB)
- Search Results (JSON array, ~10-50 KB)
- Exported Files (BibTeX, CSV, JSON, Markdown)
- Analytics Results (JSON, ~5-20 KB)

**Storage:**
- Vector Embeddings: 384 dimensions × 4 bytes = 1.5 KB per chunk
- Paper Metadata: ~5-10 KB per paper
- Query Logs: ~1 KB per query

---

## Usage Instructions

1. Copy each prompt into ERASER AI
2. Adjust colors, styles, and details as needed
3. Export diagrams in high resolution (PNG/SVG)
4. Include diagrams in architecture document
5. Add legends and notes as required

---

**Document Version:** 1.0  
**Last Updated:** December 2025

