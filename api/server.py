"""HTTP API server for ScholarX."""
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from api.main_api import api
from config.chroma_client import get_collection
from main import query_rag


app = FastAPI(
    title="ScholarX API",
    version="1.0.0",
    description="HTTP API for the ScholarX RAG system."
)


class RAGQueryRequest(BaseModel):
    """Request body for RAG query endpoint."""
    query: str = Field(..., min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)
    fetch_papers: bool = False
    use_enhanced: bool = True
    selected_paper_ids: List[str] = Field(default_factory=list)
    thread_id: Optional[str] = None


@app.get("/health")
def health_check() -> dict:
    """Basic health check endpoint."""
    try:
        collection = get_collection()
        return {
            "status": "ok",
            "service": "scholarx-api",
            "collection_name": collection.name,
            "collection_count": collection.count()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Vector store unavailable: {e}")


@app.get("/papers/{paper_id}")
def get_paper(paper_id: str) -> dict:
    """Get paper metadata by paper ID."""
    result = api.get_paper(paper_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Paper not found: {paper_id}")
    return result


@app.get("/search")
def search_papers(
    query: Optional[str] = None,
    author: Optional[str] = None,
    year: Optional[int] = None,
    limit: int = Query(default=10, ge=1, le=100)
) -> dict:
    """Search papers by query, author, and/or year."""
    return api.search(query=query, author=author, year=year, limit=limit)


@app.post("/rag/query")
def rag_query(request: RAGQueryRequest) -> dict:
    """Run the RAG pipeline and return answer with citations."""
    try:
        return query_rag(
            query=request.query,
            top_k=request.top_k,
            fetch_papers=request.fetch_papers,
            use_enhanced=request.use_enhanced,
            selected_paper_ids=request.selected_paper_ids,
            thread_id=request.thread_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {e}")


@app.get("/intent/classify")
def classify_intent(query: str = Query(..., min_length=3)) -> dict:
    """Classify user query intent."""
    return api.classify_intent(query)


@app.get("/intent/route")
def route_intent(query: str = Query(..., min_length=3)) -> dict:
    """Route a query based on intent."""
    return api.route_query(query)


@app.get("/recommendations")
def get_recommendations(limit: int = Query(default=10, ge=1, le=50)) -> List[dict]:
    """Get paper recommendations from user behavior history."""
    return api.recommend_papers(limit=limit)


@app.get("/trends")
def get_trends(years: Optional[str] = None) -> dict:
    """Analyze topic trends. Pass years as comma-separated list."""
    parsed_years = None
    if years:
        try:
            parsed_years = [int(v.strip()) for v in years.split(",") if v.strip()]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid years format. Use comma-separated integers.")
    return api.analyze_trends(years=parsed_years)
