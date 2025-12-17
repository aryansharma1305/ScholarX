"""Export capabilities - BibTeX, CSV, JSON, Markdown."""
from typing import List, Dict, Optional
from config.chroma_client import get_collection
from utils.logger import get_logger
from pathlib import Path
import json
import csv
from datetime import datetime

logger = get_logger(__name__)


def export_to_bibtex(paper_ids: Optional[List[str]] = None, filename: str = "papers.bib") -> str:
    """
    Export papers to BibTeX format.
    
    Args:
        paper_ids: List of paper IDs to export (if None, exports all)
        filename: Output filename
        
    Returns:
        Path to exported file
    """
    collection = get_collection()
    
    try:
        if paper_ids:
            papers = []
            for pid in paper_ids:
                chunks = collection.get(where={"paper_id": pid}, limit=1)
                if chunks.get("metadatas"):
                    papers.append(chunks["metadatas"][0])
        else:
            all_data = collection.get(limit=10000)
            metadatas = all_data.get("metadatas", [])
            # Deduplicate by paper_id
            seen = set()
            papers = []
            for meta in metadatas:
                pid = meta.get("paper_id")
                if pid and pid not in seen:
                    seen.add(pid)
                    papers.append(meta)
        
        bibtex_entries = []
        for i, paper in enumerate(papers):
            # Generate BibTeX entry
            entry_type = "article"  # Default
            entry_id = f"paper_{paper.get('paper_id', i)}"
            
            title = paper.get("title", "Untitled").replace("{", "").replace("}", "")
            authors = paper.get("authors", "Unknown")
            if isinstance(authors, list):
                authors = " and ".join(authors)
            year = paper.get("year", "")
            journal = paper.get("source", "arXiv")
            
            bibtex = f"""@article{{{entry_id},
    title = {{{title}}},
    author = {{{authors}}},
    year = {{{year}}},
    journal = {{{journal}}},
    url = {{{paper.get('pdf_url', '')}}}
}}"""
            bibtex_entries.append(bibtex)
        
        output_path = Path(filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(bibtex_entries))
        
        logger.info(f"Exported {len(bibtex_entries)} papers to {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error exporting to BibTeX: {e}")
        raise


def export_to_csv(paper_ids: Optional[List[str]] = None, filename: str = "papers.csv") -> str:
    """
    Export papers to CSV format.
    
    Args:
        paper_ids: List of paper IDs to export (if None, exports all)
        filename: Output filename
        
    Returns:
        Path to exported file
    """
    collection = get_collection()
    
    try:
        if paper_ids:
            papers = []
            for pid in paper_ids:
                chunks = collection.get(where={"paper_id": pid}, limit=1)
                if chunks.get("metadatas"):
                    papers.append(chunks["metadatas"][0])
        else:
            all_data = collection.get(limit=10000)
            metadatas = all_data.get("metadatas", [])
            # Deduplicate
            seen = set()
            papers = []
            for meta in metadatas:
                pid = meta.get("paper_id")
                if pid and pid not in seen:
                    seen.add(pid)
                    papers.append(meta)
        
        output_path = Path(filename)
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "paper_id", "title", "authors", "year", "abstract", "source", "pdf_url", "doi"
            ])
            writer.writeheader()
            
            for paper in papers:
                authors = paper.get("authors", "Unknown")
                if isinstance(authors, list):
                    authors = "; ".join(authors)
                
                writer.writerow({
                    "paper_id": paper.get("paper_id", ""),
                    "title": paper.get("title", ""),
                    "authors": authors,
                    "year": paper.get("year", ""),
                    "abstract": paper.get("abstract", "")[:500],  # Truncate long abstracts
                    "source": paper.get("source", ""),
                    "pdf_url": paper.get("pdf_url", ""),
                    "doi": paper.get("doi", "")
                })
        
        logger.info(f"Exported {len(papers)} papers to {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")
        raise


def export_to_json(paper_ids: Optional[List[str]] = None, filename: str = "papers.json") -> str:
    """
    Export papers to JSON format.
    
    Args:
        paper_ids: List of paper IDs to export (if None, exports all)
        filename: Output filename
        
    Returns:
        Path to exported file
    """
    collection = get_collection()
    
    try:
        if paper_ids:
            papers = []
            for pid in paper_ids:
                chunks = collection.get(where={"paper_id": pid}, limit=1)
                if chunks.get("metadatas"):
                    papers.append(chunks["metadatas"][0])
        else:
            all_data = collection.get(limit=10000)
            metadatas = all_data.get("metadatas", [])
            # Deduplicate
            seen = set()
            papers = []
            for meta in metadatas:
                pid = meta.get("paper_id")
                if pid and pid not in seen:
                    seen.add(pid)
                    papers.append(meta)
        
        output_path = Path(filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(papers)} papers to {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error exporting to JSON: {e}")
        raise


def export_to_markdown(paper_ids: Optional[List[str]] = None, filename: str = "papers.md") -> str:
    """
    Export papers to Markdown format.
    
    Args:
        paper_ids: List of paper IDs to export (if None, exports all)
        filename: Output filename
        
    Returns:
        Path to exported file
    """
    collection = get_collection()
    
    try:
        if paper_ids:
            papers = []
            for pid in paper_ids:
                chunks = collection.get(where={"paper_id": pid}, limit=1)
                if chunks.get("metadatas"):
                    papers.append(chunks["metadatas"][0])
        else:
            all_data = collection.get(limit=10000)
            metadatas = all_data.get("metadatas", [])
            # Deduplicate
            seen = set()
            papers = []
            for meta in metadatas:
                pid = meta.get("paper_id")
                if pid and pid not in seen:
                    seen.add(pid)
                    papers.append(meta)
        
        output_path = Path(filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Research Papers Library\n\n")
            f.write(f"*Exported on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            f.write(f"Total papers: {len(papers)}\n\n")
            f.write("---\n\n")
            
            for i, paper in enumerate(papers, 1):
                authors = paper.get("authors", "Unknown")
                if isinstance(authors, list):
                    authors = ", ".join(authors)
                
                f.write(f"## {i}. {paper.get('title', 'Untitled')}\n\n")
                f.write(f"**Authors:** {authors}\n\n")
                f.write(f"**Year:** {paper.get('year', 'Unknown')}\n\n")
                f.write(f"**Source:** {paper.get('source', 'Unknown')}\n\n")
                
                if paper.get("abstract"):
                    f.write(f"**Abstract:**\n\n{paper['abstract']}\n\n")
                
                if paper.get("pdf_url"):
                    f.write(f"**PDF:** [{paper['pdf_url']}]({paper['pdf_url']})\n\n")
                
                f.write("---\n\n")
        
        logger.info(f"Exported {len(papers)} papers to {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error exporting to Markdown: {e}")
        raise


def export_rag_session(query: str, answer: str, citations: List[Dict], filename: str = "rag_session.md") -> str:
    """
    Export a RAG session to Markdown.
    
    Args:
        query: User query
        answer: Generated answer
        citations: List of citation dictionaries
        filename: Output filename
        
    Returns:
        Path to exported file
    """
    output_path = Path(filename)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# RAG Session Export\n\n")
            f.write(f"*Exported on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            f.write("## Query\n\n")
            f.write(f"{query}\n\n")
            f.write("## Answer\n\n")
            f.write(f"{answer}\n\n")
            f.write("## Citations\n\n")
            
            for i, citation in enumerate(citations, 1):
                f.write(f"### Citation {i}\n\n")
                f.write(f"**Paper:** {citation.get('title', 'Unknown')}\n\n")
                f.write(f"**Authors:** {citation.get('authors', 'Unknown')}\n\n")
                f.write(f"**Year:** {citation.get('year', 'Unknown')}\n\n")
                if citation.get("chunk_text"):
                    f.write(f"**Relevant Excerpt:**\n\n{citation['chunk_text'][:300]}...\n\n")
                f.write("---\n\n")
        
        logger.info(f"Exported RAG session to {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error exporting RAG session: {e}")
        raise

